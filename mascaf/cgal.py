"""
Optional CGAL-backed native preprocessing for MASCAF.

Provides Python wrappers around three compiled C++ executables:

* ``mesh_repair`` — attempt to produce a watertight mesh.
* ``mesh_simplify`` — reduce mesh face count.
* ``mesh_skeletonize`` — extract a mean-curvature skeleton as a polylines file.

The executables are built separately from the ``cpp/`` directory using CMake
and vcpkg. See :doc:`/guide/skeletonization` for setup instructions.

Classes
-------
CGALConfig
    Locate executables and configure CMake.
CGALBuilder
    Run CMake configure/build and discover executables.
CGALOperator
    High-level Python methods for each CGAL operation.
CGALCommandResult
    Immutable record of a completed CGAL subprocess invocation.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CPP_DIR = _REPO_ROOT / "cpp"
_DEFAULT_BUILD_CONFIGS = (
    "Release",
    "Debug",
    "RelWithDebInfo",
    "MinSizeRel",
)
_EXECUTABLE_SUFFIX = ".exe" if os.name == "nt" else ""


class CGALError(RuntimeError):
    """Base exception for all CGAL integration errors."""


class CGALExecutableNotFoundError(CGALError):
    """Raised when a required CGAL executable cannot be located."""


class CGALBuildError(CGALError):
    """Raised when a CMake configure or build step fails."""


@dataclass(frozen=True)
class CGALCommandResult:
    """Immutable record of a completed CGAL executable invocation.

    Attributes
    ----------
    operation : str
        Name of the CGAL operation (e.g. ``"mesh_skeletonize"``).
    command : tuple[str, ...]
        Full command line that was executed.
    input_path : Path
        Path to the input file passed to the executable.
    output_path : Path
        Path to the output file produced by the executable.
    stdout : str
        Captured standard output from the process.
    stderr : str
        Captured standard error from the process.
    returncode : int
        Exit code returned by the process.
    """

    operation: str
    command: tuple[str, ...]
    input_path: Path
    output_path: Path
    stdout: str
    stderr: str
    returncode: int


@dataclass
class CGALConfig:
    """Configuration for locating and invoking CGAL executables and CMake.

    Attributes
    ----------
    cpp_dir : Path
        Root of the native C++ source tree (default: ``<repo>/cpp/``).
    executable_dir : Path or None
        Explicit directory to search first for compiled executables.
        If ``None``, the search falls back to build subdirectories and
        the ``MASCAF_CGAL_BIN_DIR`` environment variable.
    build_dir : Path or None
        CMake build directory. If ``None``, resolved from
        ``MASCAF_CGAL_BUILD_DIR`` or ``<cpp_dir>/build``.
    cmake_executable : str
        Name or full path of the ``cmake`` executable (default: ``"cmake"``).
    vcpkg_toolchain_file : Path or None
        Path to the vcpkg CMake toolchain file. Auto-detected from
        ``VCPKG_ROOT`` if not set.
    build_configs : tuple[str, ...]
        CMake build configurations to search when looking for executables
        (default: Release, Debug, RelWithDebInfo, MinSizeRel).
    executable_suffix : str
        Platform-specific suffix appended to executable names
        (``".exe"`` on Windows, ``""`` elsewhere).
    """

    cpp_dir: Path = field(default_factory=lambda: _DEFAULT_CPP_DIR)
    executable_dir: Path | None = None
    build_dir: Path | None = None
    cmake_executable: str = "cmake"
    vcpkg_toolchain_file: Path | None = None
    build_configs: tuple[str, ...] = _DEFAULT_BUILD_CONFIGS
    executable_suffix: str = _EXECUTABLE_SUFFIX

    def __post_init__(self) -> None:
        self.cpp_dir = Path(self.cpp_dir)
        if self.executable_dir is not None:
            self.executable_dir = Path(self.executable_dir)
        if self.build_dir is not None:
            self.build_dir = Path(self.build_dir)
        if self.vcpkg_toolchain_file is not None:
            self.vcpkg_toolchain_file = Path(self.vcpkg_toolchain_file)
        else:
            self.vcpkg_toolchain_file = self._default_vcpkg_toolchain_file()

    @classmethod
    def from_overrides(
        cls,
        *,
        cpp_dir: str | Path | None = None,
        executable_dir: str | Path | None = None,
        build_dir: str | Path | None = None,
        cmake_executable: str = "cmake",
        vcpkg_toolchain_file: str | Path | None = None,
    ) -> "CGALConfig":
        """Construct a :class:`CGALConfig` from optional keyword overrides.

        Any argument left as ``None`` falls back to the class default.
        This is the preferred factory when building a config from CLI
        arguments or environment variables.
        """
        return cls(
            cpp_dir=Path(cpp_dir) if cpp_dir is not None else _DEFAULT_CPP_DIR,
            executable_dir=(
                Path(executable_dir) if executable_dir is not None else None
            ),
            build_dir=(Path(build_dir) if build_dir is not None else None),
            cmake_executable=cmake_executable,
            vcpkg_toolchain_file=(
                Path(vcpkg_toolchain_file) if vcpkg_toolchain_file is not None else None
            ),
        )

    def default_build_dir(self) -> Path:
        """Return the effective CMake build directory.

        Priority order:

        1. ``build_dir`` set on this config instance.
        2. ``MASCAF_CGAL_BUILD_DIR`` environment variable.
        3. ``<cpp_dir>/build`` (fallback).
        """
        env_build_dir = os.environ.get("MASCAF_CGAL_BUILD_DIR")
        if self.build_dir is not None:
            return self.build_dir
        if env_build_dir:
            return Path(env_build_dir)
        return self.cpp_dir / "build"

    @staticmethod
    def _default_vcpkg_toolchain_file() -> Path | None:
        """Auto-detect the vcpkg CMake toolchain file from ``VCPKG_ROOT``.

        Returns ``None`` if ``VCPKG_ROOT`` is not set or the expected
        toolchain file does not exist.
        """
        vcpkg_root = os.environ.get("VCPKG_ROOT")
        if not vcpkg_root:
            return None
        toolchain = Path(vcpkg_root) / "scripts" / "buildsystems" / "vcpkg.cmake"
        return toolchain if toolchain.exists() else None


class CGALBuilder:
    """Handles CMake configure/build orchestration and executable discovery.

    :class:`CGALBuilder` knows how to:

    - Construct the ``cmake`` configure and build commands from a
      :class:`CGALConfig`.
    - Search a prioritised list of directories for compiled CGAL
      executables.
    - Run subprocesses and surface errors cleanly.

    Parameters
    ----------
    config : CGALConfig or None
        Configuration to use. If ``None``, a default :class:`CGALConfig`
        is constructed, optionally accepting extra keyword arguments that
        are forwarded to :meth:`CGALConfig.from_overrides`.
    **config_kwargs
        Forwarded to :meth:`CGALConfig.from_overrides` when ``config``
        is ``None``.
    """

    def __init__(
        self,
        config: CGALConfig | None = None,
        **config_kwargs,
    ) -> None:
        self.config = config or CGALConfig.from_overrides(**config_kwargs)

    def executable_filename(self, operation: str) -> str:
        """Return the platform-appropriate filename for a CGAL executable.

        Parameters
        ----------
        operation : str
            Base name of the operation (e.g. ``"mesh_skeletonize"``).

        Returns
        -------
        str
            Filename with the platform suffix applied.
        """
        return f"{operation}{self.config.executable_suffix}"

    def candidate_build_dirs(self) -> list[Path]:
        """Return an ordered list of directories likely to contain build
        output.

        Includes the build root itself plus each configured build-type
        subdirectory (e.g. ``build/Release``).
        """
        build_root = self.config.default_build_dir()
        candidates = [build_root]
        candidates.extend(
            build_root / build_config for build_config in self.config.build_configs
        )
        candidates.extend(
            self.config.cpp_dir / build_config
            for build_config in self.config.build_configs
        )
        return candidates

    def candidate_executable_dirs(self) -> list[Path]:
        """Return a deduplicated, priority-ordered list of directories to
        search for CGAL executables.

        Priority order:

        1. ``CGALConfig.executable_dir`` (if set).
        2. ``MASCAF_CGAL_BIN_DIR`` environment variable (if set).
        3. Each candidate build directory from :meth:`candidate_build_dirs`.
        4. ``CGALConfig.cpp_dir`` itself.
        """
        candidates: list[Path] = []
        if self.config.executable_dir is not None:
            candidates.append(self.config.executable_dir)
        env_bin_dir = os.environ.get("MASCAF_CGAL_BIN_DIR")
        if env_bin_dir:
            candidates.append(Path(env_bin_dir))
        candidates.extend(self.candidate_build_dirs())
        candidates.append(self.config.cpp_dir)

        unique: list[Path] = []
        seen: set[Path] = set()
        for candidate in candidates:
            resolved = candidate.resolve(strict=False)
            if resolved in seen:
                continue
            seen.add(resolved)
            unique.append(candidate)
        return unique

    def resolve_executable(self, operation: str) -> Path:
        """Locate the compiled executable for *operation*.

        Parameters
        ----------
        operation : str
            Base name of the operation (e.g. ``"mesh_skeletonize"``).

        Returns
        -------
        Path
            Absolute path to the first matching executable found.

        Raises
        ------
        CGALExecutableNotFoundError
            If the executable is not found in any candidate directory.
        """
        filename = self.executable_filename(operation)
        logger.debug("Searching for CGAL executable: %s", filename)
        for directory in self.candidate_executable_dirs():
            candidate = directory / filename
            logger.debug("  checking %s", candidate)
            if candidate.exists():
                logger.debug("Found executable at %s", candidate)
                return candidate
        searched = ", ".join(str(path) for path in self.candidate_executable_dirs())
        raise CGALExecutableNotFoundError(
            f"Could not find executable for '{operation}'. Searched: {searched}"
        )

    def configure_command(
        self,
        *,
        build_dir: str | Path | None = None,
        build_type: str = "Release",
        generator: str | None = None,
    ) -> list[str]:
        """Build the ``cmake`` configure command without running it.

        Parameters
        ----------
        build_dir : str or Path or None
            Override the build directory for this call.
        build_type : str
            CMake build type (default: ``"Release"``).
        generator : str or None
            Optional CMake generator string (e.g. ``"Ninja"``).

        Returns
        -------
        list[str]
            The full command as a list of strings, suitable for
            :func:`subprocess.run`.
        """
        build_path = (
            Path(build_dir)
            if build_dir is not None
            else self.config.default_build_dir()
        )
        command = [
            self.config.cmake_executable,
            "-S",
            str(self.config.cpp_dir),
            "-B",
            str(build_path),
            f"-DCMAKE_BUILD_TYPE={build_type}",
        ]
        if generator:
            command.extend(["-G", generator])
        if self.config.vcpkg_toolchain_file is not None:
            command.append(f"-DCMAKE_TOOLCHAIN_FILE={self.config.vcpkg_toolchain_file}")
        return command

    def build_command(
        self,
        *,
        build_dir: str | Path | None = None,
        config: str = "Release",
        target: str | None = None,
    ) -> list[str]:
        """Build the ``cmake --build`` command without running it.

        Parameters
        ----------
        build_dir : str or Path or None
            Override the build directory for this call.
        config : str
            CMake build configuration (default: ``"Release"``).
        target : str or None
            Optional build target (e.g. ``"mesh_skeletonize"``).

        Returns
        -------
        list[str]
            The full command as a list of strings.
        """
        build_path = (
            Path(build_dir)
            if build_dir is not None
            else self.config.default_build_dir()
        )
        command = [
            self.config.cmake_executable,
            "--build",
            str(build_path),
            "--config",
            config,
        ]
        if target:
            command.extend(["--target", target])
        return command

    def run_configure(
        self,
        *,
        build_dir: str | Path | None = None,
        build_type: str = "Release",
        generator: str | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Run the CMake configure step.

        Parameters
        ----------
        build_dir : str or Path or None
            Override the build directory for this call.
        build_type : str
            CMake build type (default: ``"Release"``).
        generator : str or None
            Optional CMake generator string.
        check : bool
            If ``True`` (default), raise :exc:`CGALBuildError` on failure.

        Returns
        -------
        subprocess.CompletedProcess[str]
            The completed process record.
        """
        command = self.configure_command(
            build_dir=build_dir,
            build_type=build_type,
            generator=generator,
        )
        logger.info("Running CMake configure: %s", " ".join(command))
        result = self._run_subprocess(
            command,
            cwd=self.config.cpp_dir,
            error_cls=CGALBuildError,
            check=check,
        )
        if result.returncode == 0:
            logger.info("CMake configure succeeded.")
        else:
            logger.warning("CMake configure exited with code %d.", result.returncode)
        return result

    def run_build(
        self,
        *,
        build_dir: str | Path | None = None,
        config: str = "Release",
        target: str | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Run the CMake build step.

        Parameters
        ----------
        build_dir : str or Path or None
            Override the build directory for this call.
        config : str
            CMake build configuration (default: ``"Release"``).
        target : str or None
            Optional build target to pass to ``--target``.
        check : bool
            If ``True`` (default), raise :exc:`CGALBuildError` on failure.

        Returns
        -------
        subprocess.CompletedProcess[str]
            The completed process record.
        """
        command = self.build_command(
            build_dir=build_dir,
            config=config,
            target=target,
        )
        logger.info("Running CMake build: %s", " ".join(command))
        result = self._run_subprocess(
            command,
            cwd=self.config.cpp_dir,
            error_cls=CGALBuildError,
            check=check,
        )
        if result.returncode == 0:
            logger.info("CMake build succeeded.")
        else:
            logger.warning("CMake build exited with code %d.", result.returncode)
        return result

    @staticmethod
    def _run_subprocess(
        command: Sequence[str],
        *,
        cwd: Path,
        error_cls: type[CGALError],
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        """Run *command* as a subprocess and return the completed process.

        Parameters
        ----------
        command : sequence of str
            The command and arguments to execute.
        cwd : Path
            Working directory for the subprocess.
        error_cls : type[CGALError]
            Exception class to raise on non-zero exit when *check* is
            ``True``.
        check : bool
            If ``True``, raise *error_cls* when the return code is
            non-zero.

        Returns
        -------
        subprocess.CompletedProcess[str]
            The completed process record (stdout/stderr always captured).

        Raises
        ------
        CGALError (or subclass)
            When *check* is ``True`` and the subprocess exits non-zero.
        """
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.stdout:
            logger.debug("subprocess stdout: %s", completed.stdout.strip())
        if completed.stderr:
            logger.debug("subprocess stderr: %s", completed.stderr.strip())
        if check and completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            details = stderr or stdout or "subprocess failed without output"
            raise error_cls(
                "Command failed "
                f"({completed.returncode}): {' '.join(command)}\n{details}"
            )
        return completed


class CGALOperator:
    """High-level interface for running CGAL mesh operations.

    :class:`CGALOperator` wraps the three native CGAL executables
    (``mesh_repair``, ``mesh_simplify``, ``mesh_skeletonize``) and
    exposes them as Python methods. Each method resolves the executable,
    assembles the argument list, runs the process, and returns a
    :class:`CGALCommandResult`.

    Parameters
    ----------
    config : CGALConfig or None
        Configuration used to locate executables and set up CMake.
        If ``None``, a default config is constructed.
    builder : CGALBuilder or None
        Provide a pre-built :class:`CGALBuilder` instead of constructing
        one from *config*.
    **config_kwargs
        Forwarded to :meth:`CGALConfig.from_overrides` when neither
        *config* nor *builder* is supplied.
    """

    def __init__(
        self,
        config: CGALConfig | None = None,
        builder: CGALBuilder | None = None,
        **config_kwargs,
    ) -> None:
        if builder is not None:
            self.builder = builder
            self.config = builder.config
        else:
            self.config = config or CGALConfig.from_overrides(**config_kwargs)
            self.builder = CGALBuilder(self.config)

    def repair(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> CGALCommandResult:
        """Run CGAL mesh repair on *input_path*, writing result to
        *output_path*.

        Mesh repair fills holes and attempts to produce a valid,
        watertight triangle mesh suitable for skeletonization.

        Parameters
        ----------
        input_path : str or Path
            Path to the input mesh file (OBJ recommended).
        output_path : str or Path
            Path where the repaired mesh will be written.

        Returns
        -------
        CGALCommandResult
            Record of the completed operation.
        """
        logger.info("Repairing mesh: %s -> %s", input_path, output_path)
        result = self._run_operation("mesh_repair", [input_path, output_path])
        logger.info("Mesh repair complete.")
        return result

    def simplify(
        self,
        input_path: str | Path,
        output_path: str | Path,
        target: int | float,
    ) -> CGALCommandResult:
        """Run CGAL mesh simplification on *input_path*.

        Parameters
        ----------
        input_path : str or Path
            Path to the input mesh file.
        output_path : str or Path
            Path where the simplified mesh will be written.
        target : int or float
            Simplification target. Values ``>= 1`` are treated as an
            absolute face count; values in ``(0, 1)`` are treated as a
            ratio of the original face count.

        Returns
        -------
        CGALCommandResult
            Record of the completed operation.

        Raises
        ------
        ValueError
            If *target* is not a positive number.
        """
        target_value = float(target)
        if not (0 < target_value < 1 or target_value >= 1):
            raise ValueError("target must be >= 1 or between 0 and 1")
        target_arg = (
            str(int(target_value))
            if target_value >= 1 and target_value.is_integer()
            else str(target)
        )
        logger.info(
            "Simplifying mesh: %s -> %s (target=%s)",
            input_path,
            output_path,
            target_arg,
        )
        result = self._run_operation(
            "mesh_simplify",
            [input_path, output_path, target_arg],
        )
        logger.info("Mesh simplification complete.")
        return result

    def skeletonize(
        self,
        input_path: str | Path,
        output_path: str | Path,
        quality_speed_tradeoff: float = 0.5,
        medially_centered_speed_tradeoff: float = 5.0,
    ) -> CGALCommandResult:
        """Run CGAL mean-curvature skeleton extraction on *input_path*.

        The output is a ``.polylines.txt`` file compatible with
        :meth:`mascaf.SkeletonGraph.from_txt`.

        Parameters
        ----------
        input_path : str or Path
            Path to the input mesh file (must be a closed triangle mesh).
        output_path : str or Path
            Path where the skeleton polylines file will be written.
        quality_speed_tradeoff : float
            CGAL ``quality_speed_tradeoff`` parameter (``w_H``). Controls
            the quality–speed balance; higher values favour quality.
            Default: ``0.5``.
        medially_centered_speed_tradeoff : float
            CGAL ``medially_centered_speed_tradeoff`` parameter (``w_M``).
            Higher values push the skeleton toward the medial axis.
            Default: ``5.0``.

        Returns
        -------
        CGALCommandResult
            Record of the completed operation.
        """
        logger.info(
            "Skeletonizing mesh: %s -> %s (w_H=%.3f, w_M=%.3f)",
            input_path,
            output_path,
            quality_speed_tradeoff,
            medially_centered_speed_tradeoff,
        )
        result = self._run_operation(
            "mesh_skeletonize",
            [
                input_path,
                output_path,
                str(float(quality_speed_tradeoff)),
                str(float(medially_centered_speed_tradeoff)),
            ],
        )
        logger.info("Skeletonization complete.")
        return result

    def suggest_skeletonization_parameters(
        self,
        input_path: str | Path | None = None,
    ) -> dict[str, float | str]:
        """Return suggested skeletonization parameters for a mesh.

        .. note::
            This method currently returns fixed placeholder values.
            Automatic parameter tuning based on mesh properties is
            planned for a future release.

        Parameters
        ----------
        input_path : str or Path or None
            Path to the mesh file (reserved for future use).

        Returns
        -------
        dict
            Dictionary with keys ``quality_speed_tradeoff``,
            ``medially_centered_speed_tradeoff``, and ``source``.
        """
        return {
            "quality_speed_tradeoff": 0.5,
            "medially_centered_speed_tradeoff": 5.0,
            "source": "placeholder",
        }

    def _run_operation(
        self,
        operation: str,
        args: Sequence[str | Path],
    ) -> CGALCommandResult:
        """Resolve, invoke, and record a single CGAL executable operation.

        Parameters
        ----------
        operation : str
            Base name of the CGAL executable (without suffix).
        args : sequence of str or Path
            Positional arguments passed after the executable path.
            ``args[0]`` must be the input path and ``args[1]`` the output
            path.

        Returns
        -------
        CGALCommandResult
            Immutable record of the invocation.

        Raises
        ------
        CGALExecutableNotFoundError
            If the executable cannot be located.
        CGALError
            If the subprocess exits with a non-zero return code.
        """
        executable = self.builder.resolve_executable(operation)
        resolved_args = [str(arg) for arg in args]
        command = [str(executable), *resolved_args]
        logger.debug("Executing: %s", " ".join(command))
        completed = self.builder._run_subprocess(
            command,
            cwd=self.config.cpp_dir,
            error_cls=CGALError,
            check=True,
        )
        return CGALCommandResult(
            operation=operation,
            command=tuple(command),
            input_path=Path(args[0]),
            output_path=Path(args[1]),
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )


CGALMeshProcessor = CGALOperator


__all__ = [
    "CGALError",
    "CGALBuildError",
    "CGALCommandResult",
    "CGALConfig",
    "CGALBuilder",
    "CGALExecutableNotFoundError",
    "CGALMeshProcessor",
    "CGALOperator",
]

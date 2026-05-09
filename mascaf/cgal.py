from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


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
    pass


class CGALExecutableNotFoundError(CGALError):
    pass


class CGALBuildError(CGALError):
    pass


@dataclass(frozen=True)
class CGALCommandResult:
    operation: str
    command: tuple[str, ...]
    input_path: Path
    output_path: Path
    stdout: str
    stderr: str
    returncode: int


@dataclass
class CGALConfig:
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
        env_build_dir = os.environ.get("MASCAF_CGAL_BUILD_DIR")
        if self.build_dir is not None:
            return self.build_dir
        if env_build_dir:
            return Path(env_build_dir)
        return self.cpp_dir / "build"

    @staticmethod
    def _default_vcpkg_toolchain_file() -> Path | None:
        vcpkg_root = os.environ.get("VCPKG_ROOT")
        if not vcpkg_root:
            return None
        toolchain = Path(vcpkg_root) / "scripts" / "buildsystems" / "vcpkg.cmake"
        return toolchain if toolchain.exists() else None


class CGALBuilder:
    def __init__(
        self,
        config: CGALConfig | None = None,
        **config_kwargs,
    ) -> None:
        self.config = config or CGALConfig.from_overrides(**config_kwargs)

    def executable_filename(self, operation: str) -> str:
        return f"{operation}{self.config.executable_suffix}"

    def candidate_build_dirs(self) -> list[Path]:
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
        filename = self.executable_filename(operation)
        for directory in self.candidate_executable_dirs():
            candidate = directory / filename
            if candidate.exists():
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
        command = self.configure_command(
            build_dir=build_dir,
            build_type=build_type,
            generator=generator,
        )
        return self._run_subprocess(
            command,
            cwd=self.config.cpp_dir,
            error_cls=CGALBuildError,
            check=check,
        )

    def run_build(
        self,
        *,
        build_dir: str | Path | None = None,
        config: str = "Release",
        target: str | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        command = self.build_command(
            build_dir=build_dir,
            config=config,
            target=target,
        )
        return self._run_subprocess(
            command,
            cwd=self.config.cpp_dir,
            error_cls=CGALBuildError,
            check=check,
        )

    @staticmethod
    def _run_subprocess(
        command: Sequence[str],
        *,
        cwd: Path,
        error_cls: type[CGALError],
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
        )
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
        return self._run_operation("mesh_repair", [input_path, output_path])

    def simplify(
        self,
        input_path: str | Path,
        output_path: str | Path,
        target: int | float,
    ) -> CGALCommandResult:
        target_value = float(target)
        if not (0 < target_value < 1 or target_value >= 1):
            raise ValueError("target must be >= 1 or between 0 and 1")
        target_arg = (
            str(int(target_value))
            if target_value >= 1 and target_value.is_integer()
            else str(target)
        )
        return self._run_operation(
            "mesh_simplify",
            [input_path, output_path, target_arg],
        )

    def skeletonize(
        self,
        input_path: str | Path,
        output_path: str | Path,
        quality_speed_tradeoff: float = 0.5,
        medially_centered_speed_tradeoff: float = 5.0,
    ) -> CGALCommandResult:
        return self._run_operation(
            "mesh_skeletonize",
            [
                input_path,
                output_path,
                str(float(quality_speed_tradeoff)),
                str(float(medially_centered_speed_tradeoff)),
            ],
        )

    def suggest_skeletonization_parameters(
        self,
        input_path: str | Path | None = None,
    ) -> dict[str, float | str]:
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
        executable = self.builder.resolve_executable(operation)
        command = [str(executable), *(str(arg) for arg in args)]
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

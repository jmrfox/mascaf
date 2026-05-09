from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CPP_DIR = _REPO_ROOT / "cpp"
_BUILD_CONFIGS = ("Release", "Debug", "RelWithDebInfo", "MinSizeRel")
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


class CGALMeshProcessor:
    def __init__(
        self,
        *,
        cpp_dir: str | Path | None = None,
        executable_dir: str | Path | None = None,
        build_dir: str | Path | None = None,
        cmake_executable: str = "cmake",
        vcpkg_toolchain_file: str | Path | None = None,
    ) -> None:
        self.cpp_dir = Path(cpp_dir) if cpp_dir is not None else _DEFAULT_CPP_DIR
        self.executable_dir = (
            Path(executable_dir) if executable_dir is not None else None
        )
        self.build_dir = Path(build_dir) if build_dir is not None else None
        self.cmake_executable = cmake_executable
        self.vcpkg_toolchain_file = (
            Path(vcpkg_toolchain_file)
            if vcpkg_toolchain_file is not None
            else self._default_vcpkg_toolchain_file()
        )

    @staticmethod
    def executable_filename(operation: str) -> str:
        return f"{operation}{_EXECUTABLE_SUFFIX}"

    @staticmethod
    def _default_vcpkg_toolchain_file() -> Path | None:
        vcpkg_root = os.environ.get("VCPKG_ROOT")
        if not vcpkg_root:
            return None
        toolchain = Path(vcpkg_root) / "scripts" / "buildsystems" / "vcpkg.cmake"
        return toolchain if toolchain.exists() else None

    def default_build_dir(self) -> Path:
        return self.build_dir or Path(
            os.environ.get("MASCAF_CGAL_BUILD_DIR", self.cpp_dir / "build")
        )

    def candidate_build_dirs(self) -> list[Path]:
        build_root = self.default_build_dir()
        candidates = [build_root]
        candidates.extend(build_root / config for config in _BUILD_CONFIGS)
        candidates.extend(self.cpp_dir / config for config in _BUILD_CONFIGS)
        return candidates

    def candidate_executable_dirs(self) -> list[Path]:
        candidates: list[Path] = []
        if self.executable_dir is not None:
            candidates.append(self.executable_dir)
        env_bin_dir = os.environ.get("MASCAF_CGAL_BIN_DIR")
        if env_bin_dir:
            candidates.append(Path(env_bin_dir))
        candidates.extend(self.candidate_build_dirs())
        candidates.append(self.cpp_dir)

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
            Path(build_dir) if build_dir is not None else self.default_build_dir()
        )
        command = [
            self.cmake_executable,
            "-S",
            str(self.cpp_dir),
            "-B",
            str(build_path),
            f"-DCMAKE_BUILD_TYPE={build_type}",
        ]
        if generator:
            command.extend(["-G", generator])
        if self.vcpkg_toolchain_file is not None:
            command.append(f"-DCMAKE_TOOLCHAIN_FILE={self.vcpkg_toolchain_file}")
        return command

    def build_command(
        self,
        *,
        build_dir: str | Path | None = None,
        config: str = "Release",
        target: str | None = None,
    ) -> list[str]:
        build_path = (
            Path(build_dir) if build_dir is not None else self.default_build_dir()
        )
        command = [
            self.cmake_executable,
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
            cwd=self.cpp_dir,
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
            cwd=self.cpp_dir,
            error_cls=CGALBuildError,
            check=check,
        )

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
            "mesh_simplify", [input_path, output_path, target_arg]
        )

    def _run_operation(
        self,
        operation: str,
        args: Sequence[str | Path],
    ) -> CGALCommandResult:
        executable = self.resolve_executable(operation)
        command = [str(executable), *(str(arg) for arg in args)]
        completed = self._run_subprocess(
            command, cwd=self.cpp_dir, error_cls=CGALError, check=True
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


__all__ = [
    "CGALError",
    "CGALBuildError",
    "CGALCommandResult",
    "CGALExecutableNotFoundError",
    "CGALMeshProcessor",
]

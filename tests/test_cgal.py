from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from mascaf.cgal import (
    CGALBuilder,
    CGALBuildError,
    CGALConfig,
    CGALExecutableNotFoundError,
    CGALOperator,
)


def _touch_executable(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_resolve_executable_from_explicit_directory(tmp_path):
    bin_dir = tmp_path / "bin"
    builder = CGALBuilder(CGALConfig(cpp_dir=tmp_path, executable_dir=bin_dir))
    exe_name = builder.executable_filename("mesh_repair")
    exe_path = bin_dir / exe_name
    _touch_executable(exe_path)

    assert builder.resolve_executable("mesh_repair") == exe_path


def test_resolve_executable_searches_build_configs(tmp_path):
    build_dir = tmp_path / "build"
    builder = CGALBuilder(CGALConfig(cpp_dir=tmp_path, build_dir=build_dir))
    exe_name = builder.executable_filename("mesh_simplify")
    exe_path = build_dir / "Release" / exe_name
    _touch_executable(exe_path)

    assert builder.resolve_executable("mesh_simplify") == exe_path


def test_resolve_executable_raises_when_missing(tmp_path):
    builder = CGALBuilder(CGALConfig(cpp_dir=tmp_path))

    with pytest.raises(CGALExecutableNotFoundError):
        builder.resolve_executable("mesh_repair")


def test_configure_command_includes_vcpkg_toolchain(tmp_path):
    toolchain = tmp_path / "vcpkg.cmake"
    toolchain.write_text("", encoding="utf-8")
    builder = CGALBuilder(
        CGALConfig(
            cpp_dir=tmp_path,
            build_dir=tmp_path / "build",
            vcpkg_toolchain_file=toolchain,
        )
    )

    command = builder.configure_command(generator="Ninja")

    assert command[:5] == [
        "cmake",
        "-S",
        str(tmp_path),
        "-B",
        str(tmp_path / "build"),
    ]
    assert "-G" in command
    assert "Ninja" in command
    assert f"-DCMAKE_TOOLCHAIN_FILE={toolchain}" in command


def test_build_command_can_target_specific_binary(tmp_path):
    builder = CGALBuilder(
        CGALConfig(
            cpp_dir=tmp_path,
            build_dir=tmp_path / "build",
        )
    )

    command = builder.build_command(config="Debug", target="mesh_repair")

    expected = [
        "cmake",
        "--build",
        str(tmp_path / "build"),
        "--config",
        "Debug",
        "--target",
        "mesh_repair",
    ]
    assert command == expected


def test_simplify_runs_expected_command(monkeypatch, tmp_path):
    builder = CGALBuilder(CGALConfig(cpp_dir=tmp_path, executable_dir=tmp_path))
    exe_name = builder.executable_filename("mesh_simplify")
    exe_path = tmp_path / exe_name
    _touch_executable(exe_path)
    operator = CGALOperator(builder=builder)
    captured: dict[str, object] = {}

    def fake_run(command, cwd, capture_output, text, check):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["capture_output"] = capture_output
        captured["text"] = text
        captured["check"] = check
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = operator.simplify("input.obj", "output.obj", 0.5)

    expected = [
        str(exe_path),
        "input.obj",
        "output.obj",
        "0.5",
    ]
    assert list(result.command) == expected
    assert captured["command"] == expected
    assert captured["cwd"] == str(tmp_path)
    assert captured["capture_output"]
    assert captured["text"]
    assert not captured["check"]
    assert result.stdout == "ok"


def test_skeletonize_runs_expected_command(monkeypatch, tmp_path):
    builder = CGALBuilder(CGALConfig(cpp_dir=tmp_path, executable_dir=tmp_path))
    exe_name = builder.executable_filename("mesh_skeletonize")
    exe_path = tmp_path / exe_name
    _touch_executable(exe_path)
    operator = CGALOperator(builder=builder)
    captured: dict[str, object] = {}

    def fake_run(command, cwd, capture_output, text, check):
        captured["command"] = command
        captured["cwd"] = cwd
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = operator.skeletonize(
        "input.obj",
        "output.polylines.txt",
        quality_speed_tradeoff=0.25,
        medially_centered_speed_tradeoff=8.0,
    )

    expected = [
        str(exe_path),
        "input.obj",
        "output.polylines.txt",
        "0.25",
        "8.0",
    ]
    assert list(result.command) == expected
    assert captured["command"] == expected
    assert captured["cwd"] == str(tmp_path)


def test_run_build_raises_cgal_build_error(monkeypatch, tmp_path):
    builder = CGALBuilder(
        CGALConfig(
            cpp_dir=tmp_path,
            build_dir=tmp_path / "build",
        )
    )

    def fake_run(command, cwd, capture_output, text, check):
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="build failed",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(
        CGALBuildError,
        match="build failed",
    ):
        builder.run_build()


def test_simplify_rejects_invalid_target(tmp_path):
    operator = CGALOperator(config=CGALConfig(cpp_dir=tmp_path))

    with pytest.raises(
        ValueError,
        match="target must be >= 1 or between 0 and 1",
    ):
        operator.simplify("input.obj", "output.obj", 0)


def test_suggest_skeletonization_parameters_is_placeholder(tmp_path):
    operator = CGALOperator(config=CGALConfig(cpp_dir=tmp_path))

    suggested = operator.suggest_skeletonization_parameters("input.obj")

    assert suggested == {
        "quality_speed_tradeoff": 0.5,
        "medially_centered_speed_tradeoff": 5.0,
        "source": "placeholder",
    }

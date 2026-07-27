"""Tests for mascaf.logging_config."""

from __future__ import annotations

import logging

from mascaf.logging_config import configure_logging


def test_configure_logging_debug_to_memory(tmp_path):
    log_file = tmp_path / "run.log"
    configure_logging(
        "DEBUG",
        log_file=log_file,
        console=False,
        quiet_loggers=(),
    )
    logging.getLogger("mascaf.basis_optimizer").debug("test detail")
    logging.getLogger("mascaf.basis_optimizer").info("test info")

    text = log_file.read_text(encoding="utf-8")
    assert "test detail" in text
    assert "test info" in text
    assert "mascaf.basis_optimizer" in text


def test_configure_logging_quiet_third_party(tmp_path):
    log_file = tmp_path / "run.log"
    configure_logging("DEBUG", log_file=log_file, console=False)
    logging.getLogger("trimesh").debug("should not appear")
    logging.getLogger("mascaf.test").debug("should appear")

    text = log_file.read_text(encoding="utf-8")
    assert "should appear" in text
    assert "should not appear" not in text


def test_configure_logging_none_uses_default_quiet_list(tmp_path):
    """Explicit None must quiet third-party loggers (not disable quieting)."""
    log_file = tmp_path / "run.log"
    configure_logging(
        "DEBUG",
        log_file=log_file,
        console=False,
        quiet_loggers=None,
    )
    logging.getLogger("kaleido").debug("quiet me")
    logging.getLogger("mascaf.test").debug("keep me")

    text = log_file.read_text(encoding="utf-8")
    assert "keep me" in text
    assert "quiet me" not in text

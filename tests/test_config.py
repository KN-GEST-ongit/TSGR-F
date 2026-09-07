from __future__ import annotations

import pytest

from tsgr.config import ConfigurationError, load_config


def test_default_config_is_valid() -> None:
    config = load_config()
    assert config["camera"]["target_fps"] == 30.0
    assert config["camera"]["input_is_mirrored"] is False


def test_fps_above_thirty_is_rejected() -> None:
    with pytest.raises(ConfigurationError):
        load_config(overrides={"camera": {"target_fps": 31}})

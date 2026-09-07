"""
Tests for MyKid configuration system.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from packages.shared.config import (
    MyKidConfig,
    load_config,
    get_config,
    reset_config,
)
from packages.shared.errors import ConfigurationError


class TestMyKidConfigDefaults:
    """Test that defaults are correct when no config file is provided."""

    def test_default_detection_thresholds(self):
        config = MyKidConfig()
        assert config.detection.object_confidence_threshold == 0.5
        assert config.detection.scene_risk_threshold == 0.6

    def test_default_protection(self):
        config = MyKidConfig()
        assert config.protection.blur_padding == 0.1
        assert config.protection.mode == "gaussian"

    def test_default_video(self):
        config = MyKidConfig()
        assert config.video.inference_fps == 8
        assert config.video.temporal_persistence_frames == 10

    def test_default_image(self):
        config = MyKidConfig()
        assert config.image.max_size == 1920

    def test_default_model(self):
        config = MyKidConfig()
        assert config.model.path == "models/"

    def test_default_logging(self):
        config = MyKidConfig()
        assert config.logging.level == "INFO"


class TestLoadConfigFromYAML:
    """Test loading config from YAML files."""

    def test_load_default_config(self, config_path):
        config = load_config(config_path)
        # Values should match configs/default.yaml
        assert config.detection.object_confidence_threshold == 0.5
        assert config.protection.mode == "gaussian"

    def test_load_custom_yaml(self, tmp_output_dir):
        custom_yaml = tmp_output_dir / "custom.yaml"
        custom_yaml.write_text(
            "detection:\n"
            "  object_confidence_threshold: 0.75\n"
            "protection:\n"
            "  mode: pixelate\n"
            "  blur_padding: 0.2\n"
        )
        config = load_config(str(custom_yaml))
        assert config.detection.object_confidence_threshold == 0.75
        assert config.protection.mode == "pixelate"
        assert config.protection.blur_padding == 0.2

    def test_load_nonexistent_file_uses_defaults(self):
        config = load_config("/nonexistent/path/config.yaml")
        # Should not raise, just use defaults
        assert config.detection.object_confidence_threshold == 0.5

    def test_load_empty_yaml(self, tmp_output_dir):
        empty_yaml = tmp_output_dir / "empty.yaml"
        empty_yaml.write_text("")
        config = load_config(str(empty_yaml))
        assert config.detection.object_confidence_threshold == 0.5

    def test_load_invalid_yaml_raises(self, tmp_output_dir):
        bad_yaml = tmp_output_dir / "bad.yaml"
        bad_yaml.write_text("{ invalid: yaml: content: [")
        with pytest.raises(ConfigurationError):
            load_config(str(bad_yaml))


class TestEnvironmentOverrides:
    """Test that environment variables override YAML values."""

    def test_env_override_confidence_threshold(self, config_path):
        os.environ["MYKID_DETECTION_OBJECT_CONFIDENCE_THRESHOLD"] = "0.88"
        try:
            config = load_config(config_path)
            assert config.detection.object_confidence_threshold == 0.88
        finally:
            del os.environ["MYKID_DETECTION_OBJECT_CONFIDENCE_THRESHOLD"]

    def test_env_override_protection_mode(self, config_path):
        os.environ["MYKID_PROTECTION_MODE"] = "cover"
        try:
            config = load_config(config_path)
            assert config.protection.mode == "cover"
        finally:
            del os.environ["MYKID_PROTECTION_MODE"]

    def test_env_override_video_fps(self, config_path):
        os.environ["MYKID_VIDEO_INFERENCE_FPS"] = "15"
        try:
            config = load_config(config_path)
            assert config.video.inference_fps == 15
        finally:
            del os.environ["MYKID_VIDEO_INFERENCE_FPS"]

    def test_env_override_invalid_value_raises(self, config_path):
        os.environ["MYKID_VIDEO_INFERENCE_FPS"] = "not_a_number"
        try:
            with pytest.raises(ConfigurationError):
                load_config(config_path)
        finally:
            del os.environ["MYKID_VIDEO_INFERENCE_FPS"]


class TestConfigSingleton:
    """Test the singleton config pattern."""

    def test_get_config_returns_same_instance(self):
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_reset_clears_singleton(self):
        config1 = get_config()
        reset_config()
        config2 = get_config()
        assert config1 is not config2

    def test_repr(self):
        config = MyKidConfig()
        repr_str = repr(config)
        assert "MyKidConfig" in repr_str
        assert "detection" in repr_str

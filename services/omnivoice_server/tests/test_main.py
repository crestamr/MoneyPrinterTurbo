import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from services.omnivoice_server import __main__ as main_module


class TestResolveSettings(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.config_path = Path(self.tmp) / "config.toml"

    def _write_config(self, text: str) -> Path:
        # UTF-8 on purpose: the file must not be read with the platform codec.
        self.config_path.write_text(text, encoding="utf-8")
        return self.config_path

    def _resolve(self, argv=None, *, config_path=None, env=None):
        return main_module.resolve_settings(
            argv or [],
            config_path=config_path or self.config_path,
            env={} if env is None else env,
        )

    def test_builtin_defaults_when_config_file_is_missing(self):
        missing = Path(self.tmp) / "does-not-exist.toml"

        launch = self._resolve(config_path=missing)

        self.assertEqual(launch.host, "127.0.0.1")
        self.assertEqual(launch.port, 8890)
        self.assertEqual(launch.settings.voices_dir, os.path.join("storage", "voices"))
        self.assertEqual(launch.settings.model_id, "k2-fsa/OmniVoice")
        self.assertEqual(launch.settings.device, "cuda:0")
        self.assertEqual(launch.settings.num_step, 32)
        self.assertEqual(launch.settings.idle_unload_seconds, 600)
        self.assertEqual(launch.settings.api_key, "")

    def test_omnivoice_table_supplies_defaults(self):
        self._write_config(
            "[omnivoice]\n"
            'base_url = "http://127.0.0.1:8890/v1"\n'
            'api_key = "from-config"\n'
            'model_id = "omnivoice"\n'
            'voices_dir = "D:/my-voices"\n'
            "num_step = 16\n"
            "idle_unload_seconds = 120\n"
        )

        launch = self._resolve()

        self.assertEqual(launch.settings.voices_dir, "D:/my-voices")
        self.assertEqual(launch.settings.num_step, 16)
        self.assertEqual(launch.settings.idle_unload_seconds, 120)
        self.assertEqual(launch.settings.api_key, "from-config")

    def test_client_only_keys_never_become_the_weights_repo(self):
        # [omnivoice] model_id is the client's payload model name, so reading it
        # here would make the service try to load a repo called "omnivoice".
        self._write_config('[omnivoice]\nmodel_id = "omnivoice"\n')

        launch = self._resolve()

        self.assertEqual(launch.settings.model_id, "k2-fsa/OmniVoice")

    def test_empty_string_in_config_falls_back_to_builtin_default(self):
        self._write_config('[omnivoice]\nvoices_dir = ""\napi_key = ""\n')

        launch = self._resolve()

        self.assertEqual(launch.settings.voices_dir, os.path.join("storage", "voices"))
        self.assertEqual(launch.settings.api_key, "")

    def test_cli_flag_overrides_config_file_value(self):
        self._write_config(
            '[omnivoice]\nvoices_dir = "D:/my-voices"\n'
            "num_step = 16\nidle_unload_seconds = 120\n"
            'api_key = "from-config"\n'
        )

        launch = self._resolve(
            [
                "--voices-dir",
                "D:/cli-voices",
                "--num-step",
                "8",
                "--idle-unload-seconds",
                "0",
                "--api-key",
                "from-cli",
                "--port",
                "9999",
            ]
        )

        self.assertEqual(launch.settings.voices_dir, "D:/cli-voices")
        self.assertEqual(launch.settings.num_step, 8)
        self.assertEqual(launch.settings.idle_unload_seconds, 0)
        self.assertEqual(launch.settings.api_key, "from-cli")
        self.assertEqual(launch.port, 9999)

    def test_malformed_config_file_does_not_crash(self):
        self._write_config("[omnivoice\nnum_step = not-toml ::: {{\n")

        launch = self._resolve()

        self.assertEqual(launch.settings.num_step, 32)
        self.assertEqual(launch.settings.idle_unload_seconds, 600)
        self.assertEqual(launch.settings.voices_dir, os.path.join("storage", "voices"))
        self.assertEqual(launch.port, 8890)

    def test_wrongly_typed_config_values_fall_back_to_builtin_defaults(self):
        self._write_config('[omnivoice]\nnum_step = "sixteen"\nidle_unload_seconds = []\n')

        launch = self._resolve()

        self.assertEqual(launch.settings.num_step, 32)
        self.assertEqual(launch.settings.idle_unload_seconds, 600)

    def test_missing_omnivoice_table_uses_builtin_defaults(self):
        self._write_config('[azure]\nspeech_key = ""\n')

        launch = self._resolve()

        self.assertEqual(launch.settings.num_step, 32)
        self.assertEqual(launch.settings.voices_dir, os.path.join("storage", "voices"))

    def test_utf8_config_file_is_decoded_without_the_platform_codec(self):
        # cp1252 would raise UnicodeDecodeError on these bytes and silently lose
        # the whole table through the fallback path.
        self._write_config('[omnivoice]\nvoices_dir = "D:/声音库"\nnum_step = 16\n')

        launch = self._resolve()

        self.assertEqual(launch.settings.voices_dir, "D:/声音库")
        self.assertEqual(launch.settings.num_step, 16)

    def test_config_file_with_utf8_bom_is_still_parsed(self):
        self.config_path.write_text(
            '[omnivoice]\nnum_step = 16\n', encoding="utf-8-sig"
        )

        launch = self._resolve()

        self.assertEqual(launch.settings.num_step, 16)


class TestApiKeyPrecedence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.config_path = Path(self.tmp) / "config.toml"
        self.config_path.write_text(
            '[omnivoice]\napi_key = "from-config"\n', encoding="utf-8"
        )

    def _resolve(self, argv, env):
        return main_module.resolve_settings(
            argv, config_path=self.config_path, env=env
        )

    def test_env_var_overrides_config_file(self):
        launch = self._resolve([], {"OMNIVOICE_API_KEY": "from-env"})

        self.assertEqual(launch.settings.api_key, "from-env")

    def test_cli_flag_overrides_env_var(self):
        launch = self._resolve(
            ["--api-key", "from-cli"], {"OMNIVOICE_API_KEY": "from-env"}
        )

        self.assertEqual(launch.settings.api_key, "from-cli")

    def test_empty_cli_flag_disables_auth_explicitly(self):
        launch = self._resolve(["--api-key", ""], {"OMNIVOICE_API_KEY": "from-env"})

        self.assertEqual(launch.settings.api_key, "")

    def test_empty_env_var_is_treated_as_unset(self):
        launch = self._resolve([], {"OMNIVOICE_API_KEY": ""})

        self.assertEqual(launch.settings.api_key, "from-config")


class TestRepoRoot(unittest.TestCase):
    def test_default_config_path_points_at_the_repository_root(self):
        package_dir = Path(main_module.__file__).resolve().parent

        self.assertEqual(main_module.REPO_ROOT, package_dir.parent.parent)
        self.assertEqual(
            main_module.DEFAULT_CONFIG_PATH, main_module.REPO_ROOT / "config.toml"
        )
        self.assertTrue((main_module.REPO_ROOT / "config.example.toml").is_file())


if __name__ == "__main__":
    unittest.main()

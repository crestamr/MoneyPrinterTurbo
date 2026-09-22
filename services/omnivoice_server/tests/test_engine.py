import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from services.omnivoice_server import engine


class TestEngineLifecycle(unittest.TestCase):
    def setUp(self):
        self.clock = [1000.0]
        self.eng = engine.Engine(
            model_id="k2-fsa/OmniVoice",
            device="cuda:0",
            idle_unload_seconds=60,
            clock=lambda: self.clock[0],
        )

    def test_model_is_not_loaded_before_first_use(self):
        self.assertFalse(self.eng.is_loaded)

    def test_first_access_loads_the_model_once(self):
        fake_model = MagicMock()
        with patch.object(
            engine, "_load_model", return_value=fake_model
        ) as load_mock:
            self.assertIs(self.eng.model, fake_model)
            self.assertIs(self.eng.model, fake_model)
        load_mock.assert_called_once()
        self.assertTrue(self.eng.is_loaded)

    def test_unload_releases_the_model(self):
        with patch.object(engine, "_load_model", return_value=MagicMock()):
            self.eng.model
        self.eng.unload()
        self.assertFalse(self.eng.is_loaded)

    def test_unload_is_safe_when_never_loaded(self):
        self.eng.unload()
        self.assertFalse(self.eng.is_loaded)

    def test_idle_sweep_unloads_after_the_timeout(self):
        with patch.object(engine, "_load_model", return_value=MagicMock()):
            self.eng.model
        self.clock[0] += 61
        self.eng.sweep_idle()
        self.assertFalse(self.eng.is_loaded)

    def test_idle_sweep_keeps_the_model_before_the_timeout(self):
        with patch.object(engine, "_load_model", return_value=MagicMock()):
            self.eng.model
        self.clock[0] += 30
        self.eng.sweep_idle()
        self.assertTrue(self.eng.is_loaded)

    def test_idle_sweep_is_disabled_when_timeout_is_zero(self):
        eng = engine.Engine(
            model_id="k2-fsa/OmniVoice",
            device="cuda:0",
            idle_unload_seconds=0,
            clock=lambda: self.clock[0],
        )
        with patch.object(engine, "_load_model", return_value=MagicMock()):
            eng.model
        self.clock[0] += 9999
        eng.sweep_idle()
        self.assertTrue(eng.is_loaded)


if __name__ == "__main__":
    unittest.main()

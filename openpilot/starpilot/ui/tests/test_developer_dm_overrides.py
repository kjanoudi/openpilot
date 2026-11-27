"""Developer driver monitoring overrides change only while parked."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

from openpilot.common.params import Params


def test_dm_override_callbacks_save_only_while_parked(tmp_path):
  from openpilot.selfdrive.ui.layouts.settings import developer as c3
  params = Params(str(tmp_path))
  owner = c3.DeveloperLayout.__new__(c3.DeveloperLayout)
  owner._params = params
  owner._disable_forced_toggle = Mock()
  owner._disable_timeout_toggle = Mock()
  ui = SimpleNamespace(params=params, is_offroad=lambda: False)
  with patch.object(c3, 'ui_state', ui):
    for key, callback in (('DisableDMForcedDisengage', owner._on_disable_forced),
                          ('DisableDMTimeout', owner._on_disable_timeout)):
      callback(True)
      assert not params.get_bool(key)
      ui.is_offroad = lambda: True
      callback(True)
      assert params.get_bool(key)
      ui.is_offroad = lambda: False
      callback(False)
      assert params.get_bool(key)

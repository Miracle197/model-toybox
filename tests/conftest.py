import os

import pytest

os.environ["TRAINKIT_LANG"] = "en"


@pytest.fixture(autouse=True)
def tk_home(tmp_path, monkeypatch):
    home = tmp_path / "tkhome"
    monkeypatch.setenv("TRAINKIT_HOME", str(home))
    return home

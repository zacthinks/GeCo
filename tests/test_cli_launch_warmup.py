"""Public GeCo launch CLI passes the optional startup warm-up contract."""
from __future__ import annotations

from geometric_coder import cli


def test_geco_cli_launch_warmup_defaults_on_and_can_be_disabled(monkeypatch):
    launches = []

    class Server:
        def wait(self):
            launches.append("wait")

        def close(self):
            launches.append("close")

    class Project:
        def launch(self, **kwargs):
            launches.append(kwargs)
            return Server()

    monkeypatch.setattr(
        cli.GeometricCoder, "open", lambda _path: Project(),
    )
    assert cli.main(["launch", "/some/project"]) == 0
    assert launches[0]["warmup"] is True
    assert launches[1:] == ["wait", "close"]
    launches.clear()
    assert cli.main(["launch", "/some/project", "--no-warmup"]) == 0
    assert launches[0]["warmup"] is False
    assert launches[1:] == ["wait", "close"]

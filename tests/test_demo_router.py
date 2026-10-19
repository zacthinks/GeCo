"""Router regression tests (pytest.importorskip keeps core GeCo tests independent of Flask)."""
import pytest

flask = pytest.importorskip("flask")

from demo.space import COOKIE_NAME, create_frontend


class FakeManager:
    idle_seconds = 1800

    def __init__(self):
        self.token = None
        self.borrows = 0
        self.ends = 0

    def active(self):
        return self.token is not None

    def claim(self):
        if self.token is not None:
            return None
        self.token = "private-token"
        return self.token

    def borrow(self, token):
        if token == self.token and token is not None:
            self.borrows += 1
            return 7861
        return None

    def release(self, token):
        pass

    def end(self, token):
        if token != self.token or token is None:
            return False
        self.token = None
        self.ends += 1
        return True


def test_one_slot_and_end_session(monkeypatch):
    monkeypatch.setenv("GECO_DEMO_SECURE_COOKIE", "0")
    manager = FakeManager()
    app = create_frontend(manager)
    with app.test_client() as a, app.test_client() as b:
        assert a.get("/").status_code == 200
        started = a.post("/__demo/start")
        assert started.status_code == 302
        assert COOKIE_NAME in started.headers["Set-Cookie"]
        assert "HttpOnly" in started.headers["Set-Cookie"]
        assert b.post("/__demo/start").status_code == 503
        assert b.get("/_dash-layout").status_code == 403
        assert b.post("/__demo/end").status_code == 302
        assert manager.ends == 0
        assert a.post("/__demo/end").status_code == 302
        assert manager.ends == 1
        assert b.post("/__demo/start").status_code == 302


def test_foreign_origin_cannot_claim_or_end(monkeypatch):
    monkeypatch.setenv("GECO_DEMO_SECURE_COOKIE", "0")
    manager = FakeManager()
    client = create_frontend(manager).test_client()
    assert client.post("/__demo/start", headers={"Origin": "https://attacker.example"}).status_code == 403
    assert manager.token is None
    assert client.post("/__demo/start").status_code == 302
    assert client.post("/__demo/end", headers={"Origin": "https://attacker.example"}).status_code == 403
    assert manager.token is not None

"""Run the real GeCo Dash UI for one isolated, disposable demo project."""

from __future__ import annotations

import argparse
from pathlib import Path

from geometric_coder import GeometricCoder
from geometric_coder.ui import create_app

_END_SESSION = """
<div style="position:fixed;top:9px;right:16px;z-index:1100;
            display:flex;align-items:center;gap:10px;padding:7px 10px;
            background:#fff;border:1px solid #d7e2da;border-radius:8px;
            box-shadow:0 2px 8px rgba(0,0,0,.12);font:13px system-ui">
  <span style="color:#496050">Online demo · changes are temporary</span>
  <form action="/__demo/end" method="post" style="margin:0">
    <button type="submit" style="background:#2f744a;color:#fff;border:0;
            border-radius:5px;padding:7px 10px;cursor:pointer">End Session</button>
  </form>
</div>
"""


def make_app(project_dir: Path):
    project = GeometricCoder.open(project_dir)
    app = create_app(project)
    # Keep the upstream Dash component tree, callbacks, and CSS unchanged.
    app.index_string = app.index_string.replace("</body>", _END_SESSION + "\n</body>")
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    app = make_app(args.project)
    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()

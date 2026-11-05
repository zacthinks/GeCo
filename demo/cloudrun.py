"""Production front door for Cloud Run's single-process disposable GeCo demo.

Run with one Gunicorn worker. The manager's in-process lease cannot coordinate
across workers or Cloud Run instances; deployment limits instances to one.
"""
from __future__ import annotations

import atexit
import os
from pathlib import Path

from demo.space import SlotManager, create_frontend

_TEMPLATE = Path(os.environ.get("GECO_DEMO_TEMPLATE", "/app/demo_data/starter.geco"))
manager = SlotManager(
    _TEMPLATE,
    idle_seconds=int(os.environ.get("GECO_DEMO_IDLE_SECONDS", "600")),
)
atexit.register(manager.close)
app = create_frontend(manager)

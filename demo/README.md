# GeCo online demo — Cloud Run technical preview

The `geco-demo-online` branch starts from GeCo 0.10.0 and reuses the existing one-visitor Flask/Gunicorn reverse-proxy prototype. It runs **the real GeCo Dash application**, not a replica.

**Current content is deliberately temporary:** `demo/build_starter.py` creates a 36-sentence synthetic starter (lexical TF-IDF, two example codes and simple classifiers) at image build time. It is only for verifying Cloud Run, startup, routing, callback behavior, session isolation and cost. It must not be advertised as the final research demonstration. User-authored analyses will replace it after the hosting layer works.

## How it works

- A visitor requests a disposable session using a secure, HttpOnly cookie.
- A single Flask front door clones the starter and launches GeCo in a localhost subprocess.
- The worker exposes the full Dash UI; no browser request can access a worker without a valid session cookie.
- End Session deletes the cloned project; ten minutes of inactivity also expires it.
- The starter is unchanged between visits. Container stop/restart loses all temporary work.
- A single local session slot is enforced per container; Cloud Run may briefly start more than one container, so this is **not** a global distributed lock.

## Run locally

Install GeCo from this checkout with Flask, Gunicorn and pytest. For example:

```sh
python -m pip install -e . flask gunicorn pytest
GECO_DEMO_TEMPLATE="$(pwd)/demo_data/starter.geco" GECO_DEMO_SEMANTIC=0 python -m demo.build_starter
GECO_DEMO_TEMPLATE="$(pwd)/demo_data/starter.geco" GECO_DEMO_SECURE_COOKIE=0 python -m demo.space
```

Then open http://localhost:7860. Run `python -m pytest tests/test_demo_router.py -q` for basic router regressions. Production-like validation must also exercise real Dash callbacks, cookie/session isolation, stale tabs and classifier refitting.

## Cloud Run

See [CLOUD_RUN.md](CLOUD_RUN.md) for the source deployment command, settings, limitations and launch checklist.

## Replacing the temporary content

The *hosting* layer is the first milestone. After it is reachable and smoke-tested, replace the synthetic builder with an explicitly approved, precomputed project. For a TeAL-linked project, this requires packaging the entire portable TeAL root (`.teal` artifacts and GeCo workspace) and opening the linked GeCo handle; a lone GeCo SQLite file is insufficient. Keep the baseline outside public Git history until provenance, licensing, privacy and any distribution constraints are checked.

**Not yet claimed:** a successful Docker build on Cloud Run, a live service URL, a final approved dataset, or completed production security/performance testing.

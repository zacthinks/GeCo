# GeCo hosted demo (proof of concept)

This is the real Python/Dash GeCo application, not a JavaScript replica. The demo layer remains on the permanent `geco-demo` branch.

## Architecture

- One visitor at a time; random HttpOnly, SameSite cookie grants access.
- Flask routes public traffic on port 7860; Dash is bound to localhost port 7861.
- Every new visitor gets an independent copy of a pristine GeCo project.
- End Session terminates the isolated process and destroys its project directory.
- A 30-minute idle timeout also reclaims the slot.
- The pristine project contains real TF-IDF and sentence-transformer geometries, two preloaded codes, example human annotations, and fitted classifier predictions.
- This proof of concept uses Flask's development server; run one manager process only. A production WSGI setup and resource safeguards are required before public launch.

## Local smoke test

In a Python 3.13 environment with GeCo installed:

```sh
uv sync --all-extras --dev
uv pip install flask
GECO_DEMO_TEMPLATE="$(pwd)/demo_data/starter.geco" GECO_DEMO_SEMANTIC=0 uv run python -m demo.build_starter
GECO_DEMO_TEMPLATE="$(pwd)/demo_data/starter.geco" GECO_DEMO_SECURE_COOKIE=0 uv run python -m demo.space
```

Visit http://localhost:7860; verify start, Explore, Develop, End Session, and second visitor isolation. Remove GECO_DEMO_SEMANTIC=0 to create the intended semantic geometry (downloads a sentence-transformer model).

## Hugging Face deployment

Create a **Docker Space** (not GitHub Spaces) at https://huggingface.co/new-space. The Space repository README must have YAML front matter specifying `sdk: docker` and `app_port: 7860`. Copy this branch's deployment source into the Space repository, renaming `Dockerfile.demo` to the root `Dockerfile`. Include `src/`, `demo/`, `pyproject.toml`, `uv.lock`, and `.python-version`.

This branch does not create or deploy a Space automatically. Test the build, requests and callback routing, classifier fitting, cookies, load, shutdown, and process cleanup on the actual Space before making it public.

## Limits and follow-up

This is a feasibility prototype, not a production-validated public service. The basic CPU instance has two cores, so avoid simultaneous heavy jobs. Add production request-size, timeout, CPU and memory limits, and a maximum total lease age. Validate stale-tab behavior and unauthenticated route access. Visitors must not upload confidential data. A shared-project periodic reset is intentionally not used.

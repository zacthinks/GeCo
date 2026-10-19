# Cloud Run (low-cost alternative to Hugging Face Spaces)

This is a deployment option for the **real GeCo Dash application**, using the existing disposable, single-visitor router. It is a *proof of concept*, not yet verified end-to-end on Google Cloud.

## Why Cloud Run?

With request-based billing, minimum instances 0, and very light traffic, the Cloud Run compute bill may remain in the free tier. You still need a Google Cloud project with **billing enabled**, and Cloud Build, Artifact Registry image storage, network egress, and compute beyond the free tier can incur charges. Set budget alerts (they are notifications, **not hard spending caps**).

## Deployment

Use the `geco-demo` branch source. Cloud Run's source deploy expects a root Dockerfile, so build from a checkout where `Dockerfile.cloudrun` is copied to `Dockerfile` (do not overwrite the Hugging Face-oriented `Dockerfile.demo` on master).

```sh
cp Dockerfile.cloudrun Dockerfile
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud run deploy geco-demo \
  --source . \
  --region us-east1 \
  --allow-unauthenticated \
  --cpu 2 \
  --memory 4Gi \
  --min-instances 0 \
  --max-instances 1 \
  --concurrency 8 \
  --timeout 300 \
  --cpu-throttling \
  --set-env-vars GECO_DEMO_IDLE_SECONDS=600
```

Inspect the resulting monthly bill and adjust memory/CPU downward after real measurements. A larger memory limit is possible but costs more. The build installs the necessary Python dependencies, generates the starter project with true lexical and semantic geometries, then launches one WSGI front door and (when a visitor starts) one private Dash process.

## Session limitations

**Cloud Run's max-instances setting is not a strict global lock**, and scaling and replacement can briefly result in more than one instance. The in-memory one-visitor lease is only guaranteed within an individual container. It is therefore an *approximate one-at-a-time demonstration*, not a strict global quota or a security boundary across instances.

Instances can be stopped at any time. All project changes and session tokens are disposable; a container shutdown can interrupt an active visitor. In particular, the request-based CPU allocation makes idle-background cleanup unreliable. An expired lease is reclaimed on the next request; terminated containers lose everything. The End Session control explicitly deletes the temporary project.

A future globally coordinated occupancy limit would need external coordination (and likely additional costs). Avoid promising uninterrupted 30-minute sessions on this free-tier-oriented configuration.

## Before a public launch

Run real browser tests against deployed Cloud Run for start, full Dash callback/assets routing, coding, classifier fitting, End Session, cross-visitor isolation, concurrency, and stale-session handling. Validate peak memory, build size, and CPU demand. Harden request limits, process shutdown, and abuse prevention. Restrict expensive training or use a smaller corpus if necessary.

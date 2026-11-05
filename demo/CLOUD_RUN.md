# Deploy GeCo's technical preview on Google Cloud Run

This deploys the real GeCo 0.10.0 Dash UI with a **temporary 36-sentence synthetic starter**. The goal is to validate infrastructure and obtain an online URL *before* uploading real analyses. This recipe has not yet been executed in the target Google Cloud project.

## Google Cloud project setup

A Google Cloud project with billing enabled and the `gcloud` CLI authenticated to that project is required. Choose your own project ID, configure budget alerts, and enable the Cloud Run, Cloud Build and Artifact Registry APIs. Budget alerts are **not** spending caps. Image storage, builds, egress and excess compute can cost money even with `min-instances=0`.

From a checkout of the **`geco-demo-online`** branch:

```sh
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud run deploy geco-demo \
  --source . \
  --region us-east1 \
  --allow-unauthenticated \
  --cpu 2 \
  --memory 2Gi \
  --min-instances 0 \
  --max-instances 1 \
  --concurrency 8 \
  --timeout 300 \
  --cpu-throttling \
  --set-env-vars GECO_DEMO_IDLE_SECONDS=600
```

The root `Dockerfile` installs **the checked-out GeCo source**, Flask and Gunicorn. At image build time it generates the small placeholder. No Wikipedia fetching or large embedding download runs on visitor startup. The Gunicorn front door launches exactly one local Dash subprocess when a session is claimed.

`--allow-unauthenticated` exposes this service publicly as soon as deployment completes, even if you have not linked it anywhere. For a genuinely access-restricted staging deployment, use a Google Cloud authentication arrangement instead of the public flag. Do not place sensitive data in the placeholder or embed real analyses before their review.

## Verify before website linking

1. Visit the generated HTTPS URL, start a session and confirm that Explore, Develop and the Dash callback/assets endpoints work.
2. Add a code/annotation and train a classifier; End Session; start a second session and confirm the first visitor's edits are absent.
3. Use a second browser without the lease cookie while the first is active and verify the occupied message and denied Dash access.
4. Check stale tabs, idle expiry and restarts; inspect Cloud Run logs for failed worker startups and cleanup errors.
5. Measure cold-start time, memory, image size and actual billing. Adjust CPU/memory and avoid leaving a cost-incurring service unintentionally running.

Cloud Run's one-instance limit is not a strict global mutex, the instance filesystem is ephemeral, and CPU throttling can delay background cleanup until the next request. A visitor may lose work on restart. This is a disposable, best-effort demo and **not** persistent multiuser hosting.

## Next step after the URL works

Swap the temporary starter for user-approved existing analyses and add the associated TeAL-linked project opener if necessary; test relocating/cloning the complete project pair. Publishing the final demo and embedding it on a personal website are separate approval steps.

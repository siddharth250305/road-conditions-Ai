# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Two audiences, weighted equally (confirmed 2026-11-02):

- **Portfolio reviewers**: recruiters, hiring engineers and professors judging a student ML project. They arrive cold, spend a few minutes, and want to see quickly that the work is real, careful and honest about what it has not proven.
- **Road inspectors**: people who would actually check a road photo for surface condition. They need to drop a photo, get a calibrated reading or an explicit "not sure", and see where the model looked.

## Product Purpose

Road Condition AI classifies a road-surface photo into six friction classes (dry, wet, water, fresh snow, melted snow, ice) with calibrated confidence and a class-activation heatmap, and presents a separate cross-country road-damage detection study (RDD2022, YOLOv8n trained on Japan, scored on Norway, Czech Republic and USA, then adapted with a few target labels). Success: a reviewer leaves convinced the project is rigorous; an inspector gets an answer they can weigh correctly.

## Positioning

Honesty is the mechanism. The demo never forces a label on a low-confidence answer, labels mock models on every screen, shows calibrated (temperature-scaled) confidence rather than raw softmax, and states its training-domain limits next to every result. Study charts show measured numbers only, or say "pending".

## Operating Context

- Single-page React app served by a FastAPI + ONNX Runtime (CPU) service from one Docker image (port 7860; `make demo-mock` on port 8000).
- Endpoints: `/predict` (multipart photo), `/health`, `/study`, `/samples/*`.
- Screens: an opening home view, Report (analyse a photo), Study (cross-country detection results), and a limits/provenance panel available everywhere.

## Capabilities and Constraints

- Six classes only, no "not a road" class. Low-confidence answers become an explicit "Not sure" state.
- Error wording is fixed client-side (`frontend/src/api.ts`); server details are never shown.
- The model is trained only on RSCD (China). Other countries, night, rain on the lens and other surfaces are untested.
- Real training runs have not happened yet: the deployed model is a mock and must say so (`MOCK_BANNER`).
- Stack: React 19, Vite, TypeScript, Tailwind v4, d3-scale/shape. React Three Fiber (three.js) is approved for the 3D home view (confirmed 2026-11-02).

## Brand Commitments

- Name: Road Condition AI.
- Voice: plain, precise British-leaning English; no hype; states limits without apology.
- User-binding visual brief (2026-11-02): designed from the reference photo `IMG_7651.JPG` (top-down drone shot of a city crossroads); the home view opens with 3D components; navigation uses glass buttons.

## Evidence on Hand

- Reference photo: `IMG_7651.JPG` (repo root).
- Sample inputs: `frontend/public/samples/sample-{1,2,3}.jpg`, generated placeholders, not real photos.
- No real accuracy numbers exist yet; README results tables read "pending". Never fabricate metrics, users, testimonials or deployments.

## Product Principles

1. Never overstate: a missing or uncertain answer is shown as missing or uncertain.
2. Limits travel with results: wherever a reading appears, its training-domain caveat is one glance away.
3. Show the evidence: heatmaps, calibration and provenance are surfaced, not hidden.
4. Work for both audiences: a reviewer must grasp the project in seconds; an inspector must reach the photo drop in one action.

## Accessibility & Inclusion

WCAG 2.2 AA: colour is never the only class signal (each class has a pattern), live-region announcements for results, full keyboard use, reduced-motion support. The 3D view must have a non-WebGL, reduced-motion fallback.

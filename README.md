---
title: Crop Disease Detector API
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: FastAPI crop disease decision environment
---

# Crop Disease Detector API

This Docker Space serves a FastAPI backend for the crop disease environment.

## Problem

Crop diseases can spread quickly across nearby fields. This environment simulates field observations and lets an AI agent choose an intervention action for each step:

- `treat`
- `remove`
- `monitor`
- `do_nothing`

The environment returns rewards based on severity-aware decision quality.

## Why This Is Useful

- Standardized benchmark for crop disease decision agents.
- Multi-difficulty scenarios (`easy`, `medium`, `hard`) from curated JSON datasets.
- Clear, reproducible scoring through per-task graders.
- Fast API-based evaluation flow for hackathon demos and judging.

## Endpoints

- GET /health
- POST /reset
- POST /step
- GET /state
- GET /score

## Quick Test

1. GET /health should return {"status":"ok"}
2. POST /reset with {"task":"easy"}
3. POST /step with an action payload

## Demo Flow (Judge Friendly)

Open API docs:

- `/docs`

Then run this sequence:

```bash
curl -X POST "https://vignezkumaran-crop-disease-detector.hf.space/reset" \
	-H "Content-Type: application/json" \
	-d '{"task":"easy"}'

curl -X POST "https://vignezkumaran-crop-disease-detector.hf.space/step" \
	-H "Content-Type: application/json" \
	-d '{"action":"monitor","confidence":0.85,"reasoning":"Mild/moderate disease should be monitored first"}'

curl "https://vignezkumaran-crop-disease-detector.hf.space/state"
curl "https://vignezkumaran-crop-disease-detector.hf.space/score"
```

## Expected Output Behavior

- `/reset` returns the first observation for the selected task.
- `/step` returns:
	- `reward` for the chosen action
	- `total_reward` cumulative score
	- `done` when episode ends
	- `info.history` with step-by-step decisions on completion
- `/score` returns current cumulative reward and progress.

## Scoring Summary

- Correct severe/critical actions are rewarded more.
- Dangerous inaction on severe/critical disease is penalized.
- Graders return normalized scores from `0.0` to `1.0`.

## Reliability Checks

- Health endpoint: `GET /health` returns `{"status":"ok"}`.
- Invalid actions return HTTP `422`.
- Calling `/step` before `/reset` returns HTTP `400`.
- Calling `/step` after completion returns HTTP `409`.

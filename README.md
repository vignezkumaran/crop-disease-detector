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

from __future__ import annotations

import json
from threading import Lock
from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ValidationError

from environment import CropAction, CropDiseaseEnvironment, FieldObservation


class ResetRequest(BaseModel):
    task: Literal["easy", "medium", "hard"] = "easy"
    seed: Optional[int] = Field(default=None, ge=0)


class StepRequest(BaseModel):
    action: Literal["treat", "remove", "monitor", "do_nothing"]
    target_disease: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasoning: Optional[str] = None


class ResetResponse(BaseModel):
    task: str
    observation: FieldObservation
    done: bool
    step_count: int
    total_reward: float


class StepResponse(BaseModel):
    observation: Optional[FieldObservation]
    reward: float
    total_reward: float
    done: bool
    step_count: int
    info: Dict[str, Any]


class StateResponse(BaseModel):
    initialized: bool
    task: Optional[str]
    observation: Optional[FieldObservation]
    done: bool
    step_count: int


class ScoreResponse(BaseModel):
    initialized: bool
    task: Optional[str]
    total_reward: float
    step_count: int
    done: bool


app = FastAPI(title="Crop Disease Detector API", version="1.0.0")

_env_lock = Lock()
_env: Optional[CropDiseaseEnvironment] = None
_task: Optional[str] = None
_total_reward: float = 0.0


def _require_env() -> CropDiseaseEnvironment:
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    return _env


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
        return HTMLResponse(
                """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Crop Disease Detector</title>
    <style>
        :root {
            --bg: #f4f5f6;
            --panel: #ffffff;
            --ink: #101214;
            --muted: #5b6066;
            --line: #d8dde3;
            --accent: #0f766e;
            --accent-ink: #ffffff;
            --danger: #b42318;
            --ok: #067647;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            background: var(--bg);
            color: var(--ink);
            font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        }
        .wrap {
            max-width: 980px;
            margin: 0 auto;
            padding: 24px 16px 40px;
        }
        .hero {
            border: 1px solid var(--line);
            background: var(--panel);
            border-radius: 14px;
            padding: 20px;
            margin-bottom: 14px;
        }
        h1 {
            margin: 0;
            font-size: 1.4rem;
            letter-spacing: 0.01em;
        }
        .sub {
            color: var(--muted);
            margin-top: 6px;
            font-size: 0.95rem;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 12px;
        }
        @media (min-width: 900px) {
            .grid { grid-template-columns: 1fr 1fr; }
        }
        .card {
            border: 1px solid var(--line);
            background: var(--panel);
            border-radius: 14px;
            padding: 16px;
        }
        .row {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin: 10px 0;
        }
        label {
            display: block;
            font-size: 0.82rem;
            color: var(--muted);
            margin: 8px 0 4px;
        }
        input, select, textarea {
            width: 100%;
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 10px 11px;
            font: inherit;
            color: var(--ink);
            background: #fff;
        }
        textarea { min-height: 82px; resize: vertical; }
        button {
            border: 1px solid var(--ink);
            background: #fff;
            color: var(--ink);
            border-radius: 10px;
            padding: 9px 12px;
            cursor: pointer;
            font-weight: 600;
        }
        button.primary {
            background: var(--accent);
            color: var(--accent-ink);
            border-color: var(--accent);
        }
        .meta {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin: 6px 0 12px;
        }
        .chip {
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 5px 10px;
            font-size: 0.82rem;
            color: var(--muted);
            background: #fff;
        }
        .chip.ok { border-color: #8ad2a8; color: var(--ok); }
        .chip.bad { border-color: #efb9b4; color: var(--danger); }
        pre {
            margin: 0;
            border: 1px solid var(--line);
            border-radius: 10px;
            background: #fbfbfc;
            padding: 11px;
            white-space: pre-wrap;
            word-break: break-word;
            font-size: 0.82rem;
            line-height: 1.45;
            min-height: 160px;
        }
    </style>
</head>
<body>
    <div class="wrap">
        <section class="hero">
            <h1>Crop Disease Detector</h1>
            <p class="sub">Interactive dashboard for reset, step, state, and score. Built for quick evaluator demos.</p>
        </section>

        <section class="grid">
            <article class="card">
                <strong>Episode Controls</strong>
                <label for="task">Task</label>
                <select id="task">
                    <option value="easy">easy</option>
                    <option value="medium">medium</option>
                    <option value="hard">hard</option>
                </select>
                <label for="seed">Seed (optional, reproducible episode order)</label>
                <input id="seed" type="number" min="0" placeholder="42" />
                <div class="row">
                    <button class="primary" onclick="resetEnv()">Reset</button>
                    <button onclick="getState()">State</button>
                    <button onclick="getScore()">Score</button>
                </div>

                <hr style="border:none;border-top:1px solid var(--line);margin:12px 0;" />

                <strong>Action</strong>
                <label for="action">Action</label>
                <select id="action">
                    <option value="monitor">monitor</option>
                    <option value="treat">treat</option>
                    <option value="remove">remove</option>
                    <option value="do_nothing">do_nothing</option>
                </select>

                <label for="target">Target disease (optional)</label>
                <input id="target" placeholder="Late Blight" />

                <label for="confidence">Confidence (0-1)</label>
                <input id="confidence" type="number" min="0" max="1" step="0.05" value="0.8" />

                <label for="reasoning">Reasoning (optional)</label>
                <textarea id="reasoning" placeholder="Explain the chosen action"></textarea>

                <div class="row">
                    <button class="primary" onclick="stepEnv()">Step</button>
                </div>
            </article>

            <article class="card">
                <strong>Live Output</strong>
                <div class="meta">
                    <span id="initChip" class="chip">initialized: false</span>
                    <span id="doneChip" class="chip">done: true</span>
                    <span id="taskChip" class="chip">task: -</span>
                    <span id="rewardChip" class="chip">total_reward: 0.0</span>
                </div>
                <pre id="out">Press Reset to start an episode.</pre>
            </article>
        </section>
    </div>

    <script>
        function setOut(obj) {
            document.getElementById("out").textContent = JSON.stringify(obj, null, 2);
        }

        function setMeta(payload) {
            const initialized = !!payload.initialized || !!payload.observation;
            const done = payload.done === undefined ? true : !!payload.done;
            const task = payload.task || "-";
            const total = payload.total_reward === undefined ? 0.0 : payload.total_reward;

            const initChip = document.getElementById("initChip");
            const doneChip = document.getElementById("doneChip");
            const taskChip = document.getElementById("taskChip");
            const rewardChip = document.getElementById("rewardChip");

            initChip.textContent = `initialized: ${initialized}`;
            doneChip.textContent = `done: ${done}`;
            taskChip.textContent = `task: ${task}`;
            rewardChip.textContent = `total_reward: ${Number(total).toFixed(3)}`;

            initChip.className = `chip ${initialized ? "ok" : "bad"}`;
            doneChip.className = `chip ${done ? "bad" : "ok"}`;
        }

        async function callApi(path, options = {}) {
            const res = await fetch(path, options);
            const data = await res.json();
            if (!res.ok) {
                throw { status: res.status, data };
            }
            return data;
        }

        async function resetEnv() {
            try {
                const task = document.getElementById("task").value;
                const seedRaw = document.getElementById("seed").value;
                const body = { task };
                if (seedRaw !== "") {
                    body.seed = Number(seedRaw);
                }
                const payload = await callApi("/reset", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(body),
                });
                setOut(payload);
                setMeta(payload);
            } catch (e) {
                setOut(e);
            }
        }

        async function stepEnv() {
            try {
                const action = document.getElementById("action").value;
                const target = document.getElementById("target").value || null;
                const confidence = Number(document.getElementById("confidence").value || "0.8");
                const reasoning = document.getElementById("reasoning").value || null;
                const payload = await callApi("/step", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ action, target_disease: target, confidence, reasoning }),
                });
                setOut(payload);
                setMeta(payload);
            } catch (e) {
                setOut(e);
            }
        }

        async function getState() {
            try {
                const payload = await callApi("/state");
                setOut(payload);
                setMeta(payload);
            } catch (e) {
                setOut(e);
            }
        }

        async function getScore() {
            try {
                const payload = await callApi("/score");
                setOut(payload);
                setMeta(payload);
            } catch (e) {
                setOut(e);
            }
        }

        getState();
    </script>
</body>
</html>
                """
        )


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/reset", response_model=ResetResponse)
async def reset(request: Request) -> ResetResponse:
    global _env, _task, _total_reward

    payload = ResetRequest()
    raw_body = await request.body()
    if raw_body:
        try:
            body_data = json.loads(raw_body.decode("utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid JSON body: {exc}") from exc

        try:
            payload = ResetRequest(**body_data)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc

    with _env_lock:
        task = payload.task
        _env = CropDiseaseEnvironment(task=task, seed=payload.seed)
        _task = task
        _total_reward = 0.0
        observation = _env.reset(seed=payload.seed)
        return ResetResponse(
            task=task,
            observation=observation,
            done=_env.is_done,
            step_count=_env.step_count,
            total_reward=_total_reward,
        )


@app.post("/step", response_model=StepResponse)
def step(payload: StepRequest) -> StepResponse:
    global _total_reward
    with _env_lock:
        env = _require_env()
        if env.is_done:
            raise HTTPException(status_code=409, detail="Episode is finished. Call /reset to start a new one.")

        try:
            action = CropAction(
                action=payload.action,
                target_disease=payload.target_disease,
                confidence=payload.confidence,
                reasoning=payload.reasoning,
            )
            next_obs, reward, done, info = env.step(action)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to process step: {exc}") from exc

        _total_reward += reward
        return StepResponse(
            observation=next_obs,
            reward=reward,
            total_reward=_total_reward,
            done=done,
            step_count=env.step_count,
            info=info,
        )


@app.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    with _env_lock:
        if _env is None:
            return StateResponse(
                initialized=False,
                task=None,
                observation=None,
                done=True,
                step_count=0,
            )

        return StateResponse(
            initialized=True,
            task=_task,
            observation=_env.state(),
            done=_env.is_done,
            step_count=_env.step_count,
        )


@app.get("/score", response_model=ScoreResponse)
def score() -> ScoreResponse:
    with _env_lock:
        if _env is None:
            return ScoreResponse(
                initialized=False,
                task=None,
                total_reward=0.0,
                step_count=0,
                done=True,
            )

        return ScoreResponse(
            initialized=True,
            task=_task,
            total_reward=_total_reward,
            step_count=_env.step_count,
            done=_env.is_done,
        )
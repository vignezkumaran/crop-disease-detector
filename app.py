from __future__ import annotations

import json
from threading import Lock
from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError

from environment import CropAction, CropDiseaseEnvironment, FieldObservation


class ResetRequest(BaseModel):
    task: Literal["easy", "medium", "hard"] = "easy"


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
        _env = CropDiseaseEnvironment(task=task)
        _task = task
        _total_reward = 0.0
        observation = _env.reset()
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
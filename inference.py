import json
import os
from typing import Any, Dict, Optional

from openai import OpenAI
from openenv import GenericEnvClient


# Defaults are intentionally set only for API_BASE_URL and MODEL_NAME.
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - use this to run against a local Docker image.
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Optional env metadata
ENV_REPO_ID = os.getenv("ENV_REPO_ID", "vignezkumaran/crop-disease-detector")
TASK = os.getenv("TASK", "easy")
MAX_STEPS = int(os.getenv("MAX_STEPS", "10"))


def _log(tag: str, payload: Dict[str, Any]) -> None:
    print(f"{tag} {json.dumps(payload, ensure_ascii=True)}", flush=True)


def _normalize_observation(step_result: Any) -> Any:
    if hasattr(step_result, "observation"):
        return step_result.observation
    if hasattr(step_result, "obs"):
        return step_result.obs
    if isinstance(step_result, dict):
        if "observation" in step_result:
            return step_result["observation"]
        if "obs" in step_result:
            return step_result["obs"]
    return step_result


def _is_done(step_result: Any) -> bool:
    if hasattr(step_result, "done"):
        return bool(step_result.done)
    if isinstance(step_result, dict) and "done" in step_result:
        return bool(step_result["done"])
    return False


def _reward(step_result: Any) -> float:
    if hasattr(step_result, "reward"):
        return float(step_result.reward or 0.0)
    if isinstance(step_result, dict) and "reward" in step_result:
        return float(step_result["reward"] or 0.0)
    return 0.0


def _info(step_result: Any) -> Dict[str, Any]:
    if hasattr(step_result, "info") and isinstance(step_result.info, dict):
        return step_result.info
    if isinstance(step_result, dict) and isinstance(step_result.get("info"), dict):
        return step_result["info"]
    return {}


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, (dict, list, str, int, float, bool)):
        return value
    return str(value)


def choose_action(client: OpenAI, observation: Any) -> Dict[str, Any]:
    obs_json = _to_jsonable(observation)

    prompt = (
        "You are a crop disease decision agent. "
        "Choose exactly one action from: treat, remove, monitor, do_nothing. "
        "Return strict JSON with keys: action, target_disease, confidence, reasoning. "
        "confidence must be in [0,1]."
    )

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(obs_json)},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    text = resp.choices[0].message.content or "{}"
    action = json.loads(text)

    # Keep action payload schema-safe.
    action_name = action.get("action", "monitor")
    if action_name not in {"treat", "remove", "monitor", "do_nothing"}:
        action_name = "monitor"

    confidence = action.get("confidence", 0.7)
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.7
    confidence = max(0.0, min(1.0, confidence))

    return {
        "action": action_name,
        "target_disease": action.get("target_disease"),
        "confidence": confidence,
        "reasoning": action.get("reasoning", "model-selected action"),
    }


def run_episode() -> Dict[str, Any]:
    if LOCAL_IMAGE_NAME:
        env = GenericEnvClient.from_docker_image(LOCAL_IMAGE_NAME)
        env_source = {"mode": "docker_image", "image": LOCAL_IMAGE_NAME}
    else:
        env = GenericEnvClient.from_env(ENV_REPO_ID)
        env_source = {"mode": "hub_env", "repo_id": ENV_REPO_ID}

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    env.connect()
    total_reward = 0.0
    step_count = 0
    try:
        _log(
            "START",
            {
                "task": TASK,
                "api_base_url": API_BASE_URL,
                "model_name": MODEL_NAME,
                "env": env_source,
            },
        )

        reset_result = env.reset(task=TASK)
        observation = _normalize_observation(reset_result)

        while step_count < MAX_STEPS:
            action = choose_action(client, observation)
            step_result = env.step(action)

            step_count += 1
            reward = _reward(step_result)
            done = _is_done(step_result)
            info = _info(step_result)
            total_reward += reward

            _log(
                "STEP",
                {
                    "step": step_count,
                    "action": action,
                    "reward": reward,
                    "total_reward": total_reward,
                    "done": done,
                },
            )

            if done:
                _log(
                    "END",
                    {
                        "steps": step_count,
                        "total_reward": total_reward,
                        "info": info,
                    },
                )
                return {
                    "steps": step_count,
                    "total_reward": total_reward,
                    "done": True,
                    "info": info,
                }

            observation = _normalize_observation(step_result)

        _log(
            "END",
            {
                "steps": step_count,
                "total_reward": total_reward,
                "done": False,
                "reason": "max_steps_reached",
            },
        )
        return {
            "steps": step_count,
            "total_reward": total_reward,
            "done": False,
            "reason": "max_steps_reached",
        }
    finally:
        env.disconnect()
        env.close()


if __name__ == "__main__":
    run_episode()

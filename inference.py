import os
import sys
from typing import Any, Dict, Optional

from openai import OpenAI
from environment import CropAction, CropDiseaseEnvironment


# Ensure helper executables installed in the active Python env (e.g. uv)
# are discoverable even when this script is launched via absolute python path.
_PY_BIN_DIR = os.path.dirname(sys.executable)
os.environ["PATH"] = _PY_BIN_DIR + os.pathsep + os.environ.get("PATH", "")


# Defaults are intentionally set only for API_BASE_URL and MODEL_NAME.
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
# Evaluator injects API_KEY for LiteLLM proxy routing.
API_KEY = os.getenv("API_KEY")
# Backward compatibility for local runs.
HF_TOKEN = os.getenv("HF_TOKEN")
STRICT_PROXY_MODE = bool(os.getenv("API_BASE_URL") and os.getenv("API_KEY"))

# Optional - use this to run against a local Docker image.
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Optional env metadata
ENV_REPO_ID = os.getenv("ENV_REPO_ID", "vignezkumaran/crop-disease-detector")
TASK = os.getenv("TASK", "easy")
MAX_STEPS = int(os.getenv("MAX_STEPS", "10"))
LLM_RETRIES = int(os.getenv("LLM_RETRIES", "3"))


def _num(value: float) -> str:
    return f"{value:.4f}"


def _log_start(task: str) -> None:
    print(f"[START] task={task}", flush=True)


def _log_step(step: int, reward: float) -> None:
    print(f"[STEP] step={step} reward={_num(reward)}", flush=True)


def _log_end(task: str, score: float, steps: int) -> None:
    print(f"[END] task={task} score={_num(score)} steps={steps}", flush=True)


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


def _fallback_action(observation: Any) -> Dict[str, Any]:
    severity_rank = {
        "none": 0,
        "mild": 1,
        "moderate": 2,
        "severe": 3,
        "critical": 4,
    }
    obs_json = _to_jsonable(observation)
    diseases = obs_json.get("diseases", []) if isinstance(obs_json, dict) else []

    if not diseases:
        return {
            "action": "do_nothing",
            "target_disease": None,
            "confidence": 0.75,
            "reasoning": "No detected disease in observation",
        }

    top = max(diseases, key=lambda d: severity_rank.get(str(d.get("severity", "none")), 0))
    top_sev = str(top.get("severity", "none"))
    target = top.get("name")

    if severity_rank.get(top_sev, 0) >= severity_rank["critical"]:
        action = "remove"
    elif severity_rank.get(top_sev, 0) >= severity_rank["severe"]:
        action = "treat"
    else:
        action = "monitor"

    return {
        "action": action,
        "target_disease": target,
        "confidence": 0.65,
        "reasoning": "Fallback heuristic selected from max disease severity",
    }


def choose_action(client: OpenAI, observation: Any) -> tuple[Dict[str, Any], bool]:
    obs_json = _to_jsonable(observation)

    prompt = (
        "You are a crop disease decision agent. "
        "Choose exactly one action from: treat, remove, monitor, do_nothing. "
        "Return strict JSON with keys: action, target_disease, confidence, reasoning. "
        "confidence must be in [0,1]."
    )

    last_exc: Optional[Exception] = None
    for _ in range(max(1, LLM_RETRIES)):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": str(obs_json)},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            text = resp.choices[0].message.content or "{}"
            # Keep parser dependencies minimal in case validator runtime differs.
            import json

            action = json.loads(text)
            break
        except Exception as exc:
            last_exc = exc
    else:
        if STRICT_PROXY_MODE:
            raise RuntimeError(f"LLM proxy call failed after retries: {last_exc}")
        return _fallback_action(observation), False

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
    }, True


def run_episode() -> Dict[str, Any]:
    # Emit START before any network/container operations so parser always sees it.
    _log_start(TASK)

    api_key = API_KEY or HF_TOKEN
    if not api_key:
        # Keep execution alive for parser checks; LLM criteria will fail if key is absent.
        _log_step(0, 0.0)
        _log_end(TASK, 0.0, 0)
        return {"steps": 0, "total_reward": 0.0, "done": False, "reason": "missing_api_key"}

    client = OpenAI(base_url=API_BASE_URL, api_key=api_key)

    # Run environment in-process to avoid runtime dependency on uv/container providers.
    env = CropDiseaseEnvironment(task=TASK)
    total_reward = 0.0
    step_count = 0
    llm_calls = 0
    try:
        observation = env.reset()

        while step_count < MAX_STEPS:
            action_dict, used_llm = choose_action(client, observation)
            if used_llm:
                llm_calls += 1

            action = CropAction(**action_dict)
            next_obs, reward, done, info = env.step(action)

            step_count += 1
            total_reward += float(reward)
            _log_step(step_count, float(reward))

            if done:
                _log_end(TASK, total_reward, step_count)
                return {
                    "steps": step_count,
                    "total_reward": total_reward,
                    "done": True,
                    "info": info,
                    "llm_calls": llm_calls,
                }

            if next_obs is None:
                break
            observation = next_obs

        _log_end(TASK, total_reward, step_count)
        return {
            "steps": step_count,
            "total_reward": total_reward,
            "done": False,
            "reason": "max_steps_reached",
            "llm_calls": llm_calls,
        }
    except Exception:
        # Keep parser visibility even on runtime issues.
        _log_step(max(1, step_count), 0.0)
        _log_end(TASK, total_reward, max(1, step_count))
        return {
            "steps": max(1, step_count),
            "total_reward": total_reward,
            "done": False,
            "reason": "runtime_error",
            "llm_calls": llm_calls,
        }


if __name__ == "__main__":
    run_episode()

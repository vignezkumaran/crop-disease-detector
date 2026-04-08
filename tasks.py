"""
Grader tasks for Crop Disease Detector.
Each function returns a score from 0.0 to 1.0.
"""
import json
from pathlib import Path
from typing import List, Dict, Any


SEVERITY_ORDER = {
    "none": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3,
    "critical": 4,
}


def _load_fields(task: str) -> List[Dict[str, Any]]:
    data_dir = Path(__file__).parent / "data"
    path = data_dir / f"{task}_fields.json"
    with open(path) as f:
        return json.load(f)


def _get_optimal_action(obs: Dict[str, Any]) -> str:
    diseases = obs.get("diseases", [])
    if not diseases:
        return "do_nothing"
    max_severity = max(SEVERITY_ORDER.get(d.get("severity", "none"), 0) for d in diseases)
    if max_severity >= SEVERITY_ORDER["critical"]:
        return "remove"
    if max_severity >= SEVERITY_ORDER["severe"]:
        return "treat"
    if max_severity >= SEVERITY_ORDER["mild"]:
        return "monitor"
    return "do_nothing"


def _calculate_reward(action: str, obs: Dict[str, Any]) -> float:
    optimal = _get_optimal_action(obs)
    rewards_map = {"remove": 0.5, "treat": 0.4, "monitor": 0.3, "do_nothing": 0.2}
    if action == optimal:
        return rewards_map.get(action, 0.0)
    diseases = obs.get("diseases", [])
    max_severity = max(SEVERITY_ORDER.get(d.get("severity", "none"), 0) for d in diseases)
    if max_severity == SEVERITY_ORDER["critical"] and action == "do_nothing":
        return -0.8
    if max_severity == SEVERITY_ORDER["severe"] and action == "do_nothing":
        return -0.5
    if action == "remove" and max_severity < SEVERITY_ORDER["critical"]:
        return -0.3
    return -0.2


def grade_easy(history: List[Dict[str, Any]]) -> float:
    if not history:
        return 0.0
    fields = _load_fields("easy")
    correct = 0
    for i, step in enumerate(history):
        if i >= len(fields):
            break
        obs = fields[i]
        action = step.get("action", {})
        action_type = action.get("action", "do_nothing")
        optimal = _get_optimal_action(obs)
        if action_type == optimal:
            correct += 1
    return correct / len(history)


def grade_medium(history: List[Dict[str, Any]]) -> float:
    if not history:
        return 0.0
    fields = _load_fields("medium")
    total_reward = 0.0
    max_possible = 0.0
    for i, step in enumerate(history):
        if i >= len(fields):
            break
        obs = fields[i]
        action = step.get("action", {})
        action_type = action.get("action", "do_nothing")
        reward = _calculate_reward(action_type, obs)
        total_reward += reward
        optimal = _get_optimal_action(obs)
        max_possible += _calculate_reward(optimal, obs)
    if max_possible == 0:
        return 0.0
    normalized = (total_reward + abs(min(0, total_reward))) / (max_possible + abs(min(0, max_possible)))
    return max(0.0, min(1.0, normalized))


def grade_hard(history: List[Dict[str, Any]]) -> float:
    if not history:
        return 0.0
    fields = _load_fields("hard")
    total_reward = 0.0
    max_possible = 0.0
    for i, step in enumerate(history):
        if i >= len(fields):
            break
        obs = fields[i]
        action = step.get("action", {})
        action_type = action.get("action", "do_nothing")
        reward = _calculate_reward(action_type, obs)
        total_reward += reward
        optimal = _get_optimal_action(obs)
        max_possible += _calculate_reward(optimal, obs)
        nearby = obs.get("nearby_fields", [])
        if nearby and action.get("reasoning"):
            reasoning_lower = action["reasoning"].lower()
            for nf in nearby:
                if nf.lower() in reasoning_lower:
                    total_reward += 0.1
                    break
    steps_taken = len(history)
    max_steps = len(fields)
    if steps_taken < max_steps:
        speed_bonus = 0.1 * (max_steps - steps_taken)
        total_reward += speed_bonus
        max_possible += speed_bonus
    if max_possible == 0:
        return 0.0
    normalized = (total_reward + abs(min(0, total_reward))) / (max_possible + abs(min(0, max_possible)))
    return max(0.0, min(1.0, normalized))


GRADERS = {"easy": grade_easy, "medium": grade_medium, "hard": grade_hard}


def get_grader(task: str):
    return GRADERS.get(task, grade_easy)

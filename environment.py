import json
import random
from pathlib import Path
from typing import List, Optional, Literal, Dict, Any, Tuple

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class DiseaseInfo(BaseModel):
    name: str
    severity: Literal["none", "mild", "moderate", "severe", "critical"]
    affected_area: float = Field(ge=0.0, le=100.0, description="Percentage of field affected")
    spread_risk: Literal["low", "medium", "high"]


class FieldObservation(BaseModel):
    field_id: str
    crop_type: str
    diseases: List[DiseaseInfo]
    temperature: float = Field(description="Temperature in Celsius")
    moisture: float = Field(ge=0.0, le=100.0, description="Soil moisture percentage")
    growth_stage: str
    nearby_fields: List[str] = Field(default_factory=list, description="IDs of adjacent fields")
    days_since_detection: int = Field(ge=0, description="Days since disease was first detected")
    task_difficulty: str = Field(default="easy")


class CropAction(BaseModel):
    action: Literal["treat", "remove", "monitor", "do_nothing"]
    target_disease: Optional[str] = Field(default=None, description="Primary disease being addressed")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Agent confidence score")
    reasoning: Optional[str] = Field(default=None, description="Brief justification for the action")


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------

SEVERITY_ORDER: Dict[str, int] = {
    "none": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3,
    "critical": 4,
}

OPTIMAL_REWARDS: Dict[str, float] = {
    "remove": 0.5,   # Correct removal of critical infection
    "treat": 0.4,    # Correct treatment of severe disease
    "monitor": 0.3,  # Proper monitoring of mild/moderate disease
    "do_nothing": 0.2,  # Correct inaction when healthy
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CropDiseaseEnvironment:
    """
    OpenEnv-compatible environment for crop disease detection and treatment.

    Observation: FieldObservation  (Pydantic model)
    Action     : CropAction        (Pydantic model)

    Episode flow
    ------------
    1. env.reset()           → first FieldObservation
    2. env.step(CropAction)  → (next_obs | None, reward, done, info)
    3. Repeat until done=True
    """

    ENV_ID = "crop_disease_detector"

    def __init__(self, task: str = "easy") -> None:
        self.task = task
        self._data_dir = Path(__file__).parent / "data"
        self._fields: List[Dict[str, Any]] = []
        self._current_index: int = 0
        self._current_obs: Optional[FieldObservation] = None
        self._step_count: int = 0
        self._done: bool = True
        self._episode_rewards: List[float] = []
        self._episode_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> FieldObservation:
        """Load task fields, shuffle them, and return the first observation."""
        self._fields = self._load_fields()
        random.shuffle(self._fields)
        self._current_index = 0
        self._step_count = 0
        self._done = False
        self._episode_rewards = []
        self._episode_history = []

        self._current_obs = self._make_obs(self._fields[self._current_index])
        return self._current_obs

    def state(self) -> Optional[FieldObservation]:
        """Return the current observation without advancing the episode."""
        return self._current_obs

    def step(
        self, action: CropAction
    ) -> Tuple[Optional[FieldObservation], float, bool, Dict[str, Any]]:
        """
        Process one action and advance to the next field.

        Returns
        -------
        next_obs : FieldObservation or None (when done)
        reward   : float
        done     : bool
        info     : dict with episode metadata when done
        """
        if self._done:
            raise RuntimeError("Episode is finished. Call reset() to start a new episode.")

        reward = self._calculate_reward(action, self._current_obs)

        # Record history
        self._episode_rewards.append(reward)
        self._episode_history.append(
            {
                "step": self._step_count,
                "observation": self._current_obs.model_dump(),
                "action": action.model_dump(),
                "reward": reward,
                "optimal_action": self._get_optimal_action(self._current_obs),
            }
        )
        self._step_count += 1
        self._current_index += 1

        # Check episode end
        if self._current_index >= len(self._fields):
            self._done = True
            self._current_obs = None
            info = {
                "episode_rewards": self._episode_rewards,
                "total_reward": sum(self._episode_rewards),
                "history": self._episode_history,
            }
            return None, reward, True, info

        self._current_obs = self._make_obs(self._fields[self._current_index])
        return self._current_obs, reward, False, {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_fields(self) -> List[Dict[str, Any]]:
        path = self._data_dir / f"{self.task}_fields.json"
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        with open(path, "r") as fh:
            return json.load(fh)

    def _make_obs(self, raw: Dict[str, Any]) -> FieldObservation:
        return FieldObservation(**raw, task_difficulty=self.task)

    def _get_optimal_action(self, obs: FieldObservation) -> str:
        """Determine the best action based on the highest disease severity."""
        if not obs.diseases:
            return "do_nothing"
        max_sev = max(SEVERITY_ORDER.get(d.severity, 0) for d in obs.diseases)
        if max_sev >= SEVERITY_ORDER["critical"]:
            return "remove"
        if max_sev >= SEVERITY_ORDER["severe"]:
            return "treat"
        if max_sev >= SEVERITY_ORDER["mild"]:
            return "monitor"
        return "do_nothing"

    def _calculate_reward(self, action: CropAction, obs: FieldObservation) -> float:
        """
        Reward function:
          +0.5  correct removal  (critical infection)
          +0.4  correct treatment (severe disease)
          +0.3  correct monitoring (mild/moderate)
          +0.2  correct inaction  (healthy field)
          -0.8  ignoring critical disease
          -0.5  ignoring severe disease
          -0.3  overreacting (remove on non-critical)
          -0.2  any other wrong action
        """
        optimal = self._get_optimal_action(obs)
        act = action.action

        max_sev = (
            max(SEVERITY_ORDER.get(d.severity, 0) for d in obs.diseases)
            if obs.diseases
            else 0
        )

        if act == optimal:
            return OPTIMAL_REWARDS.get(act, 0.0)

        # Wrong-action penalties
        if max_sev == SEVERITY_ORDER["critical"] and act == "do_nothing":
            return -0.8
        if max_sev == SEVERITY_ORDER["severe"] and act == "do_nothing":
            return -0.5
        if act == "remove" and max_sev < SEVERITY_ORDER["critical"]:
            return -0.3
        return -0.2

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def render(self) -> str:
        """Return a human-readable summary of the current observation."""
        obs = self._current_obs
        if obs is None:
            return "[CropDiseaseEnvironment] No active observation (episode done)."
        diseases_str = (
            ", ".join(f"{d.name} [{d.severity}]" for d in obs.diseases) or "None"
        )
        return (
            f"Field {obs.field_id} | Crop: {obs.crop_type} | "
            f"Stage: {obs.growth_stage} | "
            f"Temp: {obs.temperature}C | Moisture: {obs.moisture}% | "
            f"Diseases: {diseases_str}"
        )

    def get_episode_history(self) -> List[Dict[str, Any]]:
        return list(self._episode_history)

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def step_count(self) -> int:
        return self._step_count

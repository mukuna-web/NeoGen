"""
Scenario loader — loads YAML scenario files into Scenario objects.
"""

import os
import yaml
from ..models.scenario import Scenario, ValidationSource


def load_scenario(path: str) -> Scenario:
    """Load a scenario from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)

    # Build validation sources
    validation_sources = []
    for vs in data.get("validation_sources", []):
        validation_sources.append(ValidationSource(**vs))

    return Scenario(
        scenario_id=data.get("scenario_id", ""),
        title=data.get("title", ""),
        domain=data.get("domain", ""),
        question=data.get("question", ""),
        description=data.get("description", ""),
        constraints=data.get("constraints", []),
        target_outputs=data.get("target_outputs", []),
        success_criteria=data.get("success_criteria", []),
        suggested_specialties=data.get("suggested_specialties", []),
        num_specialties=data.get("num_specialties", 8),
        validation_sources=validation_sources,
        ground_truth=data.get("ground_truth", {}),
        background_knowledge=data.get("background_knowledge", ""),
        known_equations=data.get("known_equations", []),
        known_variables=data.get("known_variables", []),
        known_subsystems=data.get("known_subsystems", []),
        difficulty=data.get("difficulty", "hard"),
        tags=data.get("tags", []),
    )


def list_scenarios(scenarios_dir: str = None) -> list[dict]:
    """List available scenarios."""
    if scenarios_dir is None:
        scenarios_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scenarios")

    scenarios = []
    if os.path.isdir(scenarios_dir):
        for fname in sorted(os.listdir(scenarios_dir)):
            if fname.endswith((".yaml", ".yml")):
                path = os.path.join(scenarios_dir, fname)
                try:
                    with open(path) as f:
                        data = yaml.safe_load(f)
                    scenarios.append({
                        "file": fname,
                        "id": data.get("scenario_id", ""),
                        "title": data.get("title", ""),
                        "domain": data.get("domain", ""),
                        "difficulty": data.get("difficulty", ""),
                    })
                except Exception:
                    pass
    return scenarios

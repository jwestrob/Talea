"""Locate and checksum-verify Talea's frozen inference artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


ASSET_ROOT = Path(__file__).resolve().parent / "assets"
DEPLOYMENT_SHA256 = (
    "7aa5b7f320323b34d2db27b0df488732d1fcec39d72e2d0f5b774ebec1edea2e"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_deployment() -> dict:
    path = ASSET_ROOT / "deployment.json"
    observed = sha256_file(path)
    if observed != DEPLOYMENT_SHA256:
        raise ValueError(
            "deployment contract checksum mismatch: "
            f"observed {observed}, expected {DEPLOYMENT_SHA256}"
        )
    deployment = json.loads(path.read_text())
    if deployment.get("schema_version") != "talea_stable_discovery_v2":
        raise ValueError("unexpected Talea deployment schema")
    return deployment


@dataclass(frozen=True)
class RuntimeAssets:
    deployment_path: Path
    teacher_model: Path
    teacher_contract: Path
    global_model: Path
    protein_calibration: Path
    local_calibration: Path
    wedge_config: Path
    topology_config: Path
    sha256: dict[str, str]


def verified_runtime_assets(deployment: dict | None = None) -> RuntimeAssets:
    deployment = load_deployment() if deployment is None else deployment
    paths = {
        "teacher_model": ASSET_ROOT / "teacher_model.pt",
        "teacher_contract": ASSET_ROOT / "teacher_contract.json",
        "global_model": ASSET_ROOT / "global_model.joblib",
        "protein_calibration": ASSET_ROOT / "protein_calibration.json",
        "local_calibration": ASSET_ROOT / "local_calibration.json",
        "wedge_config": ASSET_ROOT / "wedge_config.json",
        "topology_config": ASSET_ROOT / "topology_config.json",
    }
    expected = {
        "teacher_model": deployment["teacher_model"]["sha256"],
        "teacher_contract": deployment["teacher_model"]["contract_sha256"],
        "global_model": deployment["global_model"]["sha256"],
        "protein_calibration": deployment["topology_dependencies"][
            "protein_calibration"
        ]["sha256"],
        "local_calibration": deployment["topology_dependencies"][
            "local_calibration"
        ]["sha256"],
        "wedge_config": deployment["topology_dependencies"]["wedge_config"][
            "sha256"
        ],
        "topology_config": deployment["topology_dependencies"][
            "operational_config"
        ]["sha256"],
    }
    observed: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing Talea runtime artifact: {path}")
        observed[name] = sha256_file(path)
        if observed[name] != expected[name]:
            raise ValueError(
                f"{name} checksum mismatch: observed {observed[name]}, "
                f"expected {expected[name]}"
            )
    return RuntimeAssets(
        deployment_path=ASSET_ROOT / "deployment.json",
        teacher_model=paths["teacher_model"],
        teacher_contract=paths["teacher_contract"],
        global_model=paths["global_model"],
        protein_calibration=paths["protein_calibration"],
        local_calibration=paths["local_calibration"],
        wedge_config=paths["wedge_config"],
        topology_config=paths["topology_config"],
        sha256=observed,
    )

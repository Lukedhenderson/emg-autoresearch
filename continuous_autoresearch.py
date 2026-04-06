from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
TRAIN_PATH = REPO_ROOT / "train.py"
RESULTS_PATH = REPO_ROOT / "results.jsonl"
BEST_PATH = REPO_ROOT / "best_results.json"
LOOP_LOG_PATH = REPO_ROOT / "loop_status.log"
RUN_OUTPUT_PATH = REPO_ROOT / "run.log"


@dataclass(frozen=True)
class Candidate:
    name: str
    window_ms: int
    overlap_ratio: float
    normalization_strategy: str
    rectify_signal: bool
    detrend_signal: bool
    remove_dc_offset: bool
    ar_order: int
    windows_per_group_class: int
    enable_cross_channel_summary: bool
    enable_channel_pair_features: bool
    channel_pair_limit: int
    model_family: str
    thresholds: dict[str, float]
    use_pca: bool
    pca_components: int
    feature_families: dict[str, bool]


def load_results() -> list[dict]:
    if not RESULTS_PATH.exists():
        return []
    rows = []
    for line in RESULTS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def grouped_score(row: dict) -> float | None:
    if row.get("status") != "success":
        return None
    split_strategy = row.get("split_strategy", "")
    if not str(split_strategy).startswith("group_kfold"):
        return None
    metrics = row.get("metrics") or {}
    score = metrics.get("f1_macro")
    return float(score) if score is not None else None


def best_grouped_result(rows: list[dict]) -> dict | None:
    grouped_rows = [row for row in rows if grouped_score(row) is not None]
    if not grouped_rows:
        return None
    return max(grouped_rows, key=lambda row: grouped_score(row) or float("-inf"))


def feature_families_signature(feature_families: dict[str, bool]) -> str:
    return ",".join(f"{name}:{int(enabled)}" for name, enabled in sorted(feature_families.items()))


def candidate_signature(candidate: Candidate) -> tuple:
    return (
        candidate.window_ms,
        round(candidate.overlap_ratio, 4),
        candidate.normalization_strategy,
        candidate.rectify_signal,
        candidate.detrend_signal,
        candidate.remove_dc_offset,
        candidate.ar_order,
        candidate.windows_per_group_class,
        candidate.enable_cross_channel_summary,
        candidate.enable_channel_pair_features,
        candidate.channel_pair_limit,
        candidate.model_family,
        round(candidate.thresholds["zc"], 4),
        round(candidate.thresholds["ssc"], 4),
        round(candidate.thresholds["wamp"], 4),
        candidate.use_pca,
        candidate.pca_components,
        feature_families_signature(candidate.feature_families),
    )


def result_signature(row: dict) -> tuple | None:
    preprocessing = row.get("preprocessing") or {}
    feature_families = row.get("feature_families") or {}
    thresholds = preprocessing.get("thresholds") or {}
    if not preprocessing or not feature_families:
        return None
    return (
        int(preprocessing.get("window_ms", -1)),
        round(float(preprocessing.get("overlap_ratio", -1.0)), 4),
        preprocessing.get("normalization_strategy"),
        bool(preprocessing.get("rectify_signal")),
        bool(preprocessing.get("detrend_signal")),
        bool(preprocessing.get("remove_dc_offset")),
        int(preprocessing.get("ar_order", -1)),
        96,
        True,
        False,
        4,
        row.get("model_family"),
        round(float(thresholds.get("zc", -1.0)), 4),
        round(float(thresholds.get("ssc", -1.0)), 4),
        round(float(thresholds.get("wamp", -1.0)), 4),
        bool(preprocessing.get("use_pca")),
        int(preprocessing.get("pca_components", 32)),
        feature_families_signature(feature_families),
    )


def current_train_text() -> str:
    return TRAIN_PATH.read_text(encoding="utf-8")


def replace_assignment(text: str, name: str, value_literal: str) -> str:
    pattern = rf"^{name} = .*$"
    replacement = f"{name} = {value_literal}"
    updated, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Failed to update assignment for {name}")
    return updated


def replace_block(text: str, name: str, value: dict[str, bool] | dict[str, float]) -> str:
    pattern = rf"{name} = \{{.*?\n\}}"
    replacement = f"{name} = {json.dumps(value, sort_keys=True, indent=4).replace('true', 'True').replace('false', 'False')}"
    updated, count = re.subn(pattern, replacement, text, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"Failed to update block for {name}")
    return updated


def apply_candidate(candidate: Candidate) -> None:
    text = current_train_text()
    text = replace_assignment(text, "WINDOW_MS", str(candidate.window_ms))
    text = replace_assignment(text, "WINDOW_OVERLAP_RATIO", repr(candidate.overlap_ratio))
    text = replace_assignment(text, "NORMALIZATION_STRATEGY", repr(candidate.normalization_strategy))
    text = replace_assignment(text, "RECTIFY_SIGNAL", "True" if candidate.rectify_signal else "False")
    text = replace_assignment(text, "DETREND_SIGNAL", "True" if candidate.detrend_signal else "False")
    text = replace_assignment(text, "REMOVE_DC_OFFSET", "True" if candidate.remove_dc_offset else "False")
    text = replace_assignment(text, "AR_ORDER", str(candidate.ar_order))
    text = replace_assignment(text, "WINDOWS_PER_GROUP_CLASS", str(candidate.windows_per_group_class))
    text = replace_assignment(text, "ENABLE_CROSS_CHANNEL_SUMMARY", "True" if candidate.enable_cross_channel_summary else "False")
    text = replace_assignment(text, "ENABLE_CHANNEL_PAIR_FEATURES", "True" if candidate.enable_channel_pair_features else "False")
    text = replace_assignment(text, "CHANNEL_PAIR_LIMIT", str(candidate.channel_pair_limit))
    text = replace_assignment(text, "MODEL_FAMILY", repr(candidate.model_family) + '  # logreg, linear_svm, rbf_svm, random_forest, extra_trees, hist_gb, small_mlp')
    text = replace_assignment(text, "USE_PCA", "True" if candidate.use_pca else "False")
    text = replace_assignment(text, "PCA_COMPONENTS", str(candidate.pca_components))
    text = replace_block(text, "FEATURE_FAMILIES", candidate.feature_families)
    text = replace_block(text, "FEATURE_THRESHOLDS", candidate.thresholds)
    TRAIN_PATH.write_text(text, encoding="utf-8")


def default_feature_families() -> dict[str, bool]:
    return {
        "time_basic": True,
        "time_emg": True,
        "hjorth": True,
        "autoregressive": True,
        "distribution": True,
        "frequency": True,
        "sample_entropy": False,
    }


def candidate_from_result(row: dict, name: str = "seed_best") -> Candidate:
    preprocessing = row["preprocessing"]
    return Candidate(
        name=name,
        window_ms=int(preprocessing["window_ms"]),
        overlap_ratio=float(preprocessing["overlap_ratio"]),
        normalization_strategy=str(preprocessing["normalization_strategy"]),
        rectify_signal=bool(preprocessing["rectify_signal"]),
        detrend_signal=bool(preprocessing["detrend_signal"]),
        remove_dc_offset=bool(preprocessing["remove_dc_offset"]),
        ar_order=int(preprocessing["ar_order"]),
        windows_per_group_class=96,
        enable_cross_channel_summary=True,
        enable_channel_pair_features=False,
        channel_pair_limit=4,
        model_family=str(row["model_family"]),
        thresholds={k: float(v) for k, v in preprocessing["thresholds"].items()},
        use_pca=bool(preprocessing["use_pca"]),
        pca_components=int(preprocessing["pca_components"]),
        feature_families={k: bool(v) for k, v in row["feature_families"].items()},
    )


def mutate_around(best: Candidate) -> list[Candidate]:
    ff = dict(best.feature_families)
    base = dict(best.__dict__)
    candidates = [
        Candidate(**{**base, "model_family": "random_forest", "name": "rf_session_thr003"}),
        Candidate(**{**base, "thresholds": {"zc": 0.02, "ssc": 0.02, "wamp": 0.02}, "name": "extra_trees_thr002"}),
        Candidate(**{**base, "thresholds": {"zc": 0.04, "ssc": 0.04, "wamp": 0.04}, "name": "extra_trees_thr004"}),
        Candidate(**{**base, "window_ms": 225, "name": "extra_trees_win225"}),
        Candidate(**{**base, "window_ms": 175, "name": "extra_trees_win175"}),
        Candidate(**{**base, "overlap_ratio": 0.6, "name": "extra_trees_overlap060"}),
        Candidate(**{**base, "overlap_ratio": 0.4, "name": "extra_trees_overlap040"}),
        Candidate(**{**base, "normalization_strategy": "per_subject", "name": "extra_trees_subject_thr003"}),
        Candidate(**{**base, "rectify_signal": True, "name": "extra_trees_rectified"}),
        Candidate(**{**base, "detrend_signal": False, "name": "extra_trees_no_detrend"}),
        Candidate(**{**base, "ar_order": 6, "name": "extra_trees_ar6"}),
        Candidate(**{**base, "ar_order": 2, "name": "extra_trees_ar2"}),
        Candidate(**{**base, "enable_channel_pair_features": True, "channel_pair_limit": 6, "name": "extra_trees_pair_features"}),
        Candidate(**{**base, "feature_families": {**ff, "frequency": False}, "name": "extra_trees_no_frequency"}),
        Candidate(**{**base, "feature_families": {**ff, "distribution": False}, "name": "extra_trees_no_distribution"}),
        Candidate(**{**base, "feature_families": {**ff, "autoregressive": False}, "name": "extra_trees_no_ar"}),
        Candidate(**{**base, "use_pca": True, "pca_components": 64, "name": "extra_trees_pca64"}),
        Candidate(**{**base, "model_family": "hist_gb", "name": "hist_gb_session_thr003"}),
    ]
    return candidates


def dedupe_candidates(candidates: list[Candidate], rows: list[dict]) -> list[Candidate]:
    seen = {result_signature(row) for row in rows}
    unique: list[Candidate] = []
    emitted: set[tuple] = set()
    for candidate in candidates:
        signature = candidate_signature(candidate)
        if signature in emitted or signature in seen:
            continue
        emitted.add(signature)
        unique.append(candidate)
    return unique


def update_best_file(rows: list[dict]) -> None:
    best = best_grouped_result(rows)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "best_grouped_result": best,
    }
    BEST_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def log_status(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    line = f"[{timestamp}] {message}\n"
    with LOOP_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line)
    print(line, end="", flush=True)


def choose_next_candidate(rows: list[dict]) -> Candidate:
    best_row = best_grouped_result(rows)
    if best_row is None:
        return Candidate(
            name="bootstrap_extra_trees",
            window_ms=200,
            overlap_ratio=0.5,
            normalization_strategy="per_session",
            rectify_signal=False,
            detrend_signal=True,
            remove_dc_offset=True,
            ar_order=4,
            windows_per_group_class=96,
            enable_cross_channel_summary=True,
            enable_channel_pair_features=False,
            channel_pair_limit=4,
            model_family="extra_trees",
            thresholds={"zc": 0.03, "ssc": 0.03, "wamp": 0.03},
            use_pca=False,
            pca_components=32,
            feature_families=default_feature_families(),
        )
    best_candidate = candidate_from_result(best_row)
    candidates = dedupe_candidates(mutate_around(best_candidate), rows)
    if not candidates:
        timestamp = int(time.time())
        return Candidate(**{**best_candidate.__dict__, "name": f"revisit_extra_trees_{timestamp}"})
    return candidates[0]


def run_one_experiment(candidate: Candidate) -> None:
    apply_candidate(candidate)
    log_status(f"starting {candidate.name}")
    with RUN_OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        subprocess.run(
            ["uv", "run", "train.py"],
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    rows = load_results()
    update_best_file(rows)
    best = best_grouped_result(rows)
    latest = rows[-1] if rows else None
    if latest is not None:
        latest_score = latest.get("metrics", {}).get("f1_macro")
        log_status(f"finished {candidate.name} latest_f1={latest_score} best_grouped_f1={grouped_score(best) if best else None}")


def main() -> None:
    LOOP_LOG_PATH.touch(exist_ok=True)
    update_best_file(load_results())
    while True:
        rows = load_results()
        candidate = choose_next_candidate(rows)
        try:
            run_one_experiment(candidate)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            log_status(f"controller_error {type(exc).__name__}: {exc}")
            time.sleep(5)


if __name__ == "__main__":
    main()

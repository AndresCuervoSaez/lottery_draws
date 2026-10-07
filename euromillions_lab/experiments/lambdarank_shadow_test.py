from __future__ import annotations

import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.candidate_v2 import FEATURE_COLUMNS, CandidateFeatureConfig, build_candidate_frame

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "euromillions_full_history.csv"
OUTDIR = BASE / "outputs" / "candidate_v2"
OUTDIR.mkdir(parents=True, exist_ok=True)

SEED = int(os.getenv("SEED", "20261007"))
N_SHADOWS = int(os.getenv("SHADOWS", "100"))
TEST_START_YEAR = int(os.getenv("TEST_START_YEAR", "2014"))

PARAM_GRID = [
    {
        "num_leaves": 7,
        "min_child_samples": 250,
        "learning_rate": 0.03,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 5.0,
    },
    {
        "num_leaves": 15,
        "min_child_samples": 150,
        "learning_rate": 0.03,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 10.0,
    },
    {
        "num_leaves": 15,
        "min_child_samples": 300,
        "learning_rate": 0.02,
        "feature_fraction": 0.70,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 20.0,
    },
]


def load_frame() -> pd.DataFrame:
    draws = pd.read_csv(DATA, parse_dates=["date"])
    frame = build_candidate_frame(draws, CandidateFeatureConfig(min_history=200))
    frame = frame.sort_values(["draw_index", "number"]).reset_index(drop=True)
    frame["year"] = frame["date"].dt.year
    return frame


def group_sizes(df: pd.DataFrame) -> list[int]:
    return df.groupby("draw_index", sort=True).size().astype(int).tolist()


def metrics_for_predictions(df: pd.DataFrame, score: np.ndarray) -> dict:
    work = df[["draw_index", "target"]].copy()
    work["score"] = score
    recalls = {5: 0, 10: 0, 15: 0, 20: 0}
    ge4_top20 = 0
    n_draws = 0

    for _, g in work.groupby("draw_index", sort=True):
        g = g.sort_values("score", ascending=False, kind="mergesort")
        n_draws += 1
        for k in recalls:
            recalls[k] += int(g.head(k)["target"].sum())
        if int(g.head(20)["target"].sum()) >= 4:
            ge4_top20 += 1

    return {
        "draws": n_draws,
        **{f"recall@{k}": recalls[k] / (5.0 * n_draws) for k in recalls},
        **{f"mean_hits@{k}": recalls[k] / n_draws for k in recalls},
        "ge4_top20": ge4_top20,
        "ge4_top20_rate": ge4_top20 / n_draws,
    }


def fit_ranker(train: pd.DataFrame, valid: pd.DataFrame, params: dict, seed: int) -> lgb.LGBMRanker:
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=250,
        random_state=seed,
        verbosity=-1,
        n_jobs=2,
        max_depth=-1,
        **params,
    )
    model.fit(
        train[FEATURE_COLUMNS],
        train["target"].astype(int),
        group=group_sizes(train),
        eval_set=[(valid[FEATURE_COLUMNS], valid["target"].astype(int))],
        eval_group=[group_sizes(valid)],
        eval_at=[5, 10, 15, 20],
        callbacks=[lgb.early_stopping(25, verbose=False)],
    )
    return model


def nested_walk_forward(frame: pd.DataFrame, labels: np.ndarray, seed: int) -> tuple[dict, list[dict]]:
    data = frame.copy()
    data["target"] = labels.astype(np.int8)
    years = sorted(data["year"].unique())
    max_year = max(years)

    fold_rows: list[dict] = []
    all_test = []
    all_scores = []

    for test_year in range(TEST_START_YEAR, max_year + 1):
        valid_year = test_year - 1
        train = data[data["year"] < valid_year]
        valid = data[data["year"] == valid_year]
        test = data[data["year"] == test_year]

        if train.empty or valid.empty or test.empty:
            continue

        best = None
        for p_idx, params in enumerate(PARAM_GRID):
            model = fit_ranker(train, valid, params, seed + test_year * 10 + p_idx)
            valid_score = model.predict(valid[FEATURE_COLUMNS], num_iteration=model.best_iteration_)
            m = metrics_for_predictions(valid, valid_score)
            candidate = (m["recall@20"], m["recall@10"], -p_idx)
            if best is None or candidate > best["key"]:
                best = {
                    "key": candidate,
                    "params_index": p_idx,
                    "params": params,
                    "best_iteration": int(model.best_iteration_ or 250),
                    "valid_metrics": m,
                }

        # Refit on all information available before test year.
        fit_data = data[data["year"] < test_year]
        selected = best["params"]
        final = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=max(20, best["best_iteration"]),
            random_state=seed + test_year * 100,
            verbosity=-1,
            n_jobs=2,
            max_depth=-1,
            **selected,
        )
        final.fit(
            fit_data[FEATURE_COLUMNS],
            fit_data["target"].astype(int),
            group=group_sizes(fit_data),
        )
        score = final.predict(test[FEATURE_COLUMNS])
        tm = metrics_for_predictions(test, score)

        fold_rows.append(
            {
                "test_year": test_year,
                "selected_params": best["params_index"],
                "best_iteration": best["best_iteration"],
                "valid_recall20": best["valid_metrics"]["recall@20"],
                **{f"test_{k}": v for k, v in tm.items()},
            }
        )
        all_test.append(test)
        all_scores.append(score)

    test_df = pd.concat(all_test, ignore_index=True)
    score = np.concatenate(all_scores)
    return metrics_for_predictions(test_df, score), fold_rows


def permute_labels_within_draw(frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    labels = frame["target"].to_numpy(copy=True)
    for _, idx in frame.groupby("draw_index", sort=False).indices.items():
        # Exactly five positives remain in every draw.
        labels[idx] = rng.permutation(labels[idx])
    return labels


def main() -> None:
    print("Building leakage-safe candidate feature matrix...")
    frame = load_frame()
    print(f"rows={len(frame):,}; draws={frame['draw_index'].nunique():,}; features={len(FEATURE_COLUMNS)}")
    print(f"date range={frame['date'].min().date()}..{frame['date'].max().date()}")

    real_labels = frame["target"].to_numpy()
    print("\nRunning real nested LambdaRank test...")
    real, folds = nested_walk_forward(frame, real_labels, SEED)
    print("REAL", json.dumps(real, sort_keys=True))

    pd.DataFrame(folds).to_csv(OUTDIR / "real_folds.csv", index=False)

    rng = np.random.default_rng(SEED + 999)
    shadow_rows = []
    print(f"\nRunning {N_SHADOWS} within-draw shadow lotteries...")
    for s in range(N_SHADOWS):
        shadow_labels = permute_labels_within_draw(frame, rng)
        m, _ = nested_walk_forward(frame, shadow_labels, SEED + 10000 + s)
        shadow_rows.append({"shadow": s, **m})
        if (s + 1) % 10 == 0:
            print(f"shadow {s+1}/{N_SHADOWS}: recall20={m['recall@20']:.5f}")

    shadows = pd.DataFrame(shadow_rows)
    shadows.to_csv(OUTDIR / "shadow_results.csv", index=False)

    # Empirical pipeline-level p values. Add-one correction.
    p_recall20 = (1 + int((shadows["recall@20"] >= real["recall@20"]).sum())) / (N_SHADOWS + 1)
    p_recall10 = (1 + int((shadows["recall@10"] >= real["recall@10"]).sum())) / (N_SHADOWS + 1)
    p_ge4 = (1 + int((shadows["ge4_top20"] >= real["ge4_top20"]).sum())) / (N_SHADOWS + 1)

    random_ge4_p = 0.075919
    exact_ge4 = binomtest(real["ge4_top20"], real["draws"], random_ge4_p, alternative="two-sided")

    summary = {
        "seed": SEED,
        "shadows": N_SHADOWS,
        "test_start_year": TEST_START_YEAR,
        "features": len(FEATURE_COLUMNS),
        "real": real,
        "shadow": {
            "recall20_mean": float(shadows["recall@20"].mean()),
            "recall20_sd": float(shadows["recall@20"].std(ddof=1)),
            "recall20_q05": float(shadows["recall@20"].quantile(0.05)),
            "recall20_q50": float(shadows["recall@20"].quantile(0.50)),
            "recall20_q95": float(shadows["recall@20"].quantile(0.95)),
            "ge4_mean": float(shadows["ge4_top20"].mean()),
            "ge4_q95": float(shadows["ge4_top20"].quantile(0.95)),
        },
        "pipeline_empirical_p": {
            "recall20_one_sided": p_recall20,
            "recall10_one_sided": p_recall10,
            "ge4_top20_one_sided": p_ge4,
        },
        "exact_random_reference": {
            "recall20": 0.40,
            "mean_hits20": 2.0,
            "ge4_top20_probability": random_ge4_p,
            "ge4_top20_expected": real["draws"] * random_ge4_p,
            "ge4_top20_two_sided_p": float(exact_ge4.pvalue),
        },
    }

    with open(OUTDIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("\nSUMMARY")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

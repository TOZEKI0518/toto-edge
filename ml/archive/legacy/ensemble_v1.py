# ==============================================================================
# ensemble_v1.py
# Part 1
# ==============================================================================

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np

LOGGER = logging.getLogger("ensemble_v1")


# ==============================================================================
# Config
# ==============================================================================


@dataclass(slots=True)
class EnsembleConfig:

    project_root: Path

    rf_prediction: Path

    lgb_prediction: Path

    output_dir: Path


def build_default_config(
    root: Path,
):

    return EnsembleConfig(

        project_root=root,

        rf_prediction=(
            root
            / "ml"
            / "rf_v5"
            / "rf_v5_predictions.csv"
        ),

        lgb_prediction=(
            root
            / "ml"
            / "lgbm_v5"
            / "lgbm_v5_predictions.csv"
        ),

        output_dir=(
            root
            / "ml"
            / "ensemble_v1"
        ),

    )


# ==============================================================================
# Logging
# ==============================================================================


def configure_logging():

    logging.basicConfig(

        level=logging.INFO,

        format="%(levelname)s: %(message)s",

    )


# ==============================================================================
# Repository
# ==============================================================================


class Repository:

    @staticmethod
    def load_prediction(
        path: Path,
    ):

        LOGGER.info(
            "Loading %s",
            path,
        )

        return pd.read_csv(path)


# ==============================================================================
# Ensemble
# ==============================================================================


class EnsembleBuilder:

    RF_WEIGHT = 0.50

    LGB_WEIGHT = 0.50

    def build(
        self,
        rf: pd.DataFrame,
        lgb: pd.DataFrame,
    ):

        merge = rf.merge(

            lgb,

            on=[

                "match_card_id",

                "season",

                "round",

                "actual",

            ],

            suffixes=(

                "_rf",

                "_lgb",

            ),

        )

        merge["home"] = (

            merge["prob_home_rf"]

            * self.RF_WEIGHT

            +

            merge["prob_home_lgb"]

            * self.LGB_WEIGHT

        )

        merge["draw"] = (

            merge["prob_draw_rf"]

            * self.RF_WEIGHT

            +

            merge["prob_draw_lgb"]

            * self.LGB_WEIGHT

        )

        merge["away"] = (

            merge["prob_away_rf"]

            * self.RF_WEIGHT

            +

            merge["prob_away_lgb"]

            * self.LGB_WEIGHT

        )

        labels = np.array(

            [

                "H",

                "D",

                "A",

            ]

        )

        probs = merge[
            [
                "home",
                "draw",
                "away",
            ]
        ].values

        merge["prediction"] = labels[
            np.argmax(
                probs,
                axis=1,
            )
        ]

        return merge
# ==============================================================================
# ensemble_v1.py
# Part 2
# Metrics / Export / CLI
# Append below Part1
# ==============================================================================

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)


# ==============================================================================
# Metrics
# ==============================================================================


class EnsembleMetrics:

    @staticmethod
    def evaluate(
        prediction: pd.DataFrame,
    ):

        actual = prediction["actual"]

        pred = prediction["prediction"]

        accuracy = accuracy_score(
            actual,
            pred,
        )

        macro = f1_score(
            actual,
            pred,
            average="macro",
        )

        weighted = f1_score(
            actual,
            pred,
            average="weighted",
        )

        loss = log_loss(

            actual,

            prediction[
                [
                    "home",
                    "draw",
                    "away",
                ]
            ].values,

            labels=np.array(
                [
                    "H",
                    "D",
                    "A",
                ]
            ),

        )

        LOGGER.info("=" * 72)

        LOGGER.info(
            "Accuracy      : %.4f",
            accuracy,
        )

        LOGGER.info(
            "Macro F1      : %.4f",
            macro,
        )

        LOGGER.info(
            "Weighted F1   : %.4f",
            weighted,
        )

        LOGGER.info(
            "Log Loss      : %.4f",
            loss,
        )

        LOGGER.info("=" * 72)

        LOGGER.info(

            "\n%s",

            classification_report(
                actual,
                pred,
            ),

        )

        LOGGER.info(

            "\n%s",

            confusion_matrix(
                actual,
                pred,
            ),

        )

        return {

            "accuracy": accuracy,

            "macro_f1": macro,

            "weighted_f1": weighted,

            "log_loss": loss,

        }


# ==============================================================================
# Export
# ==============================================================================


class EnsembleExporter:

    def __init__(
        self,
        config,
    ):

        self.config = config

    def export(
        self,
        prediction,
        metrics,
    ):

        self.config.output_dir.mkdir(

            parents=True,

            exist_ok=True,

        )

        prediction.to_csv(

            self.config.output_dir
            / "ensemble_predictions.csv",

            index=False,

            encoding="utf-8-sig",

        )

        pd.DataFrame(

            [metrics]

        ).to_csv(

            self.config.output_dir
            / "ensemble_summary.csv",

            index=False,

            encoding="utf-8-sig",

        )

        with open(

            self.config.output_dir
            / "ensemble_summary.json",

            "w",

            encoding="utf-8",

        ) as fp:

            json.dump(

                metrics,

                fp,

                indent=2,

                ensure_ascii=False,

            )

        LOGGER.info(
            "Saved ensemble results."
        )


# ==============================================================================
# CLI
# ==============================================================================


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--project-root",

        type=Path,

        default=Path(__file__).resolve().parents[1],

    )

    return parser.parse_args()


# ==============================================================================
# main
# ==============================================================================


def main():

    configure_logging()

    args = parse_args()

    config = build_default_config(
        args.project_root,
    )

    repo = Repository()

    rf = repo.load_prediction(
        config.rf_prediction,
    )

    lgb = repo.load_prediction(
        config.lgb_prediction,
    )

    ensemble = EnsembleBuilder().build(
        rf,
        lgb,
    )

    metrics = EnsembleMetrics.evaluate(
        ensemble,
    )

    EnsembleExporter(
        config,
    ).export(
        ensemble,
        metrics,
    )

    LOGGER.info("=" * 72)
    LOGGER.info("Project Alpha Ensemble v1 COMPLETE")
    LOGGER.info("=" * 72)


if __name__ == "__main__":
    main()
# ==============================================================================
# ensemble_optimizer.py
# Part 3
# Dynamic Weight Optimizer
# ==============================================================================

from itertools import product


class EnsembleWeightOptimizer:

    """
    Search the best RF/LGB weight.
    """

    @staticmethod
    def search(
        rf: pd.DataFrame,
        lgb: pd.DataFrame,
    ):

        LOGGER.info(
            "Searching Ensemble Weight..."
        )

        best_score = -1.0
        best_prediction = None
        best_rf = 0.5
        best_lgb = 0.5

        labels = np.array(["H", "D", "A"])

        merged = rf.merge(

            lgb,

            on=[

                "match_card_id",

                "season",

                "round",

                "actual",

            ],

            suffixes=("_rf", "_lgb"),

        )

        for rf_weight in np.arange(

            0.0,

            1.01,

            0.05,

        ):

            lgb_weight = 1.0 - rf_weight

            home = (

                merged["prob_home_rf"] * rf_weight

                +

                merged["prob_home_lgb"] * lgb_weight

            )

            draw = (

                merged["prob_draw_rf"] * rf_weight

                +

                merged["prob_draw_lgb"] * lgb_weight

            )

            away = (

                merged["prob_away_rf"] * rf_weight

                +

                merged["prob_away_lgb"] * lgb_weight

            )

            probability = np.column_stack(

                [

                    home,

                    draw,

                    away,

                ]

            )

            prediction = labels[

                np.argmax(

                    probability,

                    axis=1,

                )

            ]

            accuracy = accuracy_score(

                merged["actual"],

                prediction,

            )

            if accuracy > best_score:

                best_score = accuracy

                best_rf = rf_weight

                best_lgb = lgb_weight

                best_prediction = merged.copy()

                best_prediction["home"] = home

                best_prediction["draw"] = draw

                best_prediction["away"] = away

                best_prediction["prediction"] = prediction

        LOGGER.info(

            "Best RF Weight  : %.2f",

            best_rf,

        )

        LOGGER.info(

            "Best LGB Weight : %.2f",

            best_lgb,

        )

        LOGGER.info(

            "Best Accuracy   : %.4f",

            best_score,

        )

        return (

            best_prediction,

            best_rf,

            best_lgb,

            best_score,

        )
# ==============================================================================
# Replace main()
# ==============================================================================


def main():

    configure_logging()

    args = parse_args()

    config = build_default_config(
        args.project_root,
    )

    repo = Repository()

    rf = repo.load_prediction(
        config.rf_prediction,
    )

    lgb = repo.load_prediction(
        config.lgb_prediction,
    )

    prediction, rf_weight, lgb_weight, _ = (

        EnsembleWeightOptimizer.search(

            rf,

            lgb,

        )

    )

    metrics = EnsembleMetrics.evaluate(
        prediction,
    )

    metrics["rf_weight"] = rf_weight
    metrics["lgb_weight"] = lgb_weight

    EnsembleExporter(

        config,

    ).export(

        prediction,

        metrics,

    )

    LOGGER.info("=" * 72)
    LOGGER.info("Project Alpha Dynamic Ensemble COMPLETE")
    LOGGER.info(
        "RF Weight  : %.2f",
        rf_weight,
    )
    LOGGER.info(
        "LGB Weight : %.2f",
        lgb_weight,
    )
    LOGGER.info("=" * 72)


if __name__ == "__main__":
    main()
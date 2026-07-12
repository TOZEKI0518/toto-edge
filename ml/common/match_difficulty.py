from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DifficultyResult:
    difficulty: float
    difficulty_class: str
    difficulty_stars: str
    entropy: float
    confidence: float
    margin: float
    draw_probability: float


class MatchDifficulty:
    """
    H/D/Aの予測確率から、試合の予測難易度を0～1で計算する。

    難易度が高い条件：
    - 3確率が均等に近い
    - 1位と2位の確率差が小さい
    - 最大確率が低い
    - 引き分け確率が高い
    """

    def __init__(
        self,
        entropy_weight: float = 0.50,
        margin_weight: float = 0.25,
        confidence_weight: float = 0.15,
        draw_weight: float = 0.10,
    ) -> None:
        weights = [
            entropy_weight,
            margin_weight,
            confidence_weight,
            draw_weight,
        ]

        if any(weight < 0 for weight in weights):
            raise ValueError(
                "Difficulty weights cannot be negative."
            )

        weight_total = sum(weights)

        if weight_total <= 0:
            raise ValueError(
                "Difficulty weights must total more than zero."
            )

        # 合計が1でなくても自動的に正規化
        self.entropy_weight = entropy_weight / weight_total
        self.margin_weight = margin_weight / weight_total
        self.confidence_weight = confidence_weight / weight_total
        self.draw_weight = draw_weight / weight_total

    @staticmethod
    def _normalize_probabilities(
        probabilities: np.ndarray,
    ) -> np.ndarray:
        probabilities = np.asarray(
            probabilities,
            dtype=float,
        )

        probabilities = np.clip(
            probabilities,
            0.0,
            None,
        )

        total = probabilities.sum()

        if total <= 0:
            return np.full(
                len(probabilities),
                1.0 / len(probabilities),
                dtype=float,
            )

        return probabilities / total

    @staticmethod
    def _normalized_entropy(
        probabilities: np.ndarray,
    ) -> float:
        clipped = np.clip(
            probabilities,
            1e-12,
            1.0,
        )

        entropy = -float(
            np.sum(clipped * np.log(clipped))
        )

        maximum_entropy = float(
            np.log(len(probabilities))
        )

        if maximum_entropy <= 0:
            return 0.0

        return float(
            np.clip(
                entropy / maximum_entropy,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _difficulty_label(
        score: float,
    ) -> tuple[str, str]:
        if score < 0.20:
            return "Very Easy", "★"

        if score < 0.40:
            return "Easy", "★★"

        if score < 0.60:
            return "Medium", "★★★"

        if score < 0.80:
            return "Hard", "★★★★"

        return "Very Hard", "★★★★★"

    def calculate(
        self,
        prob_h: float,
        prob_d: float,
        prob_a: float,
    ) -> DifficultyResult:
        probabilities = self._normalize_probabilities(
            np.array(
                [
                    prob_h,
                    prob_d,
                    prob_a,
                ],
                dtype=float,
            )
        )

        normalized_prob_h = float(probabilities[0])
        normalized_prob_d = float(probabilities[1])
        normalized_prob_a = float(probabilities[2])

        entropy = self._normalized_entropy(
            probabilities
        )

        confidence = float(
            np.max(probabilities)
        )

        ordered = np.sort(probabilities)

        margin = float(
            ordered[-1] - ordered[-2]
        )

        margin_difficulty = 1.0 - margin
        confidence_difficulty = 1.0 - confidence

        difficulty = (
            self.entropy_weight * entropy
            + self.margin_weight * margin_difficulty
            + self.confidence_weight * confidence_difficulty
            + self.draw_weight * normalized_prob_d
        )

        difficulty = float(
            np.clip(
                difficulty,
                0.0,
                1.0,
            )
        )

        difficulty_class, difficulty_stars = (
            self._difficulty_label(difficulty)
        )

        return DifficultyResult(
            difficulty=difficulty,
            difficulty_class=difficulty_class,
            difficulty_stars=difficulty_stars,
            entropy=entropy,
            confidence=confidence,
            margin=margin,
            draw_probability=normalized_prob_d,
        )
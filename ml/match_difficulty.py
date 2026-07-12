from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class DifficultyResult:
    difficulty: float
    difficulty_class: str
    entropy: float
    confidence: float
    margin: float


class MatchDifficulty:

    def __init__(
        self,
        entropy_weight=0.50,
        margin_weight=0.25,
        confidence_weight=0.15,
        draw_weight=0.10,
    ):

        self.entropy_weight = entropy_weight
        self.margin_weight = margin_weight
        self.confidence_weight = confidence_weight
        self.draw_weight = draw_weight

    @staticmethod
    def _entropy(prob):

        p = np.clip(prob, 1e-12, 1)

        entropy = -(p * np.log(p)).sum()

        entropy /= np.log(len(prob))

        return float(entropy)

    @staticmethod
    def _difficulty_class(score):

        if score < 0.20:
            return "Very Easy ★"

        if score < 0.40:
            return "Easy ★★"

        if score < 0.60:
            return "Medium ★★★"

        if score < 0.80:
            return "Hard ★★★★"

        return "Very Hard ★★★★★"

    def calculate(
        self,
        probH,
        probD,
        probA,
    ):

        probs = np.array(
            [
                probH,
                probD,
                probA,
            ]
        )

        entropy = self._entropy(probs)

        confidence = float(np.max(probs))

        ordered = np.sort(probs)

        margin = float(
            ordered[-1] - ordered[-2]
        )

        difficulty = (

            self.entropy_weight * entropy

            + self.margin_weight * (1 - margin)

            + self.confidence_weight * (1 - confidence)

            + self.draw_weight * probD

        )

        difficulty = max(
            0,
            min(
                1,
                difficulty,
            ),
        )

        return DifficultyResult(

            difficulty=difficulty,

            difficulty_class=self._difficulty_class(
                difficulty
            ),

            entropy=entropy,

            confidence=confidence,

            margin=margin,

        )
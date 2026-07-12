import lightgbm as lgb
import numpy as np


def create_draw_classifier():
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=250,
        learning_rate=0.025,
        max_depth=3,
        num_leaves=7,
        min_child_samples=25,
        subsample=0.90,
        colsample_bytree=0.90,
        class_weight="balanced",
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def calculate_dynamic_alpha(
    draw_probabilities: np.ndarray,
    low_threshold: float,
    high_threshold: float,
    low_alpha: float,
    medium_alpha: float,
    high_alpha: float,
) -> np.ndarray:
    alpha = np.full(
        len(draw_probabilities),
        low_alpha,
        dtype=float,
    )

    medium_mask = (
        (draw_probabilities >= low_threshold)
        & (draw_probabilities < high_threshold)
    )

    high_mask = (
        draw_probabilities >= high_threshold
    )

    alpha[medium_mask] = medium_alpha
    alpha[high_mask] = high_alpha

    return alpha


def adjust_draw_probability_dynamic(
    base_probabilities: np.ndarray,
    specialist_draw_probabilities: np.ndarray,
    low_threshold: float,
    high_threshold: float,
    low_alpha: float,
    medium_alpha: float,
    high_alpha: float,
    minimum_draw_probability: float = 0.02,
    maximum_draw_probability: float = 0.65,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = calculate_dynamic_alpha(
        draw_probabilities=(
            specialist_draw_probabilities
        ),
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        low_alpha=low_alpha,
        medium_alpha=medium_alpha,
        high_alpha=high_alpha,
    )

    base_a = base_probabilities[:, 0]
    base_d = base_probabilities[:, 1]
    base_h = base_probabilities[:, 2]

    new_d = (
        (1.0 - alpha) * base_d
        + alpha * specialist_draw_probabilities
    )

    new_d = np.clip(
        new_d,
        minimum_draw_probability,
        maximum_draw_probability,
    )

    remaining = 1.0 - new_d
    non_draw_total = base_a + base_h

    safe_total = np.where(
        non_draw_total > 0,
        non_draw_total,
        1.0,
    )

    adjusted = np.zeros_like(
        base_probabilities,
        dtype=float,
    )

    adjusted[:, 0] = (
        remaining * base_a / safe_total
    )
    adjusted[:, 1] = new_d
    adjusted[:, 2] = (
        remaining * base_h / safe_total
    )

    zero_non_draw = non_draw_total <= 0

    if zero_non_draw.any():
        adjusted[zero_non_draw, 0] = (
            remaining[zero_non_draw] / 2
        )
        adjusted[zero_non_draw, 2] = (
            remaining[zero_non_draw] / 2
        )

    row_sums = adjusted.sum(
        axis=1,
        keepdims=True,
    )

    return adjusted / row_sums, alpha
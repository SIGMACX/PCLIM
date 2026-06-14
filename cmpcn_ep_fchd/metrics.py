from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_classification_metrics(
    labels: list[int] | np.ndarray,
    predictions: list[int] | np.ndarray,
    probabilities: list[list[float]] | np.ndarray,
    num_classes: int,
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    auc = None
    average_precision = None
    if probabilities.ndim == 2 and probabilities.shape[1] == num_classes:
        try:
            if num_classes == 2:
                auc = float(roc_auc_score(labels, probabilities[:, 1]))
                average_precision = float(average_precision_score(labels, probabilities[:, 1]))
            else:
                present = sorted(set(labels.tolist()))
                if len(present) > 1:
                    auc = float(
                        roc_auc_score(
                            labels,
                            probabilities,
                            multi_class="ovr",
                            average="macro",
                            labels=list(range(num_classes)),
                        )
                    )
                class_ap = []
                for class_idx in range(num_classes):
                    binary_labels = (labels == class_idx).astype(int)
                    if binary_labels.max() == binary_labels.min():
                        continue
                    class_ap.append(average_precision_score(binary_labels, probabilities[:, class_idx]))
                if class_ap:
                    average_precision = float(np.mean(class_ap))
        except ValueError:
            auc = None
            average_precision = None

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "precision_macro": float(macro_precision),
        "recall_macro": float(macro_recall),
        "f1_macro": float(macro_f1),
        "auc_macro": auc,
        "average_precision_macro": average_precision,
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=list(range(num_classes)),
        ).tolist(),
        "num_samples": int(labels.size),
    }

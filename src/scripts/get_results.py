import csv
import json
import os
import logging

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
)
from fairlearn.metrics import (
    count,
    false_positive_rate,
    false_negative_rate,
    selection_rate,
    demographic_parity_difference,
    equal_opportunity_difference,
    equalized_odds_difference,
)
from fairlearn.metrics import MetricFrame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def logits_to_probs(logits, config):
    # Posterioir probabilities are calculated differently in some experiments

    if config.get("domain_independent_loss", False):
        per_group = np.split(logits, config["num_groups"], axis=1)
        marginalized = np.sum(per_group, axis=0)
        return softmax(marginalized, axis=1)
    
    if config.get("domain_discriminative_loss", False):
        # Prior shift inference, train distribution
        prior_shift_weight = np.array(
            [
                1088/1072, 1088/16, 17746/17515, 17746/231, 6454/6273, 6454/181, 850/834, 850/16 
            ]
        ) / 100

        probs_yd = softmax(logits, axis=1) * prior_shift_weight
        per_group = np.split(probs_yd, config["num_groups"], axis=1)
        marginalized = np.sum(per_group, axis=0)

        # We shifted probs, apply softmax once more
        return softmax(marginalized, axis=1)

    return softmax(logits, axis=1)


# Fairlearn docs
def compute_error_metric(metric_value, sample_size):
    """Compute standard error of a given metric based on the assumption of
    normal distribution.

    Parameters:
    metric_value: Value of the metric
    sample_size: Number of data points associated with the metric

    Returns:
    The standard error of the metric
    """
    metric_value = metric_value / sample_size
    return 1.96 * np.sqrt(metric_value * (1.0 - metric_value)) / np.sqrt(sample_size)


def false_positive_error(y_true, y_pred):
    """Compute the standard error for the false positive rate estimate."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return compute_error_metric(fp, tn + fp)


def false_negative_error(y_true, y_pred):
    """Compute the standard error for the false negative rate estimate."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return compute_error_metric(fn, fn + tp)
        

def balanced_accuracy_error(y_true, y_pred):
    """Compute the standard error for the balanced accuracy estimate."""
    fpr_error, fnr_error = false_positive_error(y_true, y_pred), false_negative_error(
        y_true, y_pred
    )
    return np.sqrt(fnr_error**2 + fpr_error**2) / 2


if __name__ == "__main__":
    root_dir = "C:\\Users\\Duje\\Desktop\\fer\\8. semestar\\lumen\\rezultati\\02 eksperimenti\\"
    common_csv = "rezultati.csv"
    disagg_csv = "disaggregated.csv"
    experiments = [
        "01 baseline 0304",
        "02 recall ce 0304",
        "04 cielab re based",
        "05 cielab ohem",
        "08 optim params large",
        "10 transformer\\normal", 
        "11 transformer ohem",
        "12 domain discriminative\\new", 
        "13 oversampler",
        "15 focal loss\\new",
        "14 domain independent\\new",
        "16 efficient m\\new",
        "17 masked\\new",
        "18 efficient l\\new",
        "19 oversampler trio\\1 base",
        "19 oversampler trio\\2 ifw, recall_ce",
        "19 oversampler trio\\3 ifw, ohem",
        "20 dino\\new",
        "21 dino oversample",
        "22 dino undersample",
        "23 long train 04"
        "24 dd transformer"
    ]

    logging.info(f"Collecting metrics for {len(experiments)} experiments")
    for exp in experiments:

        eval_dir = os.path.join(root_dir, exp, "eval")
        chkpt = next(os.walk(eval_dir))[1][0]  # checkpoint folder
        logging.info(f"Evaluating checkpoint {chkpt} for experiment {exp}")

        with open(os.path.join(root_dir, exp, "config.json")) as f:
            config = json.load(f)

        y_true = np.load(os.path.join(eval_dir, chkpt, "y_true.npy"))
        logits = np.load(os.path.join(eval_dir, chkpt, "logits.npy"))
        groups = np.load(os.path.join(eval_dir, chkpt, "groups.npy"))
        y_prob = logits_to_probs(logits, config)
        y_pred = np.argmax(y_prob, axis=1)

        prob_path = os.path.join(eval_dir, chkpt, "probs.npy")
        np.save(prob_path, y_prob)
        logging.info(f"Saved posteriror probabilities to {prob_path}")

        metrics = dict(
            count=count,
            f1=f1_score,
            recall=recall_score,
            accuracy=accuracy_score,
            selection_rate=selection_rate,
            balanced_accuracy=balanced_accuracy_score,
            balanced_acc_error=balanced_accuracy_error,
            false_positive_rate=false_positive_rate,
            false_positive_error=false_positive_error,
            false_negative_rate=false_negative_rate,
            false_negative_error=false_negative_error,
        )
        mf = MetricFrame(
            metrics=metrics,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=groups,
        )

        dpd = demographic_parity_difference(
            y_true, y_pred, sensitive_features=groups
        ).item()
        eq_odds = equalized_odds_difference(y_true, y_pred, sensitive_features=groups)
        eq_opp = equal_opportunity_difference(
            y_true, y_pred, sensitive_features=groups
        ).item()

        diffs = mf.difference()[
            [
                "f1",
                "recall",
                "accuracy",
                "balanced_accuracy",
                "false_positive_rate",
                "false_negative_rate",
            ]
        ]
        diffs = diffs.rename(
            dict(
                f1="f1_diff",
                recall="recall_diff",
                accuracy="accuracy_diff",
                balanced_accuracy="balanced_acc_diff",
                false_positive_rate="fpr_diff",
                false_negative_rate="fnr_diff",
            )
        )

        # One row for each experiment
        fair = pd.Series([dpd, eq_odds, eq_opp], index=["dpd", "eq_odds", "eq_opp"])
        fair = fair.add(diffs, fill_value=0)
        result = mf.overall.add(fair, fill_value=0)

        header = ["experiment"] + result.keys().to_list()
        if not os.path.isfile(common_csv):
            with open(common_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)

        row = [exp] + result.to_list()
        with open(common_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        logging.info(f"Added row to {common_csv}")

        # Disaggregated metrics
        group = mf.by_group
        group = group.rename(columns={"sensitive_feature_0": "group"})
        group.to_csv(os.path.join(eval_dir, chkpt, disagg_csv))
        logging.info(f"Saved disaggregated metrics to {disagg_csv}")

    logging.info("Done")

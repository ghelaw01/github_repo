# src/components/fairness_auditor.py

import pandas as pd
from sklearn.metrics import roc_auc_score

class FairnessAuditor:
    """A class to audit model fairness across different demographic groups."""

    def __init__(self, model, data: pd.DataFrame, sensitive_attribute: str, label: str):
        """Initializes the FairnessAuditor.

        Args:
            model: The trained model to be audited.
            data: The dataset to use for auditing.
            sensitive_attribute: The column name of the sensitive attribute (e.g., 'race').
            label: The column name of the ground truth label.
        """
        self.model = model
        self.data = data
        self.sensitive_attribute = sensitive_attribute
        self.label = label
        self.groups = self.data[self.sensitive_attribute].unique()

    def calculate_demographic_parity(self) -> dict:
        """Calculates demographic parity (statistical parity).

        Returns:
            A dictionary with the positive prediction rate for each group.
        """
        parity_results = {}
        for group in self.groups:
            group_data = self.data[self.data[self.sensitive_attribute] == group]
            if not group_data.empty:
                predictions = self.model.predict(group_data.drop(columns=[self.label, self.sensitive_attribute]))
                positive_rate = predictions.mean()
                parity_results[group] = positive_rate
        return parity_results

    def calculate_equalized_odds(self) -> dict:
        """Calculates equalized odds (true positive rate parity).

        Returns:
            A dictionary with the true positive rate for each group.
        """
        odds_results = {}
        for group in self.groups:
            group_data = self.data[self.data[self.sensitive_attribute] == group]
            true_positives = group_data[group_data[self.label] == 1]
            if not true_positives.empty:
                predictions = self.model.predict(true_positives.drop(columns=[self.label, self.sensitive_attribute]))
                tpr = predictions.mean()
                odds_results[group] = tpr
        return odds_results

    def calculate_auc_by_group(self) -> dict:
        """Calculates the AUC for each demographic group.

        Returns:
            A dictionary with the AUC score for each group.
        """
        auc_results = {}
        for group in self.groups:
            group_data = self.data[self.data[self.sensitive_attribute] == group]
            if len(group_data[self.label].unique()) > 1:
                X_group = group_data.drop(columns=[self.label, self.sensitive_attribute])
                y_group = group_data[self.label]
                y_pred_proba = self.model.predict_proba(X_group)[:, 1]
                auc_results[group] = roc_auc_score(y_group, y_pred_proba)
        return auc_results

    def run_audit(self) -> dict:
        """Runs a full fairness audit.

        Returns:
            A dictionary containing the results of all fairness metrics.
        """
        audit_report = {
            "demographic_parity": self.calculate_demographic_parity(),
            "equalized_odds": self.calculate_equalized_odds(),
            "auc_by_group": self.calculate_auc_by_group()
        }
        return audit_report

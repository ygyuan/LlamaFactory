import os
import argparse
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from sklearn.metrics import precision_recall_curve
import torch
import logging
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Constants
LABEL_MAP = {
    '100': 0,
}

@dataclass
class EvaluationMetrics:
    """Data class to store evaluation metrics."""
    confusion_matrix: torch.Tensor
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    false_positive_rate: float
    hit_rate: float

class VQAEvaluator:
    """Class to handle VQA evaluation."""
    
    def __init__(self, args: argparse.Namespace):
        """Initialize evaluator with command line arguments."""
        self.args = args
        self.results: Dict[str, float] = {}
        self.test_split: List[Dict] = []
        self.datalist: Dict[str, Dict] = {}
        
    def read_data(self) -> None:
        """Read prediction results and test data."""
        src = self.args.mejson
        logger.info(f"Reading input file: {src}")
        
        try:
            # Read prediction results (jsonl format)
            with open(src, 'r') as f:
                results = []
                for line in f:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                
            # Read test data (json format)
            with open(self.args.split, 'r') as f:
                self.test_split = json.load(f)
                
            # Convert results format
            self.results = {x['question_id']: x['score'] for x in results}
            self.datalist = {data['id']: data for data in self.test_split}
            
            logger.info(f'Total results: {len(self.results)}, Total split: {len(self.test_split)}')
                    
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error reading data: {str(e)}")
            raise

    def process_data(self) -> Tuple[List[str], List[str], List[List[float]], List[int]]:
        """Process test data and prepare evaluation metrics."""
        y_id = []
        y_audit_label = []
        multi_y_scores = []
        y_multi_true = []

        for x in self.test_split:
            if x['id'] not in self.results:
                continue
                
            score = self.results[x['id']]
            label = str(x['label'])            
            mlab = LABEL_MAP.get(label, 1)
            
            y_id.append(x['id'])
            y_audit_label.append(label)
            multi_y_scores.append(score)
            y_multi_true.append(mlab)

        return y_id, y_audit_label, multi_y_scores, y_multi_true

    def calculate_metrics(self, confusion: torch.Tensor) -> EvaluationMetrics:
        """Calculate evaluation metrics from confusion matrix."""
        eps = 1e-12
        total = confusion.sum().item()
        
        precision = confusion[1, 1].item() / (confusion[1, :].sum().item() + eps)
        recall = confusion[1, 1].item() / (confusion[:, 1].sum().item() + eps)
        f1_score = 2 * precision * recall / (precision + recall + eps)
        accuracy = confusion[1, 1].item() / (confusion[1, :].sum().item() + eps)
        false_positive_rate = confusion[1, 0].item() / (confusion[:, 0].sum().item() + eps)
        hit_rate = 1 - (confusion[0, :].sum().item() / (total + eps))
        
        return EvaluationMetrics(
            confusion_matrix=confusion,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy,
            false_positive_rate=false_positive_rate,
            hit_rate=hit_rate
        )

    def get_confusion_matrix(self, threshold: float, y_true: List[int], y_scores: List[float]) -> torch.Tensor:
        """Calculate confusion matrix for given threshold."""
        confusion = torch.zeros(2, 2, dtype=torch.long)
        
        for score, true_label in zip(y_scores, y_true):
            pred_label = 1 if score >= threshold else 0
            confusion[pred_label, true_label] += 1
            
        return confusion

    def evaluate(self) -> None:
        """Run the complete evaluation pipeline."""
        try:
            self.read_data()
            y_id, y_audit_label, multi_y_scores, y_multi_true = self.process_data()
            self.evaluate_black_samples(y_id, y_audit_label, multi_y_scores, y_multi_true)
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
            raise

    @staticmethod
    def _find_best_f1_threshold(precision: np.ndarray, recall: np.ndarray, thresholds: np.ndarray, label: int) -> Tuple[float, float, float, float]:
        """Find threshold that achieves best F1 score and return corresponding metrics."""
        eps = 1e-12
        f1_scores = 2 * (precision * recall) / (precision + recall + eps)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_precision = precision[best_idx]
        best_recall = recall[best_idx]
        best_f1 = f1_scores[best_idx]
        logger.info(f"Label_{label}_Best_Threshold: {best_threshold:.6f}")
        logger.info(f"Label_{label}_Best_Precision: {best_precision:.4f}")
        logger.info(f"Label_{label}_Best_Recall: {best_recall:.4f}")
        logger.info(f"Label_{label}_Best_F1: {best_f1:.4f}")
        return best_threshold, best_precision, best_recall, best_f1

    @staticmethod
    def _output_metrics_at_precision_intervals(precision: np.ndarray, recall: np.ndarray, thresholds: np.ndarray, label: int, total_positive: int, total_samples: int) -> None:
        """Output metrics at regular precision intervals of 0.05."""
        eps = 1e-12
        f1_scores = 2 * (precision * recall) / (precision + recall + eps)
        
        # Create precision intervals
        target_precisions = np.array([0.250, 0.500, 0.750, 0.900, 0.950, 0.980, 0.990, 0.995])
        
        logger.info(f"\nMetrics at precision intervals for Label_{label}:")
        logger.info("Precision\tRecall\tF1\tThreshold\tNegativePrecisionImprovement\tNegativeRecallReduction")
        logger.info("-" * 50)
        
        # Find valid indices where precision is achievable
        valid_indices = []
        for target_precision in target_precisions:
            idx = np.abs(precision - target_precision).argmin()
            if idx < len(precision) and idx < len(recall) and idx < len(f1_scores) and idx < len(thresholds):
                valid_indices.append((target_precision, idx))

        total_negative = total_samples - total_positive
        ori_negative_precision = total_negative / (total_negative + total_positive)
        logger.info(f"Total_positive: {total_positive}, Total_negative: {total_negative}, Original Negative Precision: {ori_negative_precision:.4f}")
        # Output metrics for valid indices
        for target_precision, idx in valid_indices:
            TP = int(recall[idx] * total_positive)
            FP = int((TP / (precision[idx] + eps)) - TP)
            # logger.info(f"TP: {TP}, FP: {FP}, TN: {total_negative-FP}, FN: {total_positive-TP}")
            NegativePrecisionImprovement = (total_negative-FP) / (total_samples-FP-TP) # - ori_negative_precision
            NegativeRecallReduction = FP/(total_negative + eps)
            logger.info(f"{precision[idx]:.4f}\t{recall[idx]:.4f}\t{f1_scores[idx]:.4f}\t{thresholds[idx]:.4f}\t{NegativePrecisionImprovement:.4f}\t{NegativeRecallReduction:.4f}")

    def evaluate_black_samples(self, y_id: List[str], y_audit_label: List[str], 
                             multi_y_scores: List[List[float]], y_multi_true: List[int]) -> None:
        """Evaluate black sample performance and overall reduction effect."""
        label = 1  # Focus on label 1 (black samples)
        y_true_idx = [int(j == label) for j in y_multi_true]
        y_scores_idx = [j[label] for j in multi_y_scores]  # Extract scores for label 1
        
        precision, recall, thresholds = precision_recall_curve(y_true_idx, y_scores_idx, drop_intermediate=True)
        
        # Output metrics at regular precision intervals
        self._output_metrics_at_precision_intervals(precision, recall, thresholds, label, sum(y_true_idx), len(y_scores_idx))
        
        self._find_best_f1_threshold(precision, recall, thresholds, label)
        
        # Use provided threshold if available, otherwise calculate from precision or recall
        if hasattr(self.args, 'threshold') and self.args.threshold is not None:
            threshold = self.args.threshold
            logger.info(f"Using provided threshold: {threshold:.6f}")
        # elif hasattr(self.args, 'recall') and self.args.recall is not None:
        #    threshold = self._find_threshold_for_recall(recall, thresholds, self.args.recall)
        #    logger.info(f"Calculated threshold from recall {self.args.recall}: {threshold:.6f}")
        else:
            threshold = self._find_threshold_for_precision(precision, thresholds, self.args.precision)
            logger.info(f"Calculated threshold from precision {self.args.precision}: {threshold:.6f}")
        
        confusion = self.get_confusion_matrix(threshold, y_true_idx, y_scores_idx)
        metrics = self.calculate_metrics(confusion)
        
        # Calculate overall reduction effect
        total_samples = confusion.sum().item()
        reduced_samples = confusion[0, :].sum().item()  # Samples classified as negative
        
        # Log results
        logger.info(f"Label_{label}_Results:")
        logger.info(f"Label_{label}_Threshold: {threshold:.6f}")
        logger.info(f"Label_{label}_Total_Samples: {total_samples}")
        logger.info(f"Label_{label}_Reduced_Samples: {reduced_samples}")
        logger.info(f"Label_{label}_Reduction_Rate: {1.0 - metrics.hit_rate:.4f}")
        logger.info(f"Label_{label}_Precision: {metrics.precision:.4f}")
        logger.info(f"Label_{label}_Recall: {metrics.recall:.4f}")
        logger.info(f"Label_{label}_F1: {metrics.f1_score:.4f}")
        logger.info(f"Label_{label}_Confusion_Matrix:\n{metrics.confusion_matrix}")

        # Write detailed sample information to file
        suffix = "_".join(self.args.mejson.split('/')[-1].split('_')[1:-1])
        output_file = os.path.join(os.path.dirname(self.args.mejson), f'{suffix}_zhibai_samples_{label}.txt')
        
        with open(output_file, 'w') as f:
            f.write("ID\tScore\tTrue_Label\tPredicted_Label\tcelue\tUrl\tAudit_Label\tComment\tBackground\n") 
            
            # Sort by score in descending order
            sorted_data = sorted(zip(y_id, y_scores_idx, y_true_idx, y_audit_label), 
                              key=lambda x: x[1], reverse=True)
            
            for id, score, true_label, audit_label in sorted_data:
                data = self.datalist[id]
                pred_label = 1 if score >= threshold else 0
                
                # Parse comment and background info
                celue = data.get("strHitkeywords", "")
                url = data.get("strHitEvents", "")
                comment = data.get("text", "")
                background = ""
                f.write(f"{id}\t{score:.6f}\t{true_label}\t{pred_label}\t{celue}\t{url}\t{audit_label}\t{comment}\t{background}\n")
        
        logger.info(f"\nblack sample evaluation results written to {output_file}")

    @staticmethod
    def _find_threshold_for_precision(precision: np.ndarray, thresholds: np.ndarray, target_precision: float) -> float:
        """Find threshold that achieves target precision."""
        idx = np.argmin(np.abs(precision - target_precision))
        if idx == len(thresholds):
            idx = len(thresholds) - 1
        return thresholds[idx]

    @staticmethod
    def _find_threshold_for_recall(recall: np.ndarray, thresholds: np.ndarray, target_recall: float) -> float:
        """Find threshold that achieves target recall."""
        idx = np.argmin(np.abs(recall - target_recall))
        if idx == len(thresholds):
            idx = len(thresholds) - 1
        return thresholds[idx]

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate model predictions for VQA task')
    parser.add_argument('--mejson', type=str, required=True, help='Input JSON file path')
    parser.add_argument('--split', type=str, required=True, help='Test split file path')
    parser.add_argument('--precision', type=float, default=0.990, help='Precision threshold')
    parser.add_argument('--recall', type=float, default=None, help='Recall threshold')
    parser.add_argument('--threshold', type=float, default=None, help='Direct threshold value (overrides precision/recall-based calculation)')
    return parser.parse_args()

if __name__ == '__main__':
    try:
        args = parse_args()
        evaluator = VQAEvaluator(args)
        evaluator.evaluate()
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

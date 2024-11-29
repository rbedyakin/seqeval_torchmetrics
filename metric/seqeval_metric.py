from torchmetrics import Metric
import torch
from torch import Tensor
from typing import Dict, Tuple, List, Optional, Union, Callable
from torchmetrics.utilities.compute import _safe_divide
from metric.utils import classification_report


def precision_recall_f1(
        pred_sum: torch.Tensor, tp_sum: torch.Tensor, true_sum: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute precision, recall, f1.

        Args:
            pred_sum: Tensor of predicted positives per type.
            tp_sum: Tensor of true positives per type.
            true_sum: Tensor of actual positives per type.
        
        Returns:
            precision: Tensor of precision per type.
            recall: Tensor of recall per type. 
            f1: Tensor of f1 per type.
        
        Note: Metric value is substituted as 0 when encountering zero division."""

    precision = _safe_divide(num=tp_sum, denom=pred_sum, zero_division=0.0)
    recall = _safe_divide(num=tp_sum, denom=true_sum, zero_division=0.0)
    f1 = _safe_divide(num=2 * tp_sum,
                      denom=pred_sum + true_sum,
                      zero_division=0.0)

    return precision, recall, f1


class Seqeval(Metric):
    pred_sum: Union[List[Tensor], Tensor]
    tp_sum: Union[List[Tensor], Tensor]
    true_sum: Union[List[Tensor], Tensor]

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(self,
                 labels: List[str],
                 suffix: bool = False,
                 scheme: Optional[str] = None,
                 mode: Optional[str] = None,
                 stage: Optional[str] = None,
                 **kwargs):
        """Init Metric

        Args:
            labels: List of tags, for example ["LOC", "PER", "ORG"]. 
                Metrics will be computed for these tags.
            suffix: True if the IOB prefix is after type, False otherwise. 
                default: False
            scheme: Specify target tagging scheme. Should be one of ["IOB1", "IOB2", "IOE1", "IOE2", "IOBES", "BILOU"].
                default: None
            mode: Whether to count correct entity labels with incorrect I/B tags as true positives or not.
                If you want to only count exact matches, pass mode="strict". 
                default: None.
            stage: Optional prefix for keys in output dict
                default: None
        """
        super().__init__(**kwargs)

        self.labels = labels
        self.suffix = suffix
        self.scheme = scheme
        self.mode = mode
        self.stage = stage

        self.labels2ind = {v: i for i, v in enumerate(self.labels)}

        self._create_state(size=len(self.labels))

    def update(self, preds: List[List[str]], target: List[List[str]]) -> None:
        """Update state with predictions and targets.
        
        Args:
            preds: List of List of predicted labels (Estimated targets as returned by a tagger)
            target: List of List of reference labels (Ground truth (correct) target values)
        """

        target_names, pred_sum, tp_sum, true_sum = classification_report(
            y_true=target,
            y_pred=preds,
            suffix=self.suffix,
            scheme=self.scheme,
            mode=self.mode,
        )

        for row in zip(target_names, pred_sum, tp_sum, true_sum):
            ind = self.labels2ind.get(row[0], None)
            if ind is not None:
                self.pred_sum[ind] += torch.tensor(row[1])
                self.tp_sum[ind] += torch.tensor(row[2])
                self.true_sum[ind] += torch.tensor(row[3])

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute the final statistics.
        
        Returns:
            'metrics': dict. Summary of the scores for overall and per type
                Overall:
                    'precision': precision,
                    'recall': recall,
                    'f1': F1 score, also known as balanced F-score or F-measure,
                Per type:
                    'precision': precision,
                    'recall': recall,
                    'f1': F1 score, also known as balanced F-score or F-measure"""

        metrics = {}
        precision, recall, f1 = precision_recall_f1(pred_sum=self.pred_sum,
                                                    tp_sum=self.tp_sum,
                                                    true_sum=self.true_sum)
        for k, array in zip(['precision', 'recall', 'f1', 'number'],
                            [precision, recall, f1, self.true_sum]):
            for label, value in zip(self.labels, array):
                key = f'{label}_{k}'
                if self.stage is not None:
                    key = f'{self.stage}_{key}'
                metrics[key] = value

        precision, recall, f1 = precision_recall_f1(
            pred_sum=self.pred_sum.sum(),
            tp_sum=self.tp_sum.sum(),
            true_sum=self.true_sum.sum())
        for k, v in zip(['precision', 'recall', 'f1'],
                        [precision, recall, f1]):
            key = f'overall_{k}'
            if self.stage is not None:
                key = f'{self.stage}_{key}'
            metrics[key] = v

        return metrics

    # define common functions
    def _create_state(
        self,
        size: int,
    ) -> None:
        """Initialize the states for the different statistics."""

        self.add_state("pred_sum",
                       torch.zeros(size, dtype=torch.long),
                       dist_reduce_fx="sum")
        self.add_state("tp_sum",
                       torch.zeros(size, dtype=torch.long),
                       dist_reduce_fx="sum")
        self.add_state("true_sum",
                       torch.zeros(size, dtype=torch.long),
                       dist_reduce_fx="sum")

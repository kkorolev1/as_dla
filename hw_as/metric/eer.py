from hw_as.base.base_metric import BaseMetric
from hw_as.metric.utils import compute_eer


class EER(BaseMetric):
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
    def __call__(self, logits, target, **kwargs):
        logits = logits.detach().cpu().numpy()[..., 1]
        target = target.detach().cpu().numpy()
        bonafide_scores = logits[target == 1]
        other_scores = logits[target == 0]
        return compute_eer(bonafide_scores, other_scores)[0]
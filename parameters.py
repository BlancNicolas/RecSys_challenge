from enum import Enum


HYBRID_ICF_CB_UCF_WEIGHTS = [0.8, 0.1, 0.1]


class Fit_Parameters(Enum):
    CB_TOPK = 50
    ICF_TOPK = 300
    UCF_TOPK = 300

    CB_SHRINK = 100
    ICF_SHRINK = 100
    UCF_SHRINK = 200

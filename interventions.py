from dataclasses import dataclass
import torch

@dataclass
class PermutationIntervention:
    before_layer_num: int       # The layer number of the transformer (1st is 0) before which to perform permutation
    perm: torch.tensor = None   # int tensor of size (seq_length, ) that speficies the permutation to apply to the layer. 
                            # Can be set to None until length of prompt is known, at which point set it to a permutation of the correct length


@dataclass
class LearningRateIntervention:
    start_layer_num: int
    lr_scale: float

@dataclass
class SkipBlockIntervention:
    layer_num: int

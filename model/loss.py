import torch
import torch.nn.functional as F
from torch import nn


def nll_loss(output, target):
    return F.nll_loss(output, target)


class CrossEntropyLoss(object):
    def __init__(self):
        pass

    def __call__(self, output, target):
        return F.cross_entropy(output, target)


class PerInstanceTemperatureScalingLoss(object):
    def __init__(self, lambda_weight=0.1):
        self.label_counts = None
        self.softplus = nn.Softplus()
        self.lambda_weight = lambda_weight
        self.target_temperature = None

    def __call__(self, output, target):
        logits, temperature = output

        if self.target_temperature is None:
            counts = torch.from_numpy(self.label_counts)
            max_count = counts.max().item()
            frac = counts / max_count
            self.target_temperature = (1 - (frac.log())).to(logits.device).float()



        # Use softplus to shift network outputs into [1, Inf] range
        temperature = self.softplus(temperature.squeeze()) + 1.


        logits = self.softplus(logits)
        calibrated_logits = logits / temperature[:, None]

        ce_loss = F.cross_entropy(calibrated_logits, target)

        target_temperature = self.target_temperature[target]
        temperature_loss = F.mse_loss(temperature, target_temperature)

        loss = ce_loss + self.lambda_weight * temperature_loss

        return loss


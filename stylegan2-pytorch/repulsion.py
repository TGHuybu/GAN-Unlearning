import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader


def calc_scaling_fact(num):
    fact = 0
    if (num > 1):
        while (num//10 > 0):
            num = num // 10
            fact += 1
    return fact


class RepulsionLoss:
    def __init__(self, models, loss_type="l2exp", alpha=0.005, weight=10, dweight=1, do_scale=False):
        self.weight = weight
        self.device = torch.device('cuda')
        self.loss = loss_type
        self.alpha = alpha
        self.dweight = dweight
        self.do_scale_distance = do_scale

        if self.loss not in ["l2inv", "l2neg", "l2exp"]:
            self.loss = "l2exp"
            
        # Update mean params
        self.adapted_models = []
        for model in models:
            for param_name, param in model.named_parameters():
                _buff_param_name = param_name.replace('.', '__')
                model.register_buffer(_buff_param_name+'_estimated_mean', param.data.clone())
            self.adapted_models.append(model)

        # print(self.weight, type(self.weight))
        # print(self.alpha, type(self.alpha))


    def _compute_repulsion_loss(self, unlearning_model):
        try:
            # total_rloss = 0
            total_diff = 0
            for adapted_model in self.adapted_models:
                losses = []
                n_params = 0
                for param_name, param in unlearning_model.named_parameters():
                    _buff_param_name = param_name.replace('.', '__')
                    estimated_mean = getattr(adapted_model, f'{_buff_param_name}_estimated_mean')
                    losses.append(((param - estimated_mean) ** 2).sum())
                    n_params += param.numel()

                # add 1 to push a bit further
                sum_diff_squared = self.dweight * (sum(losses) / n_params)
                print(f"-- numel: {n_params}, mean diff: {sum_diff_squared}")

                # idea from official code, how does it work? idk...
                # not sure if this should be used, should be turned off
                # for all run in the thesis (i think)
                if self.do_scale_distance:
                    scale_factor = 10**calc_scaling_fact(sum_diff_squared)
                    sum_diff_squared = sum_diff_squared / scale_factor
                    print("-- SCALED sum diff squared: ", sum_diff_squared)

                total_diff += sum_diff_squared
            
            print(f"-- TOTAL DIFF: {total_diff}")

            if self.loss == "l2inv":
                total_rloss = self.weight * (1 / total_diff)
            elif self.loss == "l2neg":
                total_rloss = self.weight * (-total_diff)
            elif self.loss == "l2exp":
                total_rloss = self.weight * torch.exp(-self.alpha * total_diff)
            return total_rloss
        
        except AttributeError:
            return 0

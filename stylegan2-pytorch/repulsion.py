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
    def __init__(self, models, loss_type="l2exp", alpha=0.3, beta=0.7, exp_lambda=0.05, weight=10, dweight=1, do_scale=False):
        self.weight = weight
        self.device = torch.device('cuda')
        self.loss = loss_type
        self.alpha = alpha
        self.beta = beta
        self.exp_lambda = exp_lambda
        self.dweight = dweight
        self.do_scale_distance = do_scale

        if self.loss not in ["l2inv", "l2neg", "l2exp", "l2ens"]:
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
            total_diff, ang_vals = 0., []
            for adapted_model in self.adapted_models:
                diff2, n_params = 0., 0
                dot = nu = nv = 0  # For angular

                for param_name, param in unlearning_model.named_parameters():
                    _buff_param_name = param_name.replace('.', '__')
                    p_neg = getattr(adapted_model, f'{_buff_param_name}_estimated_mean')

                    diff2 += ((param - p_neg) ** 2).sum()
                    n_params += param.numel()

                    # ---- Angular thành phần (dùng cos giữa θ & θ_N) ----
                    if self.loss == "l2ens":
                        u = param.view(-1)
                        v = p_neg.view(-1)
                        dot += (u * v).sum()
                        nu  += (u * u).sum()
                        nv  += (v * v).sum()

                d2  = self.dweight * (diff2 / n_params)
                total_diff += d2
                # print(f"-- numel: {n_params}, mean diff: {sum_diff_squared}")

                if self.loss == "l2ens":
                    cos = dot / (nu.sqrt() * nv.sqrt() + 1e-8)
                    ang_vals.append(F.relu(cos))    # max(0, cos)

                # idea from official code, how does it work? idk...
                # not sure if this should be used, should be turned off
                # for all run in the thesis (i think)
                # if self.do_scale_distance:
                #     scale_factor = 10**calc_scaling_fact(sum_diff_squared)
                #     sum_diff_squared = sum_diff_squared / scale_factor
                #     print("-- SCALED sum diff squared: ", sum_diff_squared)
            
            print(f"-- TOTAL DIFF: {total_diff}")

            if self.loss == "l2inv":
                total_rloss = self.weight * (1 / total_diff)
            elif self.loss == "l2neg":
                total_rloss = self.weight * (-total_diff)
            elif self.loss == "l2exp":
                total_rloss = self.weight * torch.exp(-self.exp_lambda * total_diff)
            elif self.loss == "l2ens":
                loss_exp = self.weight * torch.exp(-self.exp_lambda * total_diff)
                loss_ang = torch.stack(ang_vals).mean()
                total_rloss = self.alpha * loss_ang + self.beta * loss_exp
            return total_rloss
        
        except AttributeError:
            return torch.tensor(0., device=self.device)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
import numpy as np
from torch.utils.data import DataLoader
from noise import make_noise, mixing_noise
from tqdm import tqdm

class ElasticWeightConsolidation:

    def __init__(self, model, weight=1000000):
        self.model = model
        self.weight = weight
        self.device = torch.device('cuda')

    def _update_mean_params(self):
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.model.register_buffer(_buff_param_name+'_estimated_mean', param.data.clone())

    # def _update_fisher_params(self ,dis, target, num, args):
    #     log_liklihoods = []
    #     for i in range(num) :
    #         noise = mixing_noise(args.batch, args.latent, args.mixing, self.device)

    #         criterion = nn.BCEWithLogitsLoss()
                
    #         generated,_ = self.model(noise)
    #         result = dis(generated)
    #         output = criterion(result, target)
    #         log_liklihoods.append(output)

    #     log_likelihood = torch.stack(log_liklihoods).mean()
    #     grad_log_liklihood = autograd.grad(log_likelihood, self.model.parameters())
    #     _buff_param_names = [param[0].replace('.', '__') for param in self.model.named_parameters()]
    #     for _buff_param_name, param in zip(_buff_param_names, grad_log_liklihood):
    #         self.model.register_buffer(_buff_param_name+'_estimated_fisher', param.data.clone() ** 2)

    # def _update_fisher_params(self, dis, target, num, args):
    #     # Khởi tạo một dict để cộng dồn bình phương gradient cho mỗi tham số
    #     fisher_info = {name: torch.zeros_like(param.data) for name, param in self.model.named_parameters()}
    #     criterion = nn.BCEWithLogitsLoss()

    #     for i in tqdm(range(num)):
    #         # Sinh noise cho mỗi vòng lặp, sử dụng lại args truyền vào
    #         noise = mixing_noise(args.batch, args.latent, args.mixing, self.device)
    #         generated, _ = self.model(noise)
    #         result = dis(generated)
    #         output = criterion(result, target)
    #         # Tính gradient ngay sau vòng lặp này và giải phóng đồ thị tính toán
    #         grads = autograd.grad(output, self.model.parameters(), retain_graph=False)
    #         for (name, _), grad in zip(self.model.named_parameters(), grads):
    #             fisher_info[name] += grad.data.clone() ** 2

    #     # Trung bình hóa các giá trị fisher
    #     for name, param in self.model.named_parameters():
    #         fisher = fisher_info[name] / num
    #         _buff_param_name = name.replace('.', '__')
    #         self.model.register_buffer(_buff_param_name + '_estimated_fisher', fisher)

    def _update_fisher_params(self, dis, target, num, args):        
        # Khởi tạo dictionary lưu Fisher cho từng parameter
        fisher_info = {}
        batch_count = 0
        criterion = nn.BCEWithLogitsLoss()

        for i in tqdm(range(num)):
            # Sinh noise ngẫu nhiên để tạo dữ liệu mẫu
            noise = mixing_noise(args.batch, args.latent, args.mixing, self.device)
            # Sinh dữ liệu mẫu từ generator
            fake_data,_ = self.model(noise)
            fake_pred = dis(fake_data)
           
            loss = criterion(fake_pred, target)
            
            # Reset gradient của generator
            self.model.zero_grad()
            loss.backward()
            
            # Tích lũy bình phương gradient cho từng tham số
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_sq = param.grad.detach() ** 2
                    if name not in fisher_info:
                        fisher_info[name] = grad_sq
                    else:
                        fisher_info[name] += grad_sq
            
            batch_count += 1

        # Trung bình hóa giá trị Fisher qua các batch và lưu vào buffer của generator
        for name, param in self.model.named_parameters():
            if name in fisher_info:
                fisher_estimate = fisher_info[name] / batch_count
                # Đổi tên buffer theo quy ước: thay dấu chấm bằng dấu gạch dưới đôi
                buffer_name = name.replace('.', '__') + '_estimated_fisher'
                self.model.register_buffer(buffer_name, fisher_estimate.clone())


    def register_ewc_params(self, dis, target,num, args):
        self._update_fisher_params(dis, target,num, args)
        self._update_mean_params()

    def _compute_consolidation_loss(self):
        try:
            losses = []
            for param_name, param in self.model.named_parameters():
                _buff_param_name = param_name.replace('.', '__')
                estimated_mean = getattr(self.model, '{}_estimated_mean'.format(_buff_param_name))
                estimated_fisher = getattr(self.model, '{}_estimated_fisher'.format(_buff_param_name))
                losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())
            return self.weight * sum(losses)
        except AttributeError:
            return 0

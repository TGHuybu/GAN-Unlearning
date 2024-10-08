from copy import deepcopy

import torch
import torch.nn as nn
import torch.autograd as autograd


class EWC:
    """
    Implementation for Elastic Weight Consolidation
    Reference link:
        https://discuss.pytorch.org/t/implementation-of-elastic-weight-consolidation-fisher-information-matrix/9309/3 
    """
    def __init__(self, model_anchor):
        self._model_anchor = model_anchor
        self._params_anchor = {n: p for n, p in model_anchor.named_parameters()}


    def _calc_fisher(self, likelihood, nsample=1):
        grads = autograd.grad(likelihood, self._model_anchor.parameters())
        fs = [(g**2).sum()/nsample for g in grads]

        param_names = [name for name, _ in self._model_anchor.named_parameters()]
        fs_dict = {n: f.item() for n, f in zip(param_names, fs)}

        return fs_dict
    

    def set_fisher(self, likelihood, nsample=1):
        self._fishers = self._calc_fisher(likelihood, nsample=nsample)


    def get_fisher(self): return self._fishers

    
    def calc_ewc(self, model, weight=5e8):
        params = {n: p for n, p in model.named_parameters()}

        ewc = 0
        for name in self._fishers:
            _ewc = self._fishers[name]*(params[name] - self._params_anchor[name])**2
            ewc += _ewc.sum()

        return weight*ewc


def loss_adapt(l_adv, ewc, model, weight=5e8):
    l_ewc = ewc.calc_ewc(model, weight=weight)
    return l_adv + l_ewc

# def make_image(epoch, generator, examples=25, dim=(5,5), figsize=(10,10)):
#     noise= torch.randn(examples, latent_size).to(device)
#     with torch.no_grad():
#         generated_images = generator(noise).detach().cpu()
#     plt.figure(figsize=figsize)
#     plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)


#     for i in range(generated_images.shape[0]):
#         # Ghép các kênh màu lại thành một ảnh RGB
#         plt.subplot(5, 5, i + 1)
#         image = generated_images[i].permute(1,2,0)
#         plt.imshow(image,interpolation='nearest',cmap='gray_r')
#         plt.axis('off')

#     plt.tight_layout()
#     path = 'D:\Adapt_rs'
#     plt.savefig(os.path.join(path,'gan_generated_image %d.png' %epoch))
#     plt.close('all')
#     # plt.show()
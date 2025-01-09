import torch
import torch.nn as nn
from tqdm import tqdm

from utils import EWC


class DCGAN():
    def __init__(self, d, g, optim_d, optim_g, latent_size):
        self.discriminator = d
        self.generator = g
        self.optim_d = optim_d
        self.optim_g = optim_g
        self.latent_size = latent_size

        self.criterion = nn.BCELoss()


    def to(self, device):
        self.device = device
        self.discriminator.to(self.device)
        self.generator.to(self.device)


    def set_train_params(self, batch_size, num_epochs, real_lb, fake_lb):
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._real_lb = real_lb
        self._fake_lb = fake_lb
        
        self.g_history = []
        self.d_history = []

    
    def train(self, dataloader, verbose=False, checkpoint_every=None):
        for epoch in range(self._num_epochs):

            avg_loss_d = 0 
            avg_loss_g = 0

            for data in tqdm(dataloader):
                real_data = data[0].to(self.device)
                label = torch.full((self._batch_size, ), self.real_lb, dtype=torch.float).to(self.device)
                
                # Train D
                self.discriminator.zero_grad()
                self._train_step_D(real_data, label)
                loss_D, D_x, D_G_z1 = self._train_step_D(real_data)
                avg_loss_d += loss_D

                # Train G 
                self.generator.zero_grad()
                loss_G, D_G_z2 = self._train_step_G(label)
                avg_loss_g += loss_G

                avg_loss_d /= len(dataloader)
                avg_loss_g /= len(dataloader)

                self.d_history.append(avg_loss_d.detach().cpu().numpy())
                self.g_history.append(avg_loss_g.detach().cpu().numpy())

                if verbose:
                    print(f'## EPOCH {epoch}: Loss_D: {avg_loss_d.item():.4f} Loss_G: {avg_loss_g.item():.4f} ' 
                          f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')
                
                # if checkpoint and (epoch % 5 == 0):
                #     make_image(epoch, generator)
                #     torch.save(generator.state_dict(), 'D:\G_md\epoch%d.pth' %epoch)
                #     torch.save(discriminator.state_dict(), 'D:\D_md\epoch%d.pth' %epoch)
                
                
    def _train_step_D(self, real_data, label):
        # Train D with real data 
        # label = torch.full((self._batch_size, ), self._real_lb, dtype=torch.float).to(self.device)
        output = self.discriminator(real_data).view(-1)
        loss_D_real = self.criterion(output, label)
        loss_D_real.backward()

        # Train D with fake label 
        noise = torch.randn(self._batch_size, self.latent_size).to(self.device)
        fake_data = self.generator(noise)
        label.fill_(self._fake_lb)
        output = self.discriminator(fake_data.detach()).view(-1)
        loss_D_fake = self.criterion(output, label)

        loss_D_fake.backward()
        self.optim_d.step()

        D_x = output.mean().item()
        D_G_z1 = output.mean().item()
        loss_D = loss_D_real + loss_D_fake

        return loss_D, D_x, D_G_z1
    

    def _train_step_G(self, label):
        noise = torch.randn(self._batch_size, self.latent_size).to(self.device)
        fake_data = self.generator(noise)
        output = self.discriminator(fake_data).view(-1)
        label.fill_(self._real_lb)

        loss_G = self.criterion(output, label)

        loss_G.backward()
        self.optim_g.step()

        D_G_z2 = output.mean().item()

        return loss_G, D_G_z2

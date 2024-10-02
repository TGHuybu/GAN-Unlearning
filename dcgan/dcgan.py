import torch
import torch.nn as nn
from tqdm import tqdm
from IPython.display import clear_output


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

    
    def train(self, dataloader, batch_size, num_epochs):
        g_history = []
        d_history = []

        real_lb = 0.99
        fake_lb = 0

        for epoch in range(num_epochs):

            avg_loss_d = 0 
            avg_loss_g = 0

            for i, data in enumerate(tqdm(dataloader)):
                
                # train D with real data 
                self.discriminator.zero_grad()
                
                real_data = data[0].to(self.device)
                label = torch.full((batch_size, ), real_lb,dtype=torch.float).to(self.device)
                output = self.discriminator(real_data).view(-1)
                loss_D_real = self.criterion(output,label)
                loss_D_real.backward()
                D_x = output.mean().item()

                # train D with fake label 
                noise = torch.randn(batch_size, self.latent_size).to(self.device)
                fake_data = self.generator(noise)
                label.fill_(fake_lb)
                output = self.discriminator(fake_data.detach()).view(-1)
                loss_D_fake = self.criterion(output, label)
                loss_D_fake.backward()
                D_G_z1 = output.mean().item()

                loss_D = loss_D_real + loss_D_fake
                avg_loss_d += loss_D

                self.optim_d.step()

                # train G 
                self.generator.zero_grad()
                noise = torch.randn(batch_size, self.latent_size).to(self.device)
                fake_data = self.generator(noise)
                output = self.discriminator(fake_data).view(-1)
                label.fill_(real_lb)
                loss_G = self.criterion(output, label)
                avg_loss_g += loss_G
                loss_G.backward()
                D_G_z2 = output.mean().item()

                self.optim_g.step()

                avg_loss_d /= len(dataloader)
                avg_loss_g /= len(dataloader)

                d_history.append(avg_loss_d.detach().cpu().numpy())
                g_history.append(avg_loss_g.detach().cpu().numpy())

                if epoch != num_epochs:
                    clear_output()

                print(  f'Loss_D: {avg_loss_d.item():.4f} Loss_G: {avg_loss_g.item():.4f} '
                        f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')
                
                # if (epoch % 5 == 0):
                #     make_image(epoch, generator)
                #     torch.save(generator.state_dict(), 'D:\G_md\epoch%d.pth' %epoch)
                #     torch.save(discriminator.state_dict(), 'D:\D_md\epoch%d.pth' %epoch)
                
                
    def discriminator_train_step():
        pass
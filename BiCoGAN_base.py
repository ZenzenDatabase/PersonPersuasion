# MODEL = 'BiCoGAN'
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.init as init
import sys
sys.path.append("..")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NLayerLeakyMLP(nn.Module):
    def __init__(self, in_features, out_features, num_layers, hidden_dim):
        super(NLayerLeakyMLP, self).__init__()
        model = [nn.Linear(in_features, hidden_dim), nn.LeakyReLU(0.1)]
        for _ in range(num_layers-1):
            model += [nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1)]
        model += [nn.Linear(hidden_dim, out_features)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x.to(device))

def Z_ScoreNormalization(data):
    mean = torch.mean(data, dim=0)
    var = torch.std(data, dim=0)
    normalized_data = (data - mean)/var
    return normalized_data

class Encoder(nn.Module):
    '''
    Encoder class in a Biconditional GAN. 
    Accepts a state tensor as input.
    Generate a tensor as output that is indistinguishable from the real dataset.
    '''

    def __init__(self, cfg):
        super().__init__()
        self.state_size = cfg['general']['state_size']
        self.noise_size = cfg['general']['noise_size']
        self.hidden_size = cfg['general']['hidden_size']
        self.layer = nn.Sequential(nn.Linear(self.state_size, self.hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_size, self.hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_size, self.noise_size))
    def forward(self, x):
        out = self.layer(x)
        return out


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

class Generator(nn.Module):
    '''
    Generator class in a Biconditional GAN. 
    The main difference is: select the input feature using SEM
    Accepts a noise tensor (latent_dim = state_dim) and a condition tensor (state and action) as input.
    Generate a next_state tensor as output that is indistinguishable from the real dataset.
    '''

    def __init__(self, GC_est, cfg):
        super().__init__()
        self.type = cfg['general']['type']
        out_features = 1 # generate the next state one by one
        self.state_dim = cfg['general']['state_size']
        hidden_dim = cfg['general']['hidden_size']
        num_layers = cfg['gan_paras']['layer_num']

        self.layer_list = []
        self.index_list = []
        for i in range(self.state_dim):
            sem_i = GC_est[i, :]
            index_i = torch.squeeze(torch.tensor(np.where(sem_i==1))).to(device)
            self.index_list.append(index_i)
            in_features = np.sum(sem_i) + 1 # add noise dimension
            layer = NLayerLeakyMLP(in_features, out_features, num_layers, hidden_dim)
            self.layer_list.append(layer)
        self.layer_list = nn.ModuleList(self.layer_list)

    def forward(self, y, z, index, normal=False):
        # y: [10332, 1536]
        # z: [10332, 768]
        ns = []
        # generate next_state one by one
        for i in range(self.state_dim):
            z_i = torch.unsqueeze(z[:, i], dim=-1)
            if self.type == "multi":
                select_idx = self.index_list[i]
                if index == "target":
                    index = 3 * torch.ones(z_i.shape).to(device)
                if (select_idx==self.state_dim+1).cpu().numpy().any():
                    select_idx = select_idx[:-1]
                    paraent_i = y.index_select(1, select_idx)
                    out_i = torch.cat([paraent_i, z_i, index.float()], dim=-1)
                else:
                    parent_i = y.index_select(1, select_idx)
                    out_i = torch.cat([parent_i, z_i], dim=-1)
            elif self.type == "single":
                if len(self.index_list[i]) == 0:
                    # no parents
                    out_i = z_i
                else:
                    parent_i = y.index_select(1, self.index_list[i])
                    out_i = torch.cat([parent_i, z_i], dim=-1)
            output = self.layer_list[i](out_i)
            ns.append(output)
        embedding = torch.cat(ns, dim=-1)
        return embedding


class Discriminator(nn.Module):
    '''
    Discriminator class in a Conditional GAN. 
    Accepts a next_state tensor and a conditioanl tensor as input
    Outputs a tensor of size 1 with the predicted class probabilities (generated or real data)
    '''

    def __init__(self, cfg):
        super().__init__()
        self.state_size = cfg['general']['state_size']
        self.noise_size = cfg['general']['noise_size']
        self.hidden_size = cfg['general']['hidden_size']
        self.con_size = cfg['general']['condition_size']
        self.layer = nn.Sequential(nn.Linear(self.state_size+self.con_size+self.noise_size, self.hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_size, self.hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_size, 1),
                                   nn.Sigmoid())

    def forward(self, x, y, z):
        out = torch.cat([x, z, y], dim=-1)
#         print('state, con, noise:',self.state_size+self.con_size+self.noise_size)
#         print('observation, noise, y(state, action), output:', x.shape, z.shape, y.shape, out.shape)
        # out = Z_ScoreNormalization(out)
        out = self.layer(out)
        return out


class LitCounter(pl.LightningModule):

    def __init__(self, GC_est, cfg):
        super().__init__()
        self.save_hyperparameters("cfg")
        self.lr_gan = cfg['gan_paras']['lr_gan']
        self.noise_size = cfg['general']['noise_size']
        self.encoder = Encoder(cfg)
        self.generator = Generator(GC_est, cfg)
        self.discriminator = Discriminator(cfg)
        self.dis_cogan = cfg['gan_paras']['dis_cogan']
        self.loss_fn = torch.nn.MSELoss()

    def dataNormal(self, x):
        d_min = x.min()
        d_max = x.max()
        dst = d_max - d_min
        d_norm = (x - d_min).true_divide(dst)
        return d_norm
    
    def forward(self, st, at):
        """
        Generates a next state using the generator
        given input noise z and conditions y
        # """
        print(type(st), type(at))
        z = torch.randn(st.shape[0], self.noise_size, device=device).float()
        y = torch.cat([st, at], dim=-1).float()
        index = "target"
        return self.generator(y, z, index)

    @torch.no_grad()
    def counterfactual(self, st, counter_at, st_):
        """
        Generates the next state using the generator
        given inferenced noise z and counterfactual conditions y
        """
        z = self.encoder(torch.from_numpy(np.array(st_)).float().to(device))
        # counter_at = ~at+2
        counter_at = (torch.from_numpy((np.array(counter_at)))).to(device)
        y = torch.cat([st.to(device), counter_at], dim=-1).float()
        index = 'target'
        next_state = self.generator(y, z, index, normal=False)
        return next_state
        
    def generator_step(self, x, y, index):
        """
        Training step for generator
        1. Sample random noise and conditions
        2. Pass noise and conditions to generator to generate next_state
        3. Classify generated next_state using the discriminator
        4. Backprop loss
        """
        # Sample random noise and conditions
        y = y.to(device).float()
        z = torch.randn(x.shape[0], self.noise_size, device=device).float()
        generated_states = self.generator(y, z, index)
#         print('generated_states, y, z:', generated_states.shape, y.shape, z.shape)
        dg_output = torch.squeeze(self.discriminator(generated_states, y, z))

        return dg_output, generated_states

    def discriminator_step(self, x, y):
        """
        Training step for discriminator
        1. Get actual next_state and conditions
        2. Predict probabilities of actual next_state and get BCE loss
        3. Get fake next_state from generator
        4. Predict probabilities of fake next_state and get BCE loss
        5. Combine loss from both and backprop
        """
#         print('x====', x.shape)
        encoded_noise = self.encoder(x.float())
        de_output = torch.squeeze(self.discriminator(x.float(), y.float(), encoded_noise))

        return de_output

    def training_step(self, batch, batch_idx, optimizer_idx):
        eps = 1e-10
        x, y = batch
        index = batch_idx
        # where x = next_state; y = (state, action)
        DG, generated_states = self.generator_step(x, y, index)
        DE = self.discriminator_step(x, y)
        # error_g = torch.mean(torch.mean(torch.abs(x - generated_states), dim=-1), dim=-1)
        error_g = self.loss_fn(generated_states.float(), x.float())

        error = torch.abs(x - generated_states)
        d_norm = (1 - torch.mean(torch.mean(self.dataNormal(error), dim=0), dim=0)).float()
        d_norm = 1.
        loss_D = torch.log(DE + eps) + torch.log(1 - d_norm*DG + eps)
        loss_EG = torch.log(d_norm*DG + eps) + torch.log(1 - DE + eps)

        if optimizer_idx == 0:
            if self.dis_cogan:
                loss = -torch.mean(loss_EG) + error_g # add some discriminator
            else:
                loss = -torch.mean(loss_EG) # totally bicogan
            loss.requires_grad_()

        if optimizer_idx == 1:
            loss = -torch.mean(loss_D)
            loss.requires_grad_()

        if optimizer_idx == 0:
            self.log('loss_gen', -torch.mean(loss_EG))
            self.log('loss_error', error_g)
        else:
            self.log('loss_dis', loss)
            
        return loss

    def configure_optimizers(self):
        eg_optimizer = torch.optim.Adam(list(self.encoder.parameters())+list(self.generator.parameters()), lr=self.lr_gan, betas=(0.0, 0.999))
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_gan, betas=(0.0, 0.999))
        return [eg_optimizer, d_optimizer]
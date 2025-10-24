import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import pandas as pd


class LoRADecoder(nn.Module):
    def __init__(self, n_particles_dataset, conf_regressor_params,  n_layers=3, hidden_dim=1024, seq_dim = 1314, out_dim=384, out_channels=16, no_heads=12, rank=4, embedding_af = True, sonly = False):
        super(LoRADecoder, self).__init__()
        self.sonly = sonly
        if sonly:
             self.W = nn.Parameter(torch.randn(out_dim, out_channels * no_heads, 3) * conf_regressor_params['std_z_init'],requires_grad=True)
        else:
            self.seq_dim = seq_dim
            self.rank = rank
            self.z_dim = conf_regressor_params['z_dim']
            self.variational_conf = conf_regressor_params['variational']
            self.std_z_init = conf_regressor_params['std_z_init']
            self.out_dim = out_dim
            self.out_channels = out_channels
            self.no_heads = no_heads
            self.conf_table = ConfTable(
                n_particles_dataset, self.z_dim, conf_regressor_params['variational'],
                conf_regressor_params['std_z_init'], conf_regressor_params['init'])

            if embedding_af:
                self.decoder = ResidualLinearMLP(self.z_dim, n_layers, hidden_dim, self.out_dim * no_heads * self.out_channels * 3)
            else:
                self.decoder = ResidualLinearMLP(self.z_dim, n_layers, hidden_dim, self.out_dim ** 3)

            if conf_regressor_params['init'] is not 'zeros':
                self._initialize_weights()
                
    def forward_s(self):
        return self.W
    
    def forward_sinit(self,o):
        s_m = self.decoder(o)
        return s_m

    def forward(self, in_dict):
        latent_variables_dict = self.encode(in_dict)
        output = self.decode(latent_variables_dict)
        
        return output
    
    def decode_z(self, z):
        return self.decoder(z)

    def encode(self, in_dict):
        latent_variables_dict = {}
        conf_dict = self.conf_table(in_dict)
        for key in conf_dict:
            latent_variables_dict[key] = conf_dict[key]
        return latent_variables_dict

    def decode(self, latent_variables_dict):
        return self.decoder(latent_variables_dict['z'])

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
                   
def Sampling(mu, logvar):

    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class ConfTable(nn.Module):
    def __init__(self, n_imgs, z_dim, variational, std_z_init, init):
        """
        n_imgs: int
        z_dim: int
        variational: bool
        """
        super(ConfTable, self).__init__()

        if init == 'zero' or init == 'zeros':
            self.conf_init = torch.tensor(
            std_z_init * np.zeros((n_imgs, z_dim))
        ).float()
        elif init == 'normal':
            self.conf_init = torch.tensor(
            std_z_init * np.random.randn(n_imgs, z_dim)
        ).float()
        else:
            data_3dva = pd.read_csv(str(init))
            z = data_3dva.to_numpy()
            z = (z-np.mean(z,axis=0))/(np.std(z,axis=0)+1e-10)
            z = z * std_z_init
            self.conf_init = torch.tensor(
                z
            ).float()
        self.variational = variational

        self.table_conf = nn.Parameter(self.conf_init, requires_grad=True)
        if variational:
            logvar_init = torch.tensor(np.ones((n_imgs, z_dim))).float()
            self.table_logvar = nn.Parameter(logvar_init, requires_grad=True)

    def initialize(self, conf):
        """
        conf: [n_imgs, z_dim] (numpy)
        """
        state_dict = self.state_dict()
        state_dict['table_conf'] = torch.tensor(conf).float()
        self.load_state_dict(state_dict)

    def forward(self, in_dict):
        """
        in_dict: dict
            index: [batch_size]
            y: [batch_size(, n_tilts), D, D]
            y_real: [batch_size(, n_tilts), D - 1, D - 1]
            R: [batch_size(, n_tilts), 3, 3]
            t: [batch_size(, n_tilts), 2]
            tilt_index: [batch_size( * n_tilts)]

        output: dict
            z: [batch_size, z_dim]
            z_logvar: [batch_size, z_dim] if variational and not pose_only
        """
        conf = self.table_conf[in_dict['index']]
        conf_dict = {'z': conf}
        if self.variational:
            logvar = self.table_logvar[in_dict['index']]
            conf_dict['z_logvar'] = logvar
        return conf_dict

    def reset(self):
        state_dict = self.state_dict()
        state_dict['table_conf'] = self.conf_init / 10.0
        self.load_state_dict(state_dict)


class ResidualLinearMLP(nn.Module):
    def __init__(self, in_dim, n_layers, hidden_dim, out_dim, nl=nn.ReLU):
        super(ResidualLinearMLP, self).__init__()
        layers = [ResidualLinear(in_dim, hidden_dim) if in_dim == hidden_dim else nn.Linear(in_dim, hidden_dim),
                  nl()]
        for n in range(n_layers):
            layers.append(ResidualLinear(hidden_dim, hidden_dim))
            layers.append(nl())
        layers.append(
            ResidualLinear(hidden_dim, out_dim) if out_dim == hidden_dim else MyLinear(hidden_dim, out_dim))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: [..., in_dim]

        output: [..., out_dim]
        """
        flat = x.view(-1, x.shape[-1])
        ret_flat = self.main(flat)
        ret = ret_flat.view(*x.shape[:-1], ret_flat.shape[-1])
        return ret


class ResidualLinear(nn.Module):
    def __init__(self, n_in, n_out):
        super(ResidualLinear, self).__init__()
        self.linear = nn.Linear(n_in, n_out)

    def forward(self, x):
        z = self.linear(x) + x
        return z


class MyLinear(nn.Linear):
    def forward(self, x):
        if x.dtype == torch.half:
            return half_linear(x, self.weight, self.bias)
        else:
            return single_linear(x, self.weight, self.bias)


def half_linear(x, weight, bias):
    return F.linear(x, weight.half(), bias.half())


def single_linear(x, weight, bias):
    return F.linear(x, weight, bias)



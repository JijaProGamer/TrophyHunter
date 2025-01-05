import torch
import math
import torch.nn.functional as F
import torch.nn as nn
from pytorch_msssim import SSIM
from torchsummary import summary

from .layers import SelfAttentionModule, ResNetLayer, DenseResNetLayer, DownscaleFilters, elastic_net_regularization


mae_criterion = nn.L1Loss()
mse_criterion = nn.MSELoss()

class Encoder(nn.Module):
    def __init__(self, args, input_resolution, latent_dims):
        super(Encoder, self).__init__()

        x_downscale = int(input_resolution[0] / 8)
        y_downscale = int(input_resolution[1] / 8)
        out_dims = 64 * x_downscale * y_downscale
        
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn = nn.BatchNorm2d(64)
        
        self.res1 = ResNetLayer(in_channels=64, out_channels=128, scale='down', kernel_size=3, stride=2)
        self.res2 = ResNetLayer(in_channels=128, out_channels=256, scale='down', kernel_size=3, stride=2) 
        self.res3 = ResNetLayer(in_channels=256, out_channels=512, scale='down', kernel_size=3, stride=2)

        self.simplify = DownscaleFilters(in_channels=512, out_channels=64)

        self.attention = SelfAttentionModule(attention_features=64)
        
        self.flatten = nn.Flatten()

        #self.transformer = nn.Linear(out_dims, latent_dims, bias=False)
        #self.transformer_bn = nn.BatchNorm1d(latent_dims)

        self.fc_mean = nn.Linear(out_dims, latent_dims)
        self.fc_var = nn.Linear(out_dims, latent_dims)

    def forward(self, x):
        x = F.silu(self.bn(self.conv(x)))

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        
        x = self.simplify(x)

        x = self.attention(x)

        x = self.flatten(x)

        mu = self.fc_mean(x)
        logvar = self.fc_var(x)

        return mu, logvar
    def forward_mu(self, x):
        x = F.silu(self.bn(self.conv(x)))

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        
        x = self.simplify(x)

        x = self.attention(x)

        x = self.flatten(x)

        mu = self.fc_mean(x)

        return mu
    def freeze_layers(self):
        for name, param in self.named_parameters():
            if not (
                name.startswith("res4") or
                name.startswith("fc_mean") or
                name.startswith("fc_var")
            ):
                param.requires_grad = False

class Decoder(nn.Module):
    def __init__(self, args, input_resolution, latent_dims):
        super(Decoder, self).__init__()

        x_downscale = int(input_resolution[0] / 8)
        y_downscale = int(input_resolution[1] / 8)

        self.input_resolution = input_resolution
        self.x_downscale = x_downscale
        self.y_downscale = y_downscale

        #self.fc = nn.Linear(latent_dims, latent_dims, bias=False)
        #self.bn = nn.BatchNorm1d(latent_dims)

        self.transformer = nn.Linear(latent_dims, x_downscale * y_downscale * 64, bias=False)
        self.transformer_bn = nn.BatchNorm1d(x_downscale * y_downscale * 64)

        self.simplify = DownscaleFilters(in_channels=64, out_channels=512)

        self.res1 = ResNetLayer(in_channels=512, out_channels=256, scale='up', kernel_size=4, stride=2)
        self.res2 = ResNetLayer(in_channels=256, out_channels=128, scale='up', kernel_size=4, stride=2) 
        self.res3 = ResNetLayer(in_channels=128, out_channels=64, scale='up', kernel_size=4, stride=2)

        self.output = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        #x = F.silu(self.bn(self.fc(x)))

        x = F.silu(self.transformer_bn(self.transformer(x)))

        x = x.view(-1, 64, self.x_downscale, self.y_downscale)

        x = self.simplify(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        x = self.output(x)

        return x
    def freeze_layers(self):
        for name, param in self.named_parameters():
            if not (
                name.startswith("fc1") or
                name.startswith("res4")
            ):
                param.requires_grad = False


class VAE(nn.Module):
    def __init__(self, args, dataset_size):
        super().__init__()

        self.args = args
        self.dataset_size = dataset_size

        self.ssim_module = SSIM(data_range=1, size_average=True, channel=3, win_size=11, win_sigma=1)
        self.ssim_module2 = SSIM(data_range=1, size_average=True, channel=3, win_size=7, win_sigma=1)
        #self.ssim_module = SSIM(data_range=1, size_average=True, channel=3, win_size=5, win_sigma=1)
        #self.ssim_module2 = SSIM(data_range=1, size_average=True, channel=3, win_size=3, win_sigma=0.5)

        #self.discriminator = Discriminator(
        #    latent_dims = args["latent_size"],
        #)

        self.encoder = Encoder(
            input_resolution=args["resolution"],
            latent_dims = args["latent_size"],
            args = args
        )

        self.decoder = Decoder(
            input_resolution=args["resolution"],
            latent_dims = args["latent_size"],
            args = args
        )

        #summary(self, (3, args["resolution"][1], args["resolution"][0]))

        #self.encoder.freeze_layers()
        #self.decoder.freeze_layers()

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        std = torch.clamp(std, min=0, max=100)

        eps = torch.randn_like(std)

        return mean + eps * std
    
    def reconstruction_loss(self, x, y): # 1e-4 to prevent NaN
        x_slim = (x + 1) / 2
        y_slim = (y + 1) / 2

        mse = torch.sqrt(mse_criterion(x_slim, y_slim) + 1e-4) * 0.3 + 1e-4 
        ssim1 = torch.pow((1 - self.ssim_module(x_slim, y_slim)), 1/3) * 0.4 + 1e-4
        ssim2 = torch.pow((1 - self.ssim_module2(x_slim, y_slim)), 1/3) * 0.3 + 1e-4
    
        #return ssim * (self.args["resolution"][0] * self.args["resolution"][1] * 3)
        return F.relu(mse + ssim1 + ssim2) * (self.args["resolution"][0] * self.args["resolution"][1] * 3)

    def disentanglement_loss(self, step, z, mu, logvar):
        log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx(z,
                                                                             mu, logvar,
                                                                             self.dataset_size,
                                                                             is_mss=self.args["is_mss"])
        mi_loss = (log_q_zCx - log_qz).mean()
        tc_loss = (log_qz - log_prod_qzi).mean()
        dw_kl_loss = (log_prod_qzi - log_pz).mean()

        kl_anneal = 1#linear_annealing(0.01, 1, step, self.args["steps_anneal"])

        return (self.args["alpha"] * mi_loss + self.args["beta"] * tc_loss + self.args["gamma"] * dw_kl_loss) * kl_anneal
        #return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    def loss(self, step, x, y, z, mu, logvar):
        #regularization = elastic_net_regularization(self, l1_lambda=self.args['l1_lambda'], l2_lambda=self.args['l2_lambda'], accepted_names=[".res"])
        reconstruction = self.reconstruction_loss(x, y)

        if self.args["disentangle"]:
            disentanglement = self.disentanglement_loss(step, z, mu, logvar)

            return reconstruction + disentanglement * self.args["disentangling_importance"]
        else:
            return reconstruction

    def forward(self, x):
        mu, logvar = self.encoder(x)

        if self.args["disentangle"]:
            z = self.reparameterize(mu, logvar)
            decoded = self.decoder(z)
            return mu, logvar, z, decoded
        else:
            decoded = self.decoder(mu)
            return mu, torch.zeros_like(mu), mu, decoded
    def muforward(self, x):
        mu, _ = self.encoder(x)
        decoded = self.decoder(mu)

        return mu, decoded    
    def zforward(self, x):
        mu, logvar = self.encoder(x)

        if self.args["disentangle"]:
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z
        else:
            return mu, torch.zeros_like(mu), mu

# Losses from https://github.com/YannDubs/disentangling-vae/blob/f0452191bab6d94eba0b4e6a065f74dcfd54ac52/disvae/models/losses.py#L523



def _get_log_pz_qz_prodzi_qzCx(latent_sample, mu, logvar, n_data, is_mss):
    batch_size, hidden_dim = latent_sample.shape

    log_q_zCx = log_density_gaussian(latent_sample, mu, logvar).sum(dim=1)

    zeros = torch.zeros_like(latent_sample)
    log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(1)

    mat_log_qz = matrix_log_density_gaussian(latent_sample, mu, logvar)

    if is_mss:
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(latent_sample.device)
        mat_log_qz = mat_log_qz + log_iw_mat.view(batch_size, batch_size, 1)

    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
    log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(1)

    return log_pz, log_qz, log_prod_qzi, log_q_zCx

def matrix_log_density_gaussian(x, mu, logvar):
    batch_size, dim = x.shape
    x = x.view(batch_size, 1, dim)
    mu = mu.view(1, batch_size, dim)
    logvar = logvar.view(1, batch_size, dim)
    return log_density_gaussian(x, mu, logvar)


def log_density_gaussian(x, mu, logvar):
    normalization = - 0.5 * (math.log(2 * math.pi) + logvar)
    inv_var = torch.exp(-logvar)
    log_density = normalization - 0.5 * ((x - mu)**2 * inv_var)
    return log_density


def log_importance_weight_matrix(batch_size, dataset_size):
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[::M + 1] = 1 / N
    W.view(-1)[1::M + 1] = strat_weight
    W[M - 1, 0] = strat_weight
    return W.log()

def linear_annealing(init, fin, step, annealing_steps):
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed
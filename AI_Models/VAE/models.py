import torch
import math
import torch.nn.functional as F
import torch.nn as nn

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

class Decoder(nn.Module):
    def __init__(self, args, input_resolution, latent_dims):
        super(Decoder, self).__init__()

        x_downscale = int(input_resolution[0] / 8)
        y_downscale = int(input_resolution[1] / 8)

        self.input_resolution = input_resolution
        self.x_downscale = x_downscale
        self.y_downscale = y_downscale

        self.transformer = nn.Linear(latent_dims, x_downscale * y_downscale * 64, bias=False)
        self.transformer_bn = nn.BatchNorm1d(x_downscale * y_downscale * 64)

        self.simplify = DownscaleFilters(in_channels=64, out_channels=512)

        self.res1 = ResNetLayer(in_channels=512, out_channels=256, scale='up', kernel_size=4, stride=2)
        self.res2 = ResNetLayer(in_channels=256, out_channels=128, scale='up', kernel_size=4, stride=2) 
        self.res3 = ResNetLayer(in_channels=128, out_channels=64, scale='up', kernel_size=4, stride=2)

        self.output = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.silu(self.transformer_bn(self.transformer(x)))

        x = x.view(-1, 64, self.x_downscale, self.y_downscale)

        x = self.simplify(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        x = self.output(x)

        return x


class VAE(nn.Module):
    def __init__(self, args, dataset_size):
        super().__init__()

        self.args = args
        self.dataset_size = dataset_size

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

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        std = torch.clamp(std, min=0, max=100)

        eps = torch.randn_like(std)

        return mean + eps * std
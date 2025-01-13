from .models import VAE
import torch
import os 
import yaml
import cv2

class VAEWrapper():
    def __init__(self, device):
        dir_path = os.path.dirname(os.path.realpath(__file__))

        with open(os.path.join(dir_path, 'hyperparameters.yaml')) as f:
            args = yaml.safe_load(f)

        args["device"] = device


        self.inverse_resolution = [args["resolution"][1], args["resolution"][0]]
        self.args = args
        self.device = device
        self.model = VAE(args, 0).to(device)

        checkpoint = torch.load(os.path.join(dir_path, args["path"]))#, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.encoder.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
    def predict(self, frame):
        vae_img = cv2.resize(frame, self.inverse_resolution, interpolation=cv2.INTER_LANCZOS4)
        vae_img = torch.tensor(vae_img, dtype=torch.float32).permute(2, 0, 1)
        vae_img = vae_img / 127.5 - 1
        vae_img = vae_img.unsqueeze(0).to(self.device)

        mu = self.model.encoder.forward_mu(vae_img)
        return mu
import os
import sys
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from argparse import Namespace
from e4e.models.psp import pSp
from util import *

class Helper:
    @ torch.no_grad()
    def __init__(self):
        self.loaded = False
        
    @ torch.no_grad()
    def projection(self, img, name, device='cuda'):
        if self.loaded == False:
            self.loaded = True
            self.device = device
            self.model_path = 'models/e4e_ffhq_encode.pt'
            self.ckpt = torch.load(self.model_path, map_location='cpu')
            self.opts = self.ckpt['opts']
            self.opts['checkpoint_path'] = self.model_path
            self.opts= Namespace(**self.opts)
            self.net = pSp(self.opts, device).eval().to(device)
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        
        img = self.transform(img).unsqueeze(0).to(self.device)
        images, w_plus = self.net(img, randomize_noise=False, return_latents=True)
        result_file = {}
        result_file['latent'] = w_plus[0]
        torch.save(result_file, name)
        return w_plus[0]



@ torch.no_grad()
def projection(img, name, device='cuda'):
    model_path = 'models/e4e_ffhq_encode.pt'
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts= Namespace(**opts)
    net = pSp(opts, device).eval().to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    img = transform(img).unsqueeze(0).to(device)
    images, w_plus = net(img, randomize_noise=False, return_latents=True)
    result_file = {}
    result_file['latent'] = w_plus[0]
    torch.save(result_file, name)
    return w_plus[0]

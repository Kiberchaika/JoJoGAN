import torch
torch.backends.cudnn.benchmark = True
from torchvision import transforms, utils
from util import *
from PIL import Image
import math
import random
import os
import glob

import numpy as np
from torch import nn, autograd, optim
from torch.nn import functional as F
from tqdm import tqdm
import lpips
import wandb
from model import *
from e4e_projection import Helper

from copy import deepcopy

from vgg_helper import caluclate_styleloss, caluclate_contentloss

os.makedirs('inversion_codes', exist_ok=True)
os.makedirs('style_images', exist_ok=True)
os.makedirs('style_images_aligned', exist_ok=True)
os.makedirs('models', exist_ok=True)

latent_dim = 512
device = 'cuda' #@param ['cuda', 'cpu']

# Load original generator
original_generator = Generator(1024, latent_dim, 8, 2).to(device)
ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
original_generator.load_state_dict(ckpt["g_ema"], strict=False)
mean_latent = original_generator.mean_latent(10000)

# to be finetuned generator
generator = deepcopy(original_generator)

transform = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

#@title Choose input face
#@markdown Add your own image to the test_input directory and put the name here
filename = 'myimg.png' #@param {type:"string"}
filepath = f'test_input/{filename}'

# uploaded = files.upload()
# filepath = list(uploaded.keys())[0]
name = strip_path_extension(filepath)+'.pt'

# aligns and crops face
aligned_face = align_face(filepath)

e4e_helper = Helper()

# my_w = restyle_projection(aligned_face, name, device, n_iters=1).unsqueeze(0)
if not os.path.exists(name):
    my_w = e4e_helper.projection(aligned_face, name).unsqueeze(0)
else:
    my_w = torch.load(name)['latent'].unsqueeze(0)

#display_image(aligned_face, title='Aligned face')

#path_to_dataset = 'dataset/charcoal_tiny_aligned/1' 
#path_to_dataset = 'dataset/metfaces_small'
#path_to_dataset = 'dataset/mexican'
path_to_dataset = 'dataset/bust'

names = []
for path in sorted(glob.glob(os.path.join(path_to_dataset, '*.*'))):
    names.append(os.path.basename(path))

targets = []
latents = []

path_to_inv = os.path.join('inversion_codes',os.path.basename(path_to_dataset))

for name in names:
    style_path = os.path.join(path_to_dataset, name)
    assert os.path.exists(style_path), f"{style_path} does not exist!"

    name = strip_path_extension(name)

    # crop and align the face
    style_aligned_path = os.path.join('style_images_aligned', f'{name}.png')
    if not os.path.exists(style_aligned_path):
        try:
            style_aligned = align_face(style_path)
        except:
            print("no face!!! " , name)
            continue
        style_aligned.save(style_aligned_path)
    else:
        style_aligned = Image.open(style_aligned_path).convert('RGB')

    # GAN invert
    style_code_path = os.path.join(path_to_inv, f'{name}.pt')
    print("add ", style_code_path)
    
    if not os.path.exists(path_to_inv):
        os.mkdir(path_to_inv)
    
    if not os.path.exists(style_code_path):
        latent = e4e_helper.projection(style_aligned, style_code_path)

        generator.eval()
        my_sample = generator(latent.unsqueeze(0), input_is_latent=True)
        generator.train()
        my_sample = transforms.ToPILImage()(utils.make_grid(my_sample, normalize=True, range=(-1, 1)))
        my_sample.save(os.path.join(path_to_inv, f'{name}.jpg'))

        style_aligned.save(os.path.join(path_to_inv, f'{name}_orig.jpg'))
    else:
        latent = torch.load(style_code_path)['latent']

    targets.append(transform(style_aligned).to(device))
    latents.append(latent.to(device))
    
targets = torch.stack(targets, 0)
latents = torch.stack(latents, 0)

#target_im = utils.make_grid(targets, normalize=True, range=(-1, 1))
#display_image(target_im, title='Style References')

#@title Finetune StyleGAN
#@markdown alpha controls the strength of the style
alpha = 0.9 #@param {type:"slider", min:0, max:1, step:0.1}
alpha = 1-alpha

#@markdown Tries to preserve color of original image by limiting family of allowable transformations. Set to false if you want to transfer color from reference image. This also leads to heavier stylization
preserve_color = False #@param{type:"boolean"}
#@markdown Number of finetuning steps. Different style reference may require different iterations. Try 200~500 iterations.
num_iter = 450 #@param {type:"number"}
#@markdown Log training on wandb and interval for image logging
log_interval = 50 #@param {type:"number"}


lpips_fn = lpips.LPIPS(net='vgg').to(device)

# reset generator
del generator
generator = deepcopy(original_generator)

g_optim = optim.Adam(generator.parameters(), lr=2e-3, betas=(0, 0.99))

# Which layers to swap for generating a family of plausible real images -> fake image
if preserve_color:
    id_swap = [7,9,11,15,16,17]
else:
    id_swap = list(range(7, generator.n_latent))

_c = 0
repeat = 1
targets = targets.repeat(repeat,1,1,1) 
latents = latents.repeat(repeat,1,1) 

for idx in tqdm(range(num_iter)):
    if preserve_color:
        random_alpha = 0
    else:
        random_alpha = np.random.uniform(alpha, 1)
        
    mean_w = generator.get_latent(torch.randn([latents.size(0), generator.n_latent, latent_dim]).to(device))
    #mean_w = generator.get_latent(torch.randn([latents.size(0), latent_dim]).to(device)).unsqueeze(1).repeat(1, generator.n_latent, 1)
    in_latent = latents.clone()

    # mix first layers too
    for j in range(0, 7):
        _alpha = np.linspace(1.0, 0.9, 7, endpoint=True)[j]
        in_latent[:, [j]] = _alpha*latents[:, [j]] + (1-_alpha)*mean_w[:, [j]]

    for c, j in enumerate(id_swap):
        _alpha = np.linspace(0.1, 0.0, len(id_swap), endpoint=True)[c]
        in_latent[:, [j]] = _alpha*latents[:, [j]] + (1-_alpha)*mean_w[:, [j]]

    # оригинальная логика смешивания       
    #in_latent[:, id_swap] = alpha*latents[:, id_swap] + (1-alpha)*mean_w[:, id_swap]

    #на первых итерациях дополнительный коэффициент для понижения
    _i = 50 #200
    _alpha = np.linspace(1.0, 0.0, _i, endpoint=True)[min(idx,_i-1)]
    in_latent = _alpha*latents + (1-_alpha)*in_latent

    # todo: брать семплы последовательно 
    # get only N random samples
    n = 2
    if in_latent.size(0) > n:
        perm = []
        for j in range(0, n):
            perm.append(_c % latents.shape[0])
            _c = _c + 1
        #perm = torch.randperm(in_latent.size(0))

        _idx = perm[:n]
        _in_latent = in_latent[_idx]
        _targets = targets[_idx]
    else:
        _in_latent = in_latent
        _targets = targets

    img = generator(_in_latent, input_is_latent=True)
    
    loss = 0
    
    #lpips_size = 512 #256
    #loss = loss + lpips_fn(F.interpolate(img, size=(lpips_size,lpips_size), mode='area'), F.interpolate(_targets, size=(lpips_size,lpips_size), mode='area')).mean()
    
    '''
    '''

    lpips_size = 512 #256
    img1 = F.interpolate(img, size=(lpips_size,lpips_size), mode='bilinear')
    img2 = F.interpolate(_targets, size=(lpips_size,lpips_size), mode='bilinear')
    loss = loss + lpips_fn(img1, img2).mean()

    lpips_size = 256 
    for x in range(0,1024,256):
        for y in range(0,1024,256):
            img1 = F.interpolate(torchvision.transforms.functional.crop(img,x,y,lpips_size,lpips_size), size=(lpips_size,lpips_size), mode='area')
            img2 = F.interpolate(torchvision.transforms.functional.crop(_targets,x,y,lpips_size,lpips_size), size=(lpips_size,lpips_size), mode='area')
            loss = loss + 0.01 * caluclate_contentloss((img1 + 1.0) / 2.0, (img2 + 1.0) / 2.0).mean()
            loss = loss - 0.01 * caluclate_styleloss((img1 + 1.0) / 2.0, (img2 + 1.0) / 2.0).mean()

    for g in g_optim.param_groups:
        if idx < 80:
            g['lr'] = 0.003
        else:
            g['lr'] = 0.001

    g_optim.zero_grad()
    loss.backward()
    g_optim.step()

    if idx % log_interval == 0:
        generator.eval()
        my_sample = generator(my_w, input_is_latent=True)
        generator.train()
        my_sample = transforms.ToPILImage()(utils.make_grid(my_sample, normalize=True, range=(-1, 1)))
        my_sample.save(f'my_sample_{idx}.png')


if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    
torch.save(
    {
        "g_ema": generator.state_dict(),
        "g_optim": g_optim.state_dict(),
    },
    f"checkpoints/checkpoint.pt",
)


#@title Generate results
n_sample = 5 #@param {type:"number"}
seed = 2 #@param {type:"number"}

torch.manual_seed(seed)
with torch.no_grad():
    generator.eval()
    z = torch.randn(n_sample, latent_dim, device=device)

    original_sample = original_generator([z], truncation=0.7, truncation_latent=mean_latent)
    sample = generator([z], truncation=0.7, truncation_latent=mean_latent)

    original_my_sample = original_generator(my_w, input_is_latent=True)
    my_sample = generator(my_w, input_is_latent=True)

# display reference images


my_output = torch.cat([original_my_sample, my_sample], 0)
transforms.ToPILImage()(utils.make_grid(my_output, normalize=True, range=(-1, 1))).save(f'my_sample.png')

output = torch.cat([original_sample, sample], 0)
transforms.ToPILImage()(utils.make_grid(output, normalize=True, range=(-1, 1), nrow=n_sample)).save(f'random_samples.png')

'''
for i in range(0,original_sample.shape[0]):
    tensor_to_pil = transforms.ToPILImage()(((original_sample[i]+1)/2 ).squeeze_(0))
    tensor_to_pil.save(f'original_sample_{i}.png')

for i in range(0,sample.shape[0]):
    tensor_to_pil = transforms.ToPILImage()(((sample[i]+1)/2 ).squeeze_(0))
    tensor_to_pil.save(f'sample_{i}.png')

for i in range(0,original_my_sample.shape[0]):
    tensor_to_pil = transforms.ToPILImage()(((original_my_sample[i]+1)/2 ).squeeze_(0))
    tensor_to_pil.save(f'original_my_sample_{i}.png')

for i in range(0,my_sample.shape[0]):
    tensor_to_pil = transforms.ToPILImage()(((my_sample[i]+1)/2 ).squeeze_(0))
    tensor_to_pil.save(f'my_sample_{i}.png')
'''
print("finished")


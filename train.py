#!/usr/bin/env python
# Author: Sicong Zhao

import argparse
from collections import deque
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from model import *
from dataloader import *
import zipfile

# Instruction:
# Download "celeba.zip" from https://drive.google.com/file/d/1EtIVXDLFNI1szq6mAso0746tm4sFxGBR/view?usp=sharing

# Unzip data
with zipfile.ZipFile("celeba.zip","r") as zip_ref:
  zip_ref.extractall("data_faces/")

parser = argparse.ArgumentParser()

# Model Params
parser.add_argument('--input_dim', type=int, default=32, help='The height / width of the input image to network.')
parser.add_argument('--output_dim', type=int, default=32, help='The height / width of the output image of the network.')
parser.add_argument('--nz', type=int, default=64, help='Size of the latent z vector.')
parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension of the auto encoder, should equal to nz.')
parser.add_argument('--ngf', type=int, default=64, help='The number of filters in the generator.')
parser.add_argument('--ndf', type=int, default=64, help='The number of filters in the discriminator.')
parser.add_argument('--nc', type=int, default=3, help='The number of input channels.')
parser.add_argument('--n_layers', type=int, default=2, help='The number of repetitive (Conv2d + ELU) structure.')
parser.add_argument('--exp', type=bool, default=False, help='Decide the way of growth of the number of layers in the 2nd \
                                                      conv block. True if exponentially, False if Linearly.')
parser.add_argument('--t_conv', type=bool, default=False, help='Decide the way of upsampling. True if use nn.ConvTranspose2d, \
                                                      False if use nn.UpsamplingNearest2d.')
parser.add_argument('--mean', type=float, default=0, help='The desired mean of the initialized weight')
parser.add_argument('--std', type=float, default=0.002, help='The desired standard deviation of the initialized weight.')

# Training Params
parser.add_argument('--batch_size', type=int, default=16, help='Dataloader batch size.')
parser.add_argument('--n_epochs', type=int, default=1000, help='Number of epochs to train for.')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate.')
parser.add_argument('--b1', type=float, default=0.5, help='Beta1 for Adam optimizer.')
parser.add_argument('--b2', type=float, default=0.999, help='Beta2 for Adam optimizer.')
parser.add_argument('--outf', default='./output/', help='Folder to output images and model checkpoints.')
parser.add_argument('--data_path', default='./data_faces', help='Which dataset to train on.')
parser.add_argument('--lambda_k', type=float, default=0.001, help='Learning rate of k.')
parser.add_argument('--gamma', type=float, default=0.75, help='Balance bewteen D and G.')
parser.add_argument('--sample_interval', type=int, default=1000, help='Save constructed images every this many iterations.')
parser.add_argument('--show_every', type=int, default=100, help='Show log info every this many iterations. Also decide model export.')
parser.add_argument('--lr_update_step', type=int, default=3000, help='Decay lr this many iterations.')
parser.add_argument('--lr_gamma', type=float, default=0.5, help='The gamma of lr_scheduler, multiplicative factor of lr decay.')

opt = parser.parse_args()

# Data Loader
data_loader = load_data(opt.batch_size, opt.data_path, opt.input_dim)

# Initialize the Generator and Discriminator
D = Discriminator(opt.nc, opt.ndf, opt.ngf, opt.n_layers, opt.hidden_dim, opt.input_dim, opt.output_dim, opt.exp, opt.t_conv)
G = Generator(opt.nc, opt.ngf, opt.n_layers, opt.hidden_dim, opt.output_dim, opt.t_conv)

if torch.cuda.is_available():
    D.cuda()
    G.cuda()

D.weight_init(opt.mean, opt.std)
G.weight_init(opt.mean, opt.std)

# Initialize optimizer
optimizer_G = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
D_optim_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=1, gamma=opt.gamma)
G_optim_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=1, gamma=opt.gamma)

# Start training
k = 0.0
M_history = []
M_rolling = deque(maxlen=opt.show_every)
prev_measure = None
best_score = math.inf

for n in range(opt.n_epochs):
    for i, (real_imgs,_) in enumerate(data_loader):
        
        # Scale the data, [0,1] -> [-1,1]
        real_imgs = real_imgs*2 - 1

        # Noise
        z = torch.empty(real_imgs.size(0), opt.nz).uniform_(-1, 1)
        if torch.cuda.is_available():
            z = z.cuda()
            real_imgs = real_imgs.cuda()
        
        # Train Discriminator
        optimizer_D.zero_grad()
        gen_imgs_d = G(z)
        d_fake = D(gen_imgs_d.detach())
        d_real = D(real_imgs)
        d_loss_real = torch.mean(torch.abs(d_real - real_imgs))
        d_loss_fake = torch.mean(torch.abs(d_fake - gen_imgs_d))
        d_loss = d_loss_real - k * d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        gen_imgs = G(z)
        g_loss = torch.mean(torch.abs(D(gen_imgs) - gen_imgs))
        g_loss.backward()
        optimizer_G.step()

        # Update k
        diff = torch.mean(opt.gamma * d_loss_real - d_loss_fake)
        k = k + opt.lambda_k * diff.item()
        k = min(max(k,0), 1) # constrain k to [0,1]

        batches_done = n * len(data_loader) + i

        # Update convergence indicator
        M = (d_loss_real + torch.abs(diff)).data
        M_history.append(M)
        M_copy = M.cpu()
        M_rolling.append(M_copy.numpy())
        

        if i % opt.show_every == 0:

            avg_rwd = np.mean(M_rolling)
            if best_score > avg_rwd:
                best_score = avg_rwd                
                torch.save(G, "%sG.pth" % opt.outf)
                torch.save(G, "%sD.pth" % opt.outf)
                print('Model_updated in epoch: %d, batch: %d' % (n, batches_done))

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] -- M: %f, k: %f -- Best AVG M: %f"
                % (n, opt.n_epochs, i, len(data_loader), d_loss.item(), g_loss.item(), M, k, best_score)
            )

        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:16], "%s%d.png" % (opt.outf, batches_done), nrow=4, normalize=True)
        # Decay learning rate when the convergence measure stalls
        if batches_done % opt.lr_update_step == opt.lr_update_step - 1:
            cur_measure = torch.tensor(M_history).mean()
            if not prev_measure:
                prev_measure = cur_measure
            elif cur_measure > prev_measure * 0.97:
                D_optim_scheduler.step()
                G_optim_scheduler.step()
                prev_measure = cur_measure
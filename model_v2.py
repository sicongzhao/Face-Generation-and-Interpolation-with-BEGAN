#!/usr/bin/env python
# Author: Sicong Zhao

import torch.nn as nn
import math

class ResNet(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.carry = 1

    def forward(self, inputs):
        self.carry += 1/10000
        if self.carry >= 1:
            self.carrt = 1
        return (1 - self.carry) * self.module(inputs) + self.carry * inputs

def conv_block(in_dim,out_dim,n_layers):
    '''
    The building block of the encoder, the output size (W*H) will be half of the input.
    
    Parameters:
        in_dim   (int): The input dimension of the block
        out_dim  (int): The output dimension of the block
        n_layers (int): The number of repetitive (Conv2d + ELU) structure. The paper suggests the larger 
                number lead to better visual resuts
    Return: 
        The constructed nn.Sequential module.
    '''
    layers = []
    for i in range(n_layers):
        # Add residual network
        resblock = ResNet(
            nn.Sequential(
                nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                nn.ELU(True),
                nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1)
                )
            )
        layers.append(resblock)
        layers.append(nn.ELU(True))
    layers.append(nn.Conv2d(in_dim,out_dim,kernel_size=1,stride=1,padding=0))
    # layers.append(nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=1,padding=1))
    layers.append(nn.AvgPool2d(kernel_size=2,stride=2))               
    return nn.Sequential(*layers)


def deconv_block(in_dim,out_dim,n_layers,t_conv=False):
    '''
    The building block of the decoder, the output size (W*H) will be doble of the input. 
    
    Parameters:
        in_dim   (int): The input dimension of the block
        out_dim  (int): The output dimension of the block
        n_layers (int): The number of repetitive (Conv2d + ELU) structure. The paper suggests the larger 
                number lead to better visual resuts
        t_conv   (bool): If True, use nn.ConvTranspose2d to upsample. Else use nn.UpsamplingNearest2d. Default False.
    
    Return: 
        The constructed nn.Sequential module.
    '''
    layers = []
    layers.append(nn.Conv2d(in_dim,out_dim,kernel_size=1,stride=1,padding=0))
    layers.append(nn.ELU(True))  
    for i in range(n_layers):
        # 
        resblock = ResNet(
            nn.Sequential(
                nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1),
                nn.ELU(True),
                nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1)
                )
            )
        layers.append(resblock)
        layers.append(nn.ELU(True))

    if t_conv:
        layers.append(nn.ConvTranspose2d(out_dim,out_dim,4,stride=2,padding=1))
        layers.append(nn.ELU(True))
    else:
        layers.append(nn.UpsamplingNearest2d(scale_factor=2))       
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    '''
    This is a class for encoder module of Discriminator.

    Attributes:
        ndf          (int): The number of filters in the discriminator.
        layer_factor (int): The scale factor of layers.
        conv1        (nn.Sequential): 1st convolutional block, transform the number of 
                                    layers to ndf, maintain size.
        conv2        (nn.Sequential): 2nd convolutional block, linearly / exponentially 
                                    increase layer size. Also down sample width / heights
                                    by half.
        conv3        (nn.Sequential): 3rd convolutional block.
        embed        (nn.Module): Linear layer, convert data to hidden dimension.
    '''

    def __init__(self,nc,ndf,n_layers,hidden_dim,input_dim,exp=False):
        '''
        The constructor for Encoder class.

        Parameters:
            nc         (int): Number of input channels.
            ndf        (int): The number of filters in the discriminator.
            n_layers   (int): The number of repetitive (Conv2d + ELU) structure.
            hidden_dim (int): The hidden dimension of the encoder. Also the output size.
            input_dim  (int): The size of the input image (height, width).
            exp        (bool): Decide the way of growth of the number of layers in the 2nd 
                             conv block. True if exponentially, False if Linearly
        '''
        power = math.log(input_dim/8, 2)

        if not power.is_integer():
            raise ValueError('The input size should be the power of 2.')
        elif input_dim <= 8:
            raise ValueError('The input should have width / height larger than 8.')

        super().__init__()

        self.layer_factor = 2
        self.ndf = ndf

        self.conv1 = nn.Sequential(
                        nn.Conv2d(nc,ndf,kernel_size=3,stride=1,padding=1),
                        nn.ELU(True),
                        conv_block(ndf, ndf*self.layer_factor, n_layers)
                        )

        conv2_layers = []

        for i in range(int(power)-1):
            if exp:
                next_layer_factor = self.layer_factor * 2
            else:
                next_layer_factor = self.layer_factor + 1
            conv_block_instance = conv_block(ndf*self.layer_factor, ndf*next_layer_factor, n_layers)

            conv2_layers.extend(list(conv_block_instance.children()))
            self.layer_factor = next_layer_factor

        self.conv2 = nn.Sequential(*conv2_layers)

        self.conv3 = nn.Sequential(
                        nn.Conv2d(ndf*self.layer_factor,ndf*self.layer_factor,kernel_size=3,stride=1,padding=1),
                        nn.ELU(True),
                        nn.Conv2d(ndf*self.layer_factor,ndf*self.layer_factor,kernel_size=3,stride=1,padding=1),
                        nn.ELU(True)
                        )

        self.embed = nn.Linear(ndf*self.layer_factor*8*8, hidden_dim)

    def forward(self,x):
        '''
        The forward function of Encoder class.

        Parameters:
            x (tensor): The input tensor.
        
        Return:
            out (tensor): The tensor processed by the instance of Encoder.
        '''
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), self.ndf * self.layer_factor * 8 * 8)
        out = self.embed(out)
        return out

    def weight_init(self, mean, std):
        '''
        The function initates the weight of each module of the instance.

        Parameters:
            mean (float): The desired mean of the initialized weight
            std  (float): The desired standard deviation of the initialized weight.
        '''
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


class Decoder(nn.Module):
    '''
    This is a class for decoder module of both Generator and Discriminator.

    Attributes:
        ngf          (int): The number of filters in the generator.
        layer_factor (int): The scale factor of layers.
        deconv1      (nn.Sequential): 1st de-convolutional block, exponentially increase 
                                    the input size by a factor of 2.
        deconv2      (nn.Sequential): 2nd de-convolutional block, preserve the size, with 
                                    tanh activation.
        embed        (nn.Module): Linear layer, convert data from hidden dimension to the 
                                    desired size.
    '''
    def __init__(self,nc,ngf,n_layers,hidden_dim,output_dim,t_conv):
        '''
        The constructor for Decoder class.

        Parameters:
            nc         (int): Number of input channels.
            ngf        (int): The number of filters in the generator.
            n_layers   (int): The number of repetitive (Conv2d + ELU) structure.
            hidden_dim (int): The hidden dimension of the encoder. Also the output size.
            output_dim (int): The expected size of the output image (height, width).
            t_conv     (bool): Decide the way of upsampling. True if use nn.ConvTranspose2d, 
                                    False if use nn.UpsamplingNearest2d.
        '''
        power = math.log(output_dim/8, 2)

        if not power.is_integer():
            raise ValueError('The input size should be the power of 2.')
        elif output_dim <= 8:
            raise ValueError('The input should have width / height larger than 8.')

        super().__init__()

        self.embed = nn.Linear(hidden_dim, ngf*8*8)

        deconv1_layers = []

        for i in range(int(power)):
            deconv_block_instance = deconv_block(ngf, ngf, n_layers, t_conv)
            deconv1_layers.extend(list(deconv_block_instance.children()))

        self.deconv1 = nn.Sequential(*deconv1_layers)

        self.deconv2 = nn.Sequential(
                        nn.Conv2d(ngf,ngf,kernel_size=3,stride=1,padding=1),
                        nn.ELU(True),
                        nn.Conv2d(ngf,ngf,kernel_size=3,stride=1,padding=1),
                        nn.ELU(True),
                        nn.Conv2d(ngf, nc, kernel_size=3, stride=1, padding=1),
                        nn.Tanh()
                        )
        
        self.ngf = ngf

    def forward(self,x):
        '''
        The forward function of Decoder class.

        Parameters:
            x (tensor): The input tensor.
        
        Return:
            out (tensor): The tensor processed by the instance of Decoder.
        '''
        out = self.embed(x)
        out = out.view(out.size(0), self.ngf, 8, 8)
        out = self.deconv1(out)
        out = self.deconv2(out)
        return out
    
    def weight_init(self, mean, std):
        '''
        The function initates the weight of each module of the instance.

        Parameters:
            mean (float): The desired mean of the initialized weight
            std  (float): The desired standard deviation of the initialized weight.
        '''
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


class Discriminator(nn.Module):
    '''
    This is a class that construct Discriminator using Encoder class and Decoder class.

    Attributes:
        encoder (obj): An instance of class Encoder.
        decoder (obj): An instance of class Decoder.
    '''
    def __init__(self, nc, ndf, ngf, n_layers, hidden_dim, input_dim, output_dim, exp=False, t_conv=False):
        '''
        The constructor for Discriminator class.

        Parameters:
            nc         (int): Number of input channels.
            ndf        (int): The number of filters in the discriminator.
            ngf        (int): The number of filters in the generator.
            n_layers   (int): The number of repetitive (Conv2d + ELU) structure.
            hidden_dim (int): The hidden dimension of the encoder. Also the output size.
            input_dim  (int): The size of the input image (height, width).
            output_dim (int): The expected size of the output image (height, width).
            exp        (bool): Decide the way of growth of the number of layers in the 2nd 
                                conv block. True if exponentially, False if Linearly
            t_conv     (bool): Decide the way of upsampling. True if use nn.ConvTranspose2d, 
                                False if use nn.UpsamplingNearest2d.
        '''
        super().__init__()
        self.encoder = Encoder(nc,ndf,n_layers,hidden_dim,input_dim,exp)
        self.decoder = Decoder(nc,ngf,n_layers,hidden_dim,output_dim,t_conv)
        
    def weight_init(self, mean, std):
        '''
        The function initates the weight of each module of the instance.

        Parameters:
            mean (float): The desired mean of the initialized weight
            std  (float): The desired standard deviation of the initialized weight.
        '''
        self.encoder.weight_init(mean, std)
        self.decoder.weight_init(mean, std)
    
    def forward(self, img):
        '''
        The forward function of Encoder class.

        Parameters:
            x (tensor): The input tensor.
        
        Return:
            out (tensor): The tensor processed by the instance of Discriminator.
        '''
        out = self.encoder(img)
        out = self.decoder(out)
        return out

class Generator(nn.Module):
    '''
    This is a class that construct Generator using Decoder class.

    Attributes:
        decoder (obj): An instance of class Decoder.
    '''
    def __init__(self, nc, ngf, n_layers, hidden_dim, output_dim, t_conv=False):
        '''
        The constructor for Generator class.

        Parameters:
            nc         (int): Number of input channels.
            ngf        (int): The number of filters in the generator.
            n_layers   (int): The number of repetitive (Conv2d + ELU) structure.
            hidden_dim (int): The hidden dimension of the encoder. Also the output size.
            output_dim (int): The expected size of the output image (height, width).
            t_conv     (bool): Decide the way of upsampling. True if use nn.ConvTranspose2d, 
                                False if use nn.UpsamplingNearest2d.
        '''
        super().__init__()
        self.decoder = Decoder(nc,ngf,n_layers,hidden_dim,output_dim,t_conv)
    
    def weight_init(self, mean, std):
        '''
        The function initates the weight of each module of the instance.

        Parameters:
            mean (float): The desired mean of the initialized weight
            std  (float): The desired standard deviation of the initialized weight.
        '''
        self.decoder.weight_init(mean, std)
    
    def forward(self, h):
        '''
        The forward function of Encoder class.

        Parameters:
            x (tensor): The input tensor.
        
        Return:
            out (tensor): The tensor processed by the instance of Generator.
        '''
        out = self.decoder(h)
        return out
        

def normal_init(m, mean, std):
    '''
    The function initates the weight of each module of m under normal distribution.

    Parameters:
        m    (nn.Module): The network to be initialized.
        mean (float): The desired mean of the initialized weight
        std  (float): The desired standard deviation of the initialized weight.
    '''
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()


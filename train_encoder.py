import math
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from torchvision import utils as tvutils
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from model import Encoder
from dataset import TextureDataset, RandomMultiCrop, TextureDatasetLmdb
from distributed import get_rank, synchronize, reduce_loss_dict, reduce_sum, get_world_size
from args import args
import utils

SCALING_FACTOR = 1

class VGG19(torch.nn.Module):

    def __init__(self):
        super(VGG19, self).__init__()

        self.block1_conv1 = torch.nn.Conv2d(3, 64, (3,3), padding=(1,1), padding_mode='reflect')
        self.block1_conv2 = torch.nn.Conv2d(64, 64, (3,3), padding=(1,1), padding_mode='reflect')

        self.block2_conv1 = torch.nn.Conv2d(64, 128, (3,3), padding=(1,1), padding_mode='reflect')
        self.block2_conv2 = torch.nn.Conv2d(128, 128, (3,3), padding=(1,1), padding_mode='reflect')

        self.block3_conv1 = torch.nn.Conv2d(128, 256, (3,3), padding=(1,1), padding_mode='reflect')
        self.block3_conv2 = torch.nn.Conv2d(256, 256, (3,3), padding=(1,1), padding_mode='reflect')
        self.block3_conv3 = torch.nn.Conv2d(256, 256, (3,3), padding=(1,1), padding_mode='reflect')
        self.block3_conv4 = torch.nn.Conv2d(256, 256, (3,3), padding=(1,1), padding_mode='reflect')

        self.block4_conv1 = torch.nn.Conv2d(256, 512, (3,3), padding=(1,1), padding_mode='reflect')
        self.block4_conv2 = torch.nn.Conv2d(512, 512, (3,3), padding=(1,1), padding_mode='reflect')
        self.block4_conv3 = torch.nn.Conv2d(512, 512, (3,3), padding=(1,1), padding_mode='reflect')
        self.block4_conv4 = torch.nn.Conv2d(512, 512, (3,3), padding=(1,1), padding_mode='reflect')

        self.relu = torch.nn.ReLU(inplace=True)
        self.downsampling = torch.nn.AvgPool2d((2,2))

    def forward(self, image):
        
        # RGB to BGR
        image = image[:, [2,1,0], :, :]

        # [0, 1] --> [0, 255]
        image = 255 * image

        # remove average color
        image[:,0,:,:] -= 103.939
        image[:,1,:,:] -= 116.779
        image[:,2,:,:] -= 123.68

        # block1
        block1_conv1 = self.relu(self.block1_conv1(image))
        block1_conv2 = self.relu(self.block1_conv2(block1_conv1))
        block1_pool = self.downsampling(block1_conv2)

        # block2
        block2_conv1 = self.relu(self.block2_conv1(block1_pool))
        block2_conv2 = self.relu(self.block2_conv2(block2_conv1))
        block2_pool = self.downsampling(block2_conv2)

        # block3
        block3_conv1 = self.relu(self.block3_conv1(block2_pool))
        block3_conv2 = self.relu(self.block3_conv2(block3_conv1))
        block3_conv3 = self.relu(self.block3_conv3(block3_conv2))
        block3_conv4 = self.relu(self.block3_conv4(block3_conv3))
        block3_pool = self.downsampling(block3_conv4)

        # block4
        block4_conv1 = self.relu(self.block4_conv1(block3_pool))
        block4_conv2 = self.relu(self.block4_conv2(block4_conv1))
        block4_conv3 = self.relu(self.block4_conv3(block4_conv2))
        block4_conv4 = self.relu(self.block4_conv4(block4_conv3))

        return [block1_conv1, block1_conv2, block2_conv1, block2_conv2, block3_conv1, block3_conv2, block3_conv3, block3_conv4, block4_conv1, block4_conv2, block4_conv3, block4_conv4]

def slicing_loss(list_activations_generated, list_activations_example):
    
    # generate VGG19 activations
    # list_activations_generated = vgg(image_generated)
    # list_activations_example   = vgg(image_example)
    
    # iterate over layers
    loss = 0
    for l in range(len(list_activations_example)):
        # get dimensions
        b = list_activations_example[l].shape[0]
        dim = list_activations_example[l].shape[1]
        n = list_activations_example[l].shape[2]*list_activations_example[l].shape[3]
        # linearize layer activations and duplicate example activations according to scaling factor
        activations_example = list_activations_example[l].view(b, dim, n).repeat(1, 1, SCALING_FACTOR*SCALING_FACTOR)
        activations_generated = list_activations_generated[l].view(b, dim, n*SCALING_FACTOR*SCALING_FACTOR)
        # sample random directions
        Ndirection = dim
        directions = torch.randn(Ndirection, dim).to(torch.device("cuda:0"))
        directions = directions / torch.sqrt(torch.sum(directions**2, dim=1, keepdim=True))
        # project activations over random directions
        projected_activations_example = torch.einsum('bdn,md->bmn', activations_example, directions)
        projected_activations_generated = torch.einsum('bdn,md->bmn', activations_generated, directions)
        # sort the projections
        sorted_activations_example = torch.sort(projected_activations_example, dim=2)[0]
        sorted_activations_generated = torch.sort(projected_activations_generated, dim=2)[0]
        # L2 over sorted lists
        loss += torch.mean( (sorted_activations_example-sorted_activations_generated)**2 ) 
    return loss
def init_weights(m):
    if type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())
    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()

def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss

def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)
    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)
    return noises

def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)
    else:
        return make_noise(batch, latent_dim, 1, device)

def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None

def compute_gradient_penalty(real_samples, fake_samples, discriminator):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.cuda.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates).view(-1, 1)
    fake = torch.autograd.Variable(torch.cuda.FloatTensor(real_samples.shape[0], 1, device="cuda").fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty            

def train(args, loader, generator, encoder, e_optim, device, vgg, random_phase):
    loader = sample_data(loader)
    pbar = range(args.iter+1)
    if get_rank() == 0:
        print("Training set path: %s, output directory path: %s" % (args.input, args.output))
        print("General settings: image size: %d, channel_multiplier: %d, lr: %.5f"%(args.image_size[0], args.channel_multiplier, args.lr))
        print("Intra-texture settings: textures_per_batch: %d, crops_per_texture: %d"%(args.textures_per_batch, args.crops_per_texture))
        print("Inter-texture settings: WGAN: True, gp weight: %.5f, n_critic: %d, noise_dx: %r, noise_std: %.5f"%(args.gp ,args.n_critic, args.noise_dx, args.noise_std))
        print("Augmentation settings: Random Flip: %r, Random 90-degree Rotattion: %r"%(args.random_flip, args.random_90_rotate))
        print("TextonBroadcast settings: number of textons per module: %d, max resolution to apply module: %d, phase noise is %r"%(args.n_textons, args.max_texton_size, args.random_phase_noise))
        pbar = tqdm(pbar, initial=args.start_iter, file=sys.stdout)

    latent_loss_val, image_loss_val, loss_dict = 0, 0, {}
    if args.distributed:
        e_module = encoder.module
    else:
        e_module = encoder
    sample_z = torch.randn(args.batch_size, args.latent_dim, device=device) # Fixed latent vector for monitoring during training 
    # phase_noise = None if args.random_phase_noise else 0
    phase_noise = random_phase
    requires_grad(generator, False)
    requires_grad(encoder, True) 
    for idx in pbar:
        i = idx + args.start_iter
        if i > args.iter:
            print("Done!")
            break
        ##### Training Encoder with latent ##############################################    
        noise = [mixing_noise(args.batch_size, args.latent_dim, args.mixing, device)]
        fake_img, real_y = generator(noise, phase_noise=phase_noise)
        latent, pred_y = encoder(fake_img)
        y_loss = F.mse_loss(real_y, pred_y) 
        latent_loss = F.mse_loss(latent, noise[0])      
        loss_dict["latent"] = latent_loss.detach()
        loss_dict["y"] = y_loss.detach()
        
        ##### Training Encoder with image ##################################################    
        real_img = next(loader)
        real_img = real_img.to(device).view(-1, 3, args.image_size[0], args.image_size[0])
        latent,_ = encoder(real_img)
        fake_img, _ = generator([latent], phase_noise=phase_noise)
        list_activations_example = vgg(real_img)
        list_activations_generated = vgg(fake_img)
        image_loss = slicing_loss(list_activations_generated, list_activations_example)
        loss_dict["image"] = image_loss.detach()
        loss = args.latent_weight * latent_loss + args.y_weight * y_loss + args.image_weight * image_loss
        encoder.zero_grad()
        loss.backward()
        e_optim.step()

        ##### Logging #############################################################
        loss_reduced = reduce_loss_dict(loss_dict)
        latent_loss_val = loss_reduced["latent"].mean().item()
        y_loss_val = loss_reduced["y"].mean().item()
        image_loss_val = loss_reduced["image"].mean().item()
        if get_rank() == 0:
            pbar.set_description(f"latent: {latent_loss_val:4.4f}; y: {y_loss_val:4.4f}; image: {image_loss_val:4.4f}")
            if i % args.save_img_every == 0:
                with torch.no_grad():
                    img,_ = generator([sample_z.to(device)], phase_noise=phase_noise)
                    latent,_ = encoder(img)
                    sample, _ = generator([latent], phase_noise=phase_noise)
                    tvutils.save_image(sample, f"{args.output}/{str(i).zfill(10)}.png", nrow=int(sample.size(0)**0.5), normalize=True, range=(-1, 1)) 
                    
            # if wandb and args.wandb:
            #     logs = {"Generator": g_loss_val, "Discriminator": d_loss_val, "Real Score": real_score_val, "Fake Score": fake_score_val}
            #     if i % args.save_img_every == 0:
            #         sample = utils.deprocess_image(sample.detach())
            #         images = wandb.Image(sample, caption="Generated textures")
            #         logs["images"] = images
            #     wandb.log(logs)    

            if i % args.save_ckpt_every == 0:
                torch.save({"e": e_module.state_dict(), "e_optim": e_optim.state_dict(), "args": args}, f"{args.output}/{str(i).zfill(10)}.pt")

if __name__ == "__main__":
    device = args.device     
    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    random_phase = 2*math.pi*torch.rand(1).to(device)

    if args.model_name == "texture":
        from model import MultiScaleTextureGenerator
        generator = MultiScaleTextureGenerator(size=args.image_size[0], style_dim=args.latent_dim, n_mlp=args.n_mlp, channel_multiplier=args.channel_multiplier, max_texton_size=args.max_texton_size, n_textons=args.n_textons)
    else:
        from model import Generator
        generator = Generator(size=args.image_size[0], style_dim=args.latent_dim, n_mlp=args.n_mlp, channel_multiplier=args.channel_multiplier)
    generator.to(device)
    vgg = VGG19().to(torch.device("cuda"))
    vgg.load_state_dict(torch.load("vgg19.pth")) 

    if args.load_ckpt is not None:
        print("Loading %s model from %s:" % (args.model_name, args.load_ckpt))
        ckpt = torch.load(args.load_ckpt, map_location=lambda storage, loc: storage)
        generator.load_state_dict(ckpt["g_ema"])
        ckpt_name = os.path.splitext(os.path.basename(args.load_ckpt.strip("/")))[0]

    encoder = Encoder(style_dim=args.latent_dim, n_textons=args.n_textons)
    encoder.to(device)

    # Apply the initialization function to the encoder
    encoder.apply(init_weights)

    e_reg_ratio = args.e_reg_every / (args.e_reg_every + 1)
    e_optim = optim.Adam(encoder.parameters(), lr=args.lr * e_reg_ratio, betas=(0 ** e_reg_ratio, 0.99 ** e_reg_ratio))
    if args.load_e is not None:
        print("load model:", args.load_ckpt)
        ckpt = torch.load(args.load_ckpt, map_location=lambda storage, loc: storage)
        try:
            ckpt_name = os.path.basename(args.load_ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name.strip("/"))[0])
            pass
        except ValueError:
            pass 
        encoder.load_state_dict(ckpt["e"], strict=False) if "e" in ckpt else None
        
    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")  
    args.output = os.path.join(args.output, "training", args.model_name,"start_iter_%08d"%args.start_iter, timestr)
    utils.mkdir(args.output) 

    train_sub_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)])
    train_transforms = transforms.Compose([RandomMultiCrop(args.image_size[0], args.crops_per_texture, args.random_90_rotate, args.random_flip), transforms.Lambda(lambda crops: torch.stack([train_sub_transforms(crop) for crop in crops]))])
    
    
    dataset = TextureDataset(args.input, train_transforms, args.image_size[0])        

    loader = data.DataLoader(dataset, batch_size=args.textures_per_batch, sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed), drop_last=True)

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan2 texture")

    train(args, loader, generator, encoder, e_optim, device, vgg, random_phase)
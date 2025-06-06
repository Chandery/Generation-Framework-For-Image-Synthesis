import torch.nn as nn
import lightning as L
import numpy as np
import torch
from . import networks
import itertools
from utils.fid import (
    ImageListDataset, 
    get_activations_from_dataloader,
    calculate_frechet_distance,
)
from utils.image_pool import ImagePool
import PIL
import random
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid

class CycleGAN(L.LightningModule):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 netG=None,
                 netD=None,
                 n_layers_D=3,
                 init_type='normal',
                 init_gain=0.02,
                 norm='batch',
                 features=64, 
                 pool_size=50,
                 gan_mode='lsgan',
                 no_dropout=True,
                 lambda_identity=0.5,
                 lambda_A=10,
                 lambda_B=10,
                 direction='AtoB',
                 is_Train=False,
                 root_dir=None,
                 init_lr=0.0002,
                 beta=0.5,
                 scheduler_policy=None,
                 display_freq=10,
                 val_batch_num=5,
                 val_batch_total=None,
                 *args,
                 **kwargs):
        super(CycleGAN, self).__init__()
        self.is_Train = is_Train
        self.automatic_optimization = False
        
        if is_Train:
            self.val_batch_idx = [random.randint(0, val_batch_total-1) for _ in range(val_batch_num)]
            if 0 not in self.val_batch_idx:
                self.val_batch_idx[-1]=0
            print(f"val_batch_idx: {self.val_batch_idx}")
        
        self.loss_names = ['D_A', 'D_B', 'G_A', 'G_B', 'cycle_A', 'cycle_B', 'idt_A', 'idt_B']
        self.model_names = ['G_A', 'G_B', 'D_A', 'D_B'] if is_Train else ['G_A', 'G_B']
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'fake_A', 'rec_A', 'rec_B']
        if is_Train:
            self.visual_names.extend(['idt_A', 'idt_B'])
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.pool_size = pool_size
        self.gan_mode = gan_mode
        self.no_dropout = no_dropout
        self.lambda_identity = lambda_identity
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.netG = netG if netG is not None else "resnet_9blocks"
        self.netD = netD if netD is not None else "basic"
        self.init_type = init_type
        self.init_gain = init_gain
        self.n_layers_D = n_layers_D
        self.norm = norm
        self.direction = direction  
        self.root_dir = root_dir
        self.init_lr = init_lr
        self.beta = beta
        self.schedule_policy = scheduler_policy
        self.display_freq = display_freq
        
        self.netG_A = self.make_G(self.in_channels, self.out_channels, self.features, self.netG, self.norm, not self.no_dropout, self.init_type, self.init_gain)
        self.netG_B = self.make_G(self.out_channels, self.in_channels, self.features, self.netG, self.norm, not self.no_dropout, self.init_type, self.init_gain)
        
        if self.is_Train:
            if self.lambda_identity > 0.0:
                assert self.in_channels==self.out_channels, "in_channels and out_channels must be the same"
                
            self.netD_A = self.make_D(self.out_channels, self.features, self.netD, self.n_layers_D, self.norm, self.init_type, self.init_gain)
            self.netD_B = self.make_D(self.in_channels, self.features, self.netD, self.n_layers_D, self.norm, self.init_type, self.init_gain)

            self.criterionGAN = networks.GANLoss(self.gan_mode)
            self.criterionCycle = nn.L1Loss()
            self.criterionIdt = nn.L1Loss()
            
            self.fake_A_pool = ImagePool(pool_size)
            self.fake_B_pool = ImagePool(pool_size)
        
    def make_G(self, in_channels, out_channels, features, netG, norm, use_dropout, init_type, init_gain):
        return networks.define_G(in_channels, out_channels, features, netG, norm, use_dropout, init_type, init_gain)
    def make_D(self, in_channels, features, netD, n_layers_D, norm, init_type, init_gain):
        return networks.define_D(in_channels, features, netD, n_layers_D, norm, init_type, init_gain)
    
    def set_input(self, input):
        AtoB = self.direction == 'AtoB' 
        self.input = input['A'] if AtoB else input['B']
        self.target = input['B'] if AtoB else input['A']
        self.real_A = self.input
        self.real_B = self.target
        self.image_paths = input['A_paths'] if AtoB else input['B_paths']
    
    def training_step(self, batch, batch_idx):
        self.set_input(batch) 
        self.optimize_parameters()
        
        losses = self.get_current_losses()
        batch_size = batch['A'].size(0)  # 获取batch size
        for name, loss in losses.items():
            self.log(f'train/{name}', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=False, batch_size=batch_size)
    
        if batch_idx == 0: # ? update learning rate per epoch
            self.update_learning_rate()
    
    def calculate_fid(self, fake_list, real_list):
        if len(fake_list) == 0 or len(real_list) == 0:
            raise ValueError("Fake or real list is empty")
        
        fake_ds = ImageListDataset(fake_list, transforms=transforms.ToTensor())
        real_ds = ImageListDataset(real_list, transforms=transforms.ToTensor())
        fake_dl = DataLoader(fake_ds, batch_size=len(fake_list), shuffle=False, drop_last=False)
        real_dl = DataLoader(real_ds, batch_size=len(real_list), shuffle=False, drop_last=False)
        
        fake_act = get_activations_from_dataloader(fake_dl, device=self.device)
        real_act = get_activations_from_dataloader(real_dl, device=self.device)
        
        fake_mu, fake_sigma = np.mean(fake_act, axis=0), np.cov(fake_act, rowvar=False)
        real_mu, real_sigma = np.mean(real_act, axis=0), np.cov(real_act, rowvar=False)
        
        fid = calculate_frechet_distance(fake_mu, fake_sigma, real_mu, real_sigma)
        return fid
    
    def validation_step(self, batch, batch_idx):
        if batch_idx in self.val_batch_idx:
            self.set_input(batch)
            self.forward()
            self.get_loss_G()
            self.get_loss_D_A()
            self.get_loss_D_B()
            
            losses = self.get_current_losses()
            batch_size = batch['A'].size(0)  # 获取batch size
            for name, loss in losses.items():
                self.log(f'val/{name}', loss, sync_dist=False, batch_size=batch_size)
            
            
            Fake_list = []
            Real_list = []
            target_name = 'B' if self.direction == 'AtoB' else 'A'
            
            images = self.get_current_visuals()
            for name, image in images.items():
                for k in range(image.shape[0]):
                    img = image[k].cpu().numpy()
                    img = img.transpose(1, 2, 0)
                    
                    if name == f'fake_{target_name}' and batch_idx == 0:
                        Fake_list.append(img)
                    elif name == f'real_{target_name}' and batch_idx == 0:
                        Real_list.append(img)
                if batch_idx == 0:
                    imgs = (image+1)*127.5
                    imgs = imgs.type(torch.uint8)
                    row = int(image.shape[0]**0.5)
                    grid = make_grid(imgs, nrow=row)
                    
                    self.logger.experiment.add_image(f'val images/{name}', grid, self.current_epoch)
                    
            if batch_idx == 0:
                fid = self.calculate_fid(Fake_list, Real_list)
                print(f"[Validation Log] epoch {self.current_epoch} raise a validation, FID={fid}")
                self.log('val/fid', fid, sync_dist=False,batch_size=batch_size)
            
    def test_step(self, batch, batch_idx):
        self.set_input(batch)
        self.forward()
        
        filename = batch['A_paths'][0].split('/')[-1].split('.')[0]
        dir_name = f'{self.root_dir}/results/{filename}'
        
        images = self.get_current_visuals()
        for name, image in images.items():
            self.save_image(image, f'{dir_name}/{name}.png')

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = (
            self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(sd, strict=False)
        )
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
    
    def save_image(self, image, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        image = image.cpu().numpy().squeeze()
        if image.ndim == 3:
            image = image.transpose(1, 2, 0)
        image = (image + 1) / 2.0
        image = (image * 255.0).astype(np.uint8)
        image = PIL.Image.fromarray(image)
        image.save(path)
    
    def forward(self):
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)
        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)
        
    def loss_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D
    
    def get_loss_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.loss_D_basic(self.netD_A, self.real_B, fake_B)
        return self.loss_D_A

    
    def get_loss_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.loss_D_basic(self.netD_B, self.real_A, fake_A)
        return self.loss_D_B
    
    def get_loss_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.lambda_identity
        lambda_A = self.lambda_A
        lambda_B = self.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        return self.loss_G
    
    def backward_G(self):
        self.manual_backward(self.loss_G)
        
    def get_current_losses(self):
        losses = {}
        for name in self.loss_names:
            if isinstance(name, str):
                losses[name] = getattr(self, 'loss_' + name)
        return losses
    
    def get_current_visuals(self):
        visuals = {}
        for name in self.visual_names:
            visuals[name] = getattr(self, name)
        return visuals

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad
    
    def optimize_parameters(self):
        opt_G, opt_D= self.optimizers() # !!!!! VeryImportant! if dont use this, self.global_step wil not be updated
        self.forward()
        
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        opt_G.zero_grad()
        loss_G=self.get_loss_G()
        self.manual_backward(loss_G)
        opt_G.step()

        self.set_requires_grad([self.netD_A, self.netD_B], True)
        opt_D.zero_grad()
        loss_D_A=self.get_loss_D_A()
        loss_D_B=self.get_loss_D_B()
        self.manual_backward(loss_D_A)
        self.manual_backward(loss_D_B)
        opt_D.step()
        
    def update_learning_rate(self):
        old_lr = self.optimizers()[0].param_groups[0]['lr']
        
        # ** check if optimizer has executed step
        # ? this is for fixing the bug of lightning
        for optimizer in self.optimizers():
            # ? check if optimizer has executed step
            if len(optimizer.state) == 0:
                return
        
        # ? update learning rate only if optimizer has executed step
        for scheduler in self.lr_schedulers():
            if self.schedule_policy.lr_policy == 'plateau':
                scheduler.step(self.schedule_policy.metric)
            else:
                scheduler.step()

        lr = self.optimizers()[0].param_groups[0]['lr']
        if old_lr != lr:
            print('change learning rate %.7f -> %.7f' % (old_lr, lr))
        
    def configure_optimizers(self):
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), 
                                                            self.netG_B.parameters()), 
                                            lr=self.init_lr, 
                                            betas=(self.beta, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), 
                                                            self.netD_B.parameters()), 
                                            lr=self.init_lr, 
                                            betas=(self.beta, 0.999))
        self.optimizers_list = [self.optimizer_G, self.optimizer_D]
        
        if self.schedule_policy is not None:
            self.schedulers_list = [networks.get_scheduler(optimizer, self.schedule_policy) for optimizer in self.optimizers_list]
            schedulers = [{
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            } for scheduler in self.schedulers_list]
            return [self.optimizer_G, self.optimizer_D], schedulers
        return [self.optimizer_G, self.optimizer_D]
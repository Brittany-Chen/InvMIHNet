import logging
import os
from collections import OrderedDict

import torch
from torch.nn.parallel import DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import ReconstructionLoss
from models.modules.Quantization import Quantization
import numpy as np
import models.modules.Unet_common as common
from models.model import *
from .model import *
from PIL import Image

logger = logging.getLogger('base')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise


def guide_loss(output, bicubic_image):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(output, bicubic_image)
    return loss.to(device)


def reconstruction_loss(rev_input, input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(rev_input, input)
    return loss.to(device)


def low_frequency_loss(ll_input, gt_input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(ll_input, gt_input)
    return loss.to(device)



def image_save(img,save_dir,img_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img_path = os.path.join(save_dir, img_name)
    at_images_np = img.detach().cpu().numpy()
    adv_img = at_images_np[0]
    adv_img = np.moveaxis(adv_img, 0, 2)
    img_pil = Image.fromarray(adv_img.astype(np.uint8))
    img_pil.save(img_path)


dwt = common.DWT()
iwt = common.IWT()



class InvMIHNet(BaseModel):
    def __init__(self, opt):
        super(InvMIHNet, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        test_opt = opt['test']
        self.train_opt = train_opt
        self.test_opt = test_opt

        self.netG = networks.define_G(opt).to(self.device)
        self.netH = Model().to(self.device)
        init_model(self.netH)

        # print network
        self.print_network()
        self.load()
        self.load_H(self.netH)

        self.Quantization = Quantization()

        if self.is_train:
            self.netG.train()
            self.netH.train()

            # loss
            self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw'])
            self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])


            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params_G = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params_G.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params_G, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            optim_params_H = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params_H.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_H = torch.optim.Adam(optim_params_H, lr=train_opt['lr_H'], betas=(train_opt['beta1_H'], train_opt['beta2_H']), eps=1e-6, weight_decay=train_opt['weight_decay_H'])

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.schedulers_H = torch.optim.lr_scheduler.StepLR(self.optimizer_H, train_opt['weight_step'], gamma=train_opt['lr_gamma'])

            self.log_dict = OrderedDict()

    def feed_data(self, data):
        self.ref_L = data['LQ'].to(self.device)  # LQ
        self.real_H = data['GT'].to(self.device)  # GT

    def gaussian_batch(self, dims):
        return torch.randn(tuple(dims)).to(self.device)

    def loss_forward(self, out, y, z):
        l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)

        z = z.reshape([out.shape[0], -1])
        l_forw_ce = self.train_opt['lambda_ce_forw'] * torch.sum(z**2) / z.shape[0]

        return l_forw_fit, l_forw_ce

    def loss_backward(self, x, y):
        x_samples = self.netG(x=y, rev=True)
        x_samples_image = x_samples[:, :3, :, :]
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(x, x_samples_image)

        return l_back_rec

    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def optimize_parameters(self, step):
        # torch.autograd.set_detect_anomaly(True)

        # downscaling
        self.input = self.real_H[1: ,: , :, :]
        self.output = self.netG(x=self.input)

        zshape = self.output[:, 3:, :, :].shape
        LR_ref = self.ref_L[1: ,: , :, :].detach()

        secret = self.output[:, :3, :, :]
        new_H = int(self.input.shape[2] / self.opt['scale_H'])
        new_W = int(self.input.shape[3] / self.opt['scale_W'])
        if self.opt['scale_H'] * self.opt['scale_W'] == 4:
            cover = torch.cat((self.real_H[:1, :3, :new_H, :new_W], self.real_H[:1, :3, :new_H, new_W:],
                               self.real_H[:1, :3, new_H:, :new_W], self.real_H[:1, :3, new_H:, new_W:]), dim=0)
        elif self.opt['scale_H'] * self.opt['scale_W'] == 6:
            cover = torch.cat((self.real_H[:1, :3, :new_H, :new_W], self.real_H[:1, :3, :new_H, new_W:new_W * 2],
                               self.real_H[:1, :3, new_H:new_H * 2, :new_W],
                               self.real_H[:1, :3, new_H:new_H * 2, new_W:new_W * 2],
                               self.real_H[:1, :3, new_H * 2:new_H * 4, :new_W],
                               self.real_H[:1, :3, new_H * 2:new_H * 3, new_W:new_W * 2],), dim=0)
        elif self.opt['scale_H'] * self.opt['scale_W'] == 8:
            cover = torch.cat((self.real_H[:1, :3, :new_H, :new_W], self.real_H[:1, :3, :new_H, new_W:new_W * 2],
                               self.real_H[:1, :3, new_H:new_H * 2, :new_W],
                               self.real_H[:1, :3, new_H:new_H * 2, new_W:new_W * 2],
                               self.real_H[:1, :3, new_H * 2:new_H * 3, :new_W],
                               self.real_H[:1, :3, new_H * 2:new_H * 3, new_W:new_W * 2],
                               self.real_H[:1, :3, new_H * 3:new_H * 4, :new_W],
                               self.real_H[:1, :3, new_H * 3:new_H * 4, new_W:new_W * 2]), dim=0)
        elif self.opt['scale_H'] * self.opt['scale_W'] == 9:
            cover = torch.cat((self.real_H[:1, :3, :new_H, :new_W], self.real_H[:1, :3, :new_H, new_W:new_H * 2],
                               self.real_H[:1, :3, :new_H, new_W * 2:new_W * 3],
                               self.real_H[:1, :3, new_H:new_H * 2, :new_W],
                               self.real_H[:1, :3, new_H:new_H * 2, new_W:new_H * 2],
                               self.real_H[:1, :3, new_H:new_H * 2, new_W * 2:new_W * 3],
                               self.real_H[:1, :3, new_H * 2:new_H * 3, :new_W],
                               self.real_H[:1, :3, new_H * 2:new_H * 3, new_W:new_H * 2],
                               self.real_H[:1, :3, new_H * 2:new_H * 3, new_W * 2:new_W * 3],), dim=0)
        elif self.opt['scale_H'] * self.opt['scale_W'] == 16:
            cover = torch.cat((self.real_H[:1, :3, :new_H, :new_W], self.real_H[:1, :3, :new_H, new_W:new_W * 2],
                               self.real_H[:1, :3, :new_H, new_W * 2:new_W * 3],
                               self.real_H[:1, :3, :new_H, new_W * 3:new_W * 4],
                               self.real_H[:1, :3, new_H:new_H * 2, :new_W],
                               self.real_H[:1, :3, new_H:new_H * 2, new_W:new_W * 2],
                               self.real_H[:1, :3, new_H:new_H * 2, new_W * 2:new_W * 3],
                               self.real_H[:1, :3, new_H:new_H * 2, new_W * 3:new_W * 4],
                               self.real_H[:1, :3, new_H * 2:new_H * 3, :new_W],
                               self.real_H[:1, :3, new_H * 2:new_H * 3, new_W:new_W * 2],
                               self.real_H[:1, :3, new_H * 2:new_H * 3, new_W * 2:new_W * 3],
                               self.real_H[:1, :3, new_H * 2:new_H * 3, new_W * 3:new_W * 4],
                               self.real_H[:1, :3, new_H * 3:new_H * 4, :new_W],
                               self.real_H[:1, :3, new_H * 3:new_H * 4, new_W:new_W * 2],
                               self.real_H[:1, :3, new_H * 3:new_H * 4, new_W * 2:new_W * 3],
                               self.real_H[:1, :3, new_H * 3:new_H * 4, new_W * 3:new_W * 4],
                               ), dim=0)

        cover_input = dwt(cover)
        secret_input = dwt(secret)
        input_img = torch.cat((cover_input, secret_input), 1)

        # hiding
        output = self.netH(input_img)

        channel_in = self.opt['network_G']['in_nc']

        output_steg = output.narrow(1, 0, 4 * channel_in)
        output_z = output.narrow(1, 4 * channel_in, output.shape[1] - 4 * channel_in)
        steg_img = iwt(output_steg)
        steg_img = self.Quantization(steg_img)

        # concealing
        output_z_guass = gauss_noise(output_z.shape)

        output_rev = torch.cat((output_steg, output_z_guass), 1)
        output_image = self.netH(output_rev, rev=True)

        secret_rev = output_image.narrow(1, 4 * channel_in, output_image.shape[1] - 4 * channel_in)
        secret_rev_1 = iwt(secret_rev)


        # loss functions
        g_loss = guide_loss(steg_img.cuda(), cover.cuda())
        r_loss = reconstruction_loss(secret_rev_1, secret[:,:3,:,:])
        steg_low = output_steg.narrow(1, 0, channel_in)
        cover_low = cover_input.narrow(1, 0, channel_in)
        l_loss = low_frequency_loss(steg_low, cover_low)

        total_loss = self.train_opt['lamda_reconstruction'] * r_loss + self.train_opt['lamda_guide'] * g_loss + self.train_opt['lamda_low_frequency'] * l_loss

        l_forw_fit, l_forw_ce = self.loss_forward(secret_rev_1, LR_ref, self.output[:, 3:, :, :])

        # upscaling
        LR = self.Quantization(secret_rev_1)
        gaussian_scale = self.train_opt['gaussian_scale'] if self.train_opt['gaussian_scale'] != None else 1
        y_ = torch.cat((LR, gaussian_scale * self.gaussian_batch(zshape)), dim=1)

        l_back_rec = self.loss_backward(self.real_H[1:,:,:,:], y_)

        total_loss.backward(retain_graph=True)

        loss = l_forw_fit + l_back_rec + l_forw_ce
        print("step", step, "_loss:", loss)
        loss.backward()

        # gradient clipping
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])
            nn.utils.clip_grad_norm_(self.netH.parameters(), self.train_opt['gradient_clipping'])

        self.optimizer_G.step()
        self.optimizer_H.step()
        self.optimizer_H.zero_grad()
        self.optimizer_G.zero_grad()

        if step % self.opt['logger']['save_checkpoint_freq'] == 0:
            # save_path = os.path.join(self.opt['path']['models'], save_filename)
            torch.save({'net': self.netH.state_dict()}, os.path.join(self.opt['path']['models'], 'IIH_' + str(step)+'.pth'))


    def test(self):
        Lshape = self.ref_L[1:, :, :, :].shape

        input_dim = Lshape[1]
        self.input = self.real_H[1:]

        zshape = [Lshape[0], input_dim * (self.opt['scale_W'] * self.opt['scale_H']) - Lshape[1], Lshape[2], Lshape[3]]

        gaussian_scale = 1
        if self.test_opt and self.test_opt['gaussian_scale'] != None:
            gaussian_scale = self.test_opt['gaussian_scale']

        self.netG.eval()
        self.netH.eval()
        new_H = int(self.input.shape[2] / self.opt['scale_H'])
        new_W = int(self.input.shape[3] / self.opt['scale_W'])
        if self.opt['scale_H']*self.opt['scale_W'] == 4:
            cover = torch.cat((self.real_H[:1, :3, :new_H, :new_W], self.real_H[:1, :3, :new_H, new_W:], self.real_H[:1, :3, new_H:, :new_W], self.real_H[:1, :3, new_H:, new_W:]), dim=0)
        elif self.opt['scale_H']*self.opt['scale_W'] == 6:
            cover = torch.cat((self.real_H[:1, :3, :new_H, :new_W], self.real_H[:1, :3, :new_H, new_W:new_W * 2],
                               self.real_H[:1, :3, new_H:new_H * 2, :new_W], self.real_H[:1, :3, new_H:new_H * 2, new_W:new_W * 2],
                               self.real_H[:1, :3, new_H*2:new_H*4, :new_W], self.real_H[:1, :3, new_H*2:new_H*3, new_W:new_W * 2],), dim=0)
        elif self.opt['scale_H']*self.opt['scale_W'] == 8:
            cover = torch.cat((self.real_H[:1, :3, :new_H, :new_W], self.real_H[:1, :3, :new_H, new_W:new_W * 2], self.real_H[:1, :3, new_H:new_H * 2, :new_W], self.real_H[:1, :3, new_H:new_H * 2, new_W:new_W*2],
                               self.real_H[:1, :3, new_H * 2:new_H * 3, :new_W], self.real_H[:1, :3, new_H * 2:new_H * 3, new_W:new_W * 2], self.real_H[:1, :3, new_H * 3:new_H * 4, :new_W], self.real_H[:1, :3, new_H * 3:new_H * 4, new_W:new_W * 2]), dim=0)
        elif self.opt['scale_H']*self.opt['scale_W'] == 9:
            cover = torch.cat((self.real_H[:1, :3, :new_H, :new_W], self.real_H[:1, :3, :new_H, new_W:new_H * 2], self.real_H[:1, :3, :new_H, new_W * 2:new_W * 3],
                               self.real_H[:1, :3, new_H:new_H * 2, :new_W], self.real_H[:1, :3, new_H:new_H * 2, new_W:new_H * 2], self.real_H[:1, :3, new_H:new_H * 2, new_W * 2:new_W * 3],
                               self.real_H[:1, :3, new_H * 2:new_H * 3, :new_W], self.real_H[:1, :3, new_H * 2:new_H * 3, new_W:new_H * 2], self.real_H[:1, :3, new_H * 2:new_H * 3, new_W * 2:new_W * 3],), dim=0)
        elif self.opt['scale_H']*self.opt['scale_W'] == 16:
            cover = torch.cat((self.real_H[:1, :3, :new_H, :new_W], self.real_H[:1, :3, :new_H, new_W:new_W * 2],
                               self.real_H[:1, :3, :new_H, new_W * 2:new_W * 3],
                               self.real_H[:1, :3, :new_H, new_W * 3:new_W * 4],
                               self.real_H[:1, :3, new_H:new_H * 2, :new_W],
                               self.real_H[:1, :3, new_H:new_H * 2, new_W:new_W * 2],
                               self.real_H[:1, :3, new_H:new_H * 2, new_W * 2:new_W * 3],
                               self.real_H[:1, :3, new_H:new_H * 2, new_W * 3:new_W * 4],
                               self.real_H[:1, :3, new_H * 2:new_H * 3, :new_W],
                               self.real_H[:1, :3, new_H * 2:new_H * 3, new_W:new_W * 2],
                               self.real_H[:1, :3, new_H * 2:new_H * 3, new_W * 2:new_W * 3],
                               self.real_H[:1, :3, new_H * 2:new_H * 3, new_W * 3:new_W * 4],
                               self.real_H[:1, :3, new_H * 3:new_H * 4, :new_W],
                               self.real_H[:1, :3, new_H * 3:new_H * 4, new_W:new_W * 2],
                               self.real_H[:1, :3, new_H * 3:new_H * 4, new_W * 2:new_W * 3],
                               self.real_H[:1, :3, new_H * 3:new_H * 4, new_W * 3:new_W * 4],
                               ), dim=0)

        self.cover = self.real_H[:1, :3, :, :]
        self.secret = self.real_H[1:, :3, :, :]

        with torch.no_grad():
            output = self.netG(x=self.input)[:, :3, :, :]
            cover_input = dwt(cover)
            secret_input = dwt(output)
            input_img = torch.cat((cover_input, secret_input), 1)
            channel_in = self.opt['network_G']['in_nc']

            output_inn = self.netH(input_img)
            output_steg = output_inn.narrow(1, 0, 4 * channel_in)
            output_z = output_inn.narrow(1, 4 * channel_in, output_inn.shape[1] - 4 * channel_in)
            steg_img = iwt(output_steg)

            output_z_guass = gauss_noise(output_z.shape)

            output_rev = torch.cat((output_steg, output_z_guass), 1)
            output_image = self.netH(output_rev, rev=True)

            secret_rev = output_image.narrow(1, 4 * channel_in, output_image.shape[1] - 4 * channel_in)
            secret_rev = iwt(secret_rev)

            self.forw_L = secret_rev
            self.forw_L = self.Quantization(self.forw_L).cuda()
            y_forw = torch.cat((self.forw_L, gaussian_scale * self.gaussian_batch(zshape)), dim=1)
            self.fake_H = self.netG(x=y_forw, rev=True)[:, :3, :, :]

            self.secret_recover = self.fake_H

            if self.opt['scale_H'] * self.opt['scale_W'] == 4:
                steg_1 = torch.cat((steg_img[0], steg_img[1]), 2)
                steg_2 = torch.cat((steg_img[2], steg_img[3]), 2)
                steg = torch.cat((steg_1, steg_2), 1)
            elif self.opt['scale_H'] * self.opt['scale_W'] == 6:
                steg_1 = torch.cat((steg_img[0], steg_img[1]), 2)
                steg_2 = torch.cat((steg_img[2], steg_img[3]), 2)
                steg_3 = torch.cat((steg_img[4], steg_img[5]), 2)
                steg = torch.cat((steg_1, steg_2, steg_3), 1)
            elif self.opt['scale_H'] * self.opt['scale_W'] == 8:
                steg_1 = torch.cat((steg_img[0], steg_img[1]), 2)
                steg_2 = torch.cat((steg_img[2], steg_img[3]), 2)
                steg_3 = torch.cat((steg_img[4], steg_img[5]), 2)
                steg_4 = torch.cat((steg_img[6], steg_img[7]), 2)
                steg = torch.cat((steg_1, steg_2, steg_3, steg_4), 1)
            elif self.opt['scale_H'] * self.opt['scale_W'] == 9:
                steg_1 = torch.cat((steg_img[0], steg_img[1], steg_img[2]), 2)
                steg_2 = torch.cat((steg_img[3], steg_img[4], steg_img[5]), 2)
                steg_3 = torch.cat((steg_img[6], steg_img[7], steg_img[8]), 2)
                steg = torch.cat((steg_1, steg_2, steg_3), 1)
            elif self.opt['scale_H'] * self.opt['scale_W'] == 16:
                steg_1 = torch.cat((steg_img[0], steg_img[1], steg_img[2], steg_img[3]), 2)
                steg_2 = torch.cat((steg_img[4], steg_img[5], steg_img[6], steg_img[7]), 2)
                steg_3 = torch.cat((steg_img[8], steg_img[9], steg_img[10], steg_img[11]), 2)
                steg_4 = torch.cat((steg_img[12], steg_img[13], steg_img[14], steg_img[15]), 2)
                steg = torch.cat((steg_1, steg_2, steg_3, steg_4), 1)
            self.steg = self.Quantization(steg)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['cover'] = self.cover.detach().float().cpu()
        out_dict['secret'] = self.secret.detach().float().cpu()
        out_dict['secret_recover'] = self.secret_recover.detach().float().cpu()
        out_dict['steg'] = self.steg.detach().float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network IIR structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for IIR [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def load_H(self, net):
        load_path_H = self.opt['path']['pretrain_model_H']
        if load_path_H is not None:
            logger.info('Loading model for IIH [{:s}] ...'.format(load_path_H))
            state_dicts = torch.load(load_path_H)
            network_state_dict = {k.replace("module.", ""): v for k, v in state_dicts['net'].items() if
                                  'tmp_var' not in k}
            network_state_dict = {k: v for k, v in network_state_dict.items() if 'rect' not in k}
            net.load_state_dict(network_state_dict)

    def save(self, iter_label):
        self.save_network(self.netG, 'IIR', iter_label)

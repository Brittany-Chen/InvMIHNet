import torch
import logging
import models.modules.discriminator_vgg_arch as SRGAN_arch
from models.modules.Inv_arch import *
from models.modules.Subnet_constructor import subnet
import math
logger = logging.getLogger('base')


####################
# define network
####################
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    subnet_type = which_model['subnet_type']
    if opt_net['init']:
        init = opt_net['init']
    else:
        init = 'xavier'

    down_num = len(opt_net['block_num'])
    # down_num = int(math.log(opt_net['scale_W'], 2))

    use_ConvDownsampling = False

    if which_model['use_ConvDownsampling']:
        use_ConvDownsampling = True

    netG = InvRescaleNet(opt_net['in_nc'], opt_net['out_nc'], subnet(subnet_type, init), opt_net['block_num'], down_num, use_ConvDownsampling=use_ConvDownsampling, down_scale_W=opt_net['scale_W'], down_scale_H=opt_net['scale_H'])

    return netG

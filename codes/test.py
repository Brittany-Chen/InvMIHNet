import os
import os.path as osp
import logging
import time
import argparse
import torchvision
from collections import OrderedDict

import numpy as np
import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model


#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

model = create_model(opt)
count=0
for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    for data in test_loader:
        model.feed_data(data)

        model.test()

        visuals = model.get_current_visuals()

        cover = visuals['cover']
        secret = visuals['secret']
        secret_recover = visuals['secret_recover']
        steg = visuals['steg']

        cover_path = dataset_dir + "\\" + "cover"
        secret_path = dataset_dir + "\\" + "secret"
        secret_recover_path = dataset_dir + "\\" + "secret_recover"
        steg_path = dataset_dir + "\\" + "steg"


        if not os.path.exists(cover_path):
            os.makedirs(cover_path)
        if not os.path.exists(secret_path):
            os.makedirs(secret_path)
        if not os.path.exists(secret_recover_path):
            os.makedirs(secret_recover_path)
        if not os.path.exists(steg_path):
            os.makedirs(steg_path)


        save_cover_path = osp.join(cover_path, str(count)+'_clean_cover.png')
        torchvision.utils.save_image(cover, save_cover_path)

        save_steg_path = osp.join(steg_path, str(count) + '_steg.png')
        torchvision.utils.save_image(steg, save_steg_path)

        for i in range(int(secret.shape[0])):
            save_secret_path = osp.join(secret_path, str(count) + '_' + str(i) + '_clean_secret.png')
            save_secret_recover_path = osp.join(secret_recover_path, str(count) + '_' + str(i) + '_secret_recover.png')
            torchvision.utils.save_image(secret[i:i+1,:,:,:], save_secret_path)
            torchvision.utils.save_image(secret_recover[i:i + 1, :, :, :], save_secret_recover_path)

        count = count + 1







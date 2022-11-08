import pprint
import sys
from subprocess import Popen, PIPE
# importing numpy to fix https://github.com/pytorch/pytorch/issues/37377
import numpy
import torch
import glob
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torchvision import transforms
import torch.optim as optim
from FSDnoisy18k.dataset.FSDnoisy18k import *
from IPython import embed
sys.path.append('./src')
sys.path.append('./da')
from data_augment import SpecAugment, RandomGaussianBlur, GaussNoise, RandTimeShift, RandFreqShift, TimeReversal, Compander
from rnd_resized_crop import RandomResizedCrop_diy

from utils_train_eval import *

import models.audio_models as audio_mod
import models.resnetAudio as resnet_mod
import models.MLP_head as mlp
import yaml
import random

import os
import argparse

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def parse_args():
    parser = argparse.ArgumentParser(description='Code for self_supervised_ACL')
    parser.add_argument('-p', '--params_yaml', dest='params_yaml', action='store', required=False, type=str)

    config = parser.parse_args()

    print('\nYaml file with parameters defining the experiment: %s\n' % str(config.params_yaml))

    # Read parameters file from yaml passed by argument
    args = yaml.full_load(open(config.params_yaml))

    return args

def my_collate(batch):
    # for validation and test
    imgs, targets, index = zip(*batch)
    individual_sizes = []
    for i in range(len(imgs)):
        individual_sizes.append(imgs[i].size()[0])
    #Repeat tensors to know each sample index!!

    return torch.cat(imgs), torch.from_numpy(np.concatenate((targets))), torch.from_numpy(np.repeat(index, individual_sizes))

def data_config(args):

    output, _ = Popen('uname', stdout=PIPE).communicate()
    print(output.decode("utf-8"))

    trainset, valset, testset = get_dataset(args)

    mean = [trainset.data_mean]
    std = [trainset.data_std]

    train_transform = transforms.Compose([
        RandTimeShift(do_rand_time_shift=args['da']['do_rand_time_shift'], Tshift=args['da']['Tshift']),
        RandFreqShift(do_rand_freq_shift=args['da']['do_rand_freq_shift'], Fshift=args['da']['Fshift']),
        RandomResizedCrop_diy(do_randcrop=args['da']['do_randcrop'], scale=args['da']['rc_scale'],
                              ratio=args['da']['rc_ratio']),
        transforms.RandomApply([TimeReversal(do_time_reversal=args['da']['do_time_reversal'])], p=0.5),
        Compander(do_compansion=args['da']['do_compansion'], comp_alpha=args['da']['comp_alpha']),
        SpecAugment(do_time_warp=args['da']['do_time_warp'], W=args['da']['W'],
                    do_freq_mask=args['da']['do_freq_mask'], F=args['da']['F'], m_f=args['da']['m_f'],
                    reduce_mask_range=args['da']['reduce_mask_range'],
                    do_time_mask=args['da']['do_time_mask'], T=args['da']['T'], m_t=args['da']['m_t'],
                    mask_val=args['da']['mask_val']),
        GaussNoise(stdev_gen=args['da']['awgn_stdev_gen']),
        RandomGaussianBlur(do_blur=args['da']['do_blur'], max_ksize=args['da']['blur_max_ksize'],
                           stdev_x=args['da']['blur_stdev_x']),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),

    ])

    trainset.transform = train_transform
    valset.transform = test_transform
    testset.transform = test_transform

    trainset.pslab_transform = test_transform

    #__import__("pdb").set_trace()


    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args['learn']['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
    #embed()
    track_loader = torch.utils.data.DataLoader(trainset, batch_size=args['learn']['batch_size'], shuffle=False, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args['learn']['test_batch_size'], collate_fn=my_collate, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args['learn']['test_batch_size'], collate_fn=my_collate, shuffle=False, num_workers=8, pin_memory=True)

    print('############# Data loaded #############')

    return train_loader, val_loader, test_loader, track_loader


def main(args):

    print('\nExperimental setup:')
    print('pctrl=')
    pprint.pprint(args['ctrl'], width=1, indent=4)
    print('plearn=')
    pprint.pprint(args['learn'], width=1, indent=4)
    print('pda=')
    pprint.pprint(args['da'], width=1, indent=4)

    job_start = time.time()

    #os.environ['CUDA_VISIBLE_DEVICES'] = str(args['learn']['cuda_dev'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device is', device)

    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(args['learn']['seed_initialization'])  # CPU seed
    if device == "cuda":
        torch.cuda.manual_seed_all(args['learn']['seed_initialization'])  # GPU seed

    random.seed(args['learn']['seed_initialization'])  # python seed for image transformation

    train_loader, val_loader, test_loader, track_loader = data_config(args)
    st = time.time()

    # select model===========================================================================
    # select model===========================================================================
    # note model outputs a certain embedding size (h_i, h_j) which must match input to head
    if args['learn']['network'] == 'res18':
        model = resnet_mod.resnet18(args, num_classes=args["learn"]["num_classes"]).to(device)

        if args['learn']['downstream']==1:
            #model.load_state_dict(torch.load("pth/ResNet_best.pth"))
            #model_dir = 'results_models/models_res18_finetune_noisy10_SI271828/'
            model_dir = 'results_models/models_res18_finetune_clean'+os.environ['SLURM_ARRAY_TASK_ID']+'_SI271828/'
            model_dir = glob.glob(model_dir+"best*.pth")[0]
            #model_dir = 'results_models/models_res18_unsup'+os.environ['SLURM_ARRAY_TASK_ID']+'_SI271828/best.pth'
            model.load_state_dict(torch.load(model_dir))
            print('loaded model from ', model_dir)

            #model.fc = torch.nn.Linear(512, args["learn"]["num_classes"]).to(device)




    # training loop===========================================================================
        # test

        # loss_test, acc_test, acc_test_balanced, acc_test_per_class = 0.0, 0.0, 0.0, 0.0
    loss_test, acc_test, acc_test_balanced, acc_test_per_class = eval_model(args, model, device, test_loader)
    print('Balanced_Accuracy: {:.3f}\n'.format(acc_test_balanced))
    print('Average_Accuracy: {:.3f}'.format(acc_test))

if __name__ == "__main__":
    args = parse_args()
    #__import__("pdb").set_trace()
    args['learn']['experiment_name']+= str(os.environ['SLURM_ARRAY_TASK_ID'])
    #args['learn']['experiment_name']+= str(10)
    #print(args)
    # train
    main(args)

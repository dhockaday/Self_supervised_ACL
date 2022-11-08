import pprint
from subprocess import Popen, PIPE
import sys
import glob
# importing numpy to fix https://github.com/pytorch/pytorch/issues/37377
import numpy
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torchvision import transforms
import torch.optim as optim
from FSDnoisy18k.dataset.FSDnoisy18k import *
import models.MLP_head as mlp
sys.path.append('./src')
sys.path.append('./da')

from data_augment import SpecAugment, RandomGaussianBlur, GaussNoise, RandTimeShift, RandFreqShift, TimeReversal, Compander
from rnd_resized_crop import RandomResizedCrop_diy

from utils_train_eval import *
from utils.criterion import *
import models.audio_models as audio_mod
import models.resnetAudio as resnet_mod
import models.MLP_head as mlp
import yaml
import random

import os
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Code for self_supervised_ACL')
    parser.add_argument('-p', '--params_yaml', dest='params_yaml', action='store', required=False, type=str)
    #parser.add_argument('-test_type',type=str,required=True)

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

    _, _, testset = get_dataset(args)

    mean =[-1.4003801669235545]
    std = [6.049307765609413e-05]

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),

    ])


    testset.transform = test_transform

    test_loader = torch.utils.data.DataLoader(testset, batch_size=args['learn']['test_batch_size'], collate_fn=my_collate, shuffle=False, num_workers=8, pin_memory=True)

    print('############# test Data loaded #############')
    return test_loader



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device is', device)
    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(args['learn']['seed_initialization'])  # CPU seed
    if device == "cuda":
        torch.cuda.manual_seed_all(args['learn']['seed_initialization'])  # GPU seed

    random.seed(args['learn']['seed_initialization'])  # python seed for image transformation

    test_loader = data_config(args)

    model = resnet_mod.resnet18(args, num_classes=args["learn"]["num_classes"]).to(device)
    model.fc = torch.nn.Linear(512, args["learn"]["num_classes"]).to(device)
    #model_dir = 'results_models/models_res18_lin_eval'+str(os.environ['SLURM_ARRAY_TASK_ID'])+'_SI271828/'
    model_dir = 'results_models/models_res18_lin_eval10_SI271828/'
    model_dir = glob.glob(model_dir+"best*.pth")[0]
    model.load_state_dict(torch.load(model_dir))
    print('loaded model from ', model_dir)

    loss_test, acc_test, acc_test_balanced, acc_test_per_class = eval_model(args, model, device, test_loader)

    print('Average  Test Accuracy: {:.3f}\n'.format(acc_test))
    print('balanced Test Accuracy: {:.3f}\n'.format(acc_test_balanced))
    print('balanced Test Accuracy per class: {:.3f}\n'.format(acc_test_per_class))


if __name__ == "__main__":
    args = parse_args()

    main(args)

import argparse
import os
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
from utils import Logger, mkdir_p, AverageMeter
from dataloader_segments import Dataset_training, Dataset_testing
from torch.utils import data
import math
from tools import numerically_integrate_rk4
# from models_change import NeurVec_b1,NeurVec_b2,NeurVec_b3,NeurVec_b4, NeurVec_b5,NeurVec_b6,NeurVec_b7,NeurVec_b8,NeurVec_b9,NeurVec_b10, NeurVec_b11,NeurVec_b12,NeurVec_b13,NeurVec_b14,NeurVec_b15,NeurVec_b19,NeurVec_b20,NeurVec_b21,NeurVec_b22,NeurVec_b23,NeurVec_b24,NeurVec_b25,NeurVec_b26,NeurVec_b27,NeurVec_b28,NeurVec_b29,NeurVec_shuffle_true,NeurVec_morelayer,NeurVec_swish,NeurVec_norm,NeurVec_higher_rational,NeurVec_c1,NeurVec_c2, NeurVec_c3, NeurVec_shuffle_true_jump, NeurVec_shuffle_true_mlp
from models_change import *

parser = argparse.ArgumentParser(description='PyTorch')
# Datasets
parser.add_argument('--train_dir', default='', type=str, help='train set dir')
parser.add_argument('--test_dir', default='', type=str, help='test set dir')
parser.add_argument('--ckpt', default='test', type=str, help='checkpoint')

parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--workers', default=2, type=int, metavar='N', help='number of data loading workers (default: 32)')# 将workers从32修改到了4，电脑内存不足
parser.add_argument('--gpu_id', default=0, type=str, help='gpu id')

parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--batch_size', default=32, type=int, help='batch size') # 初始是32
parser.add_argument('--epoch', default=1000, type=int, help='epoch')

parser.add_argument('--train_coarse', default=1, type=int, help='train coarse')

parser.add_argument('--dt', default=0.1, type=float, help='dt')
parser.add_argument('--T_train', default=10, type=int, help='train time step')
parser.add_argument('--T_test', default=10, type=int, help='test time step')
parser.add_argument('--length', default=499, type=int, help='length')

parser.add_argument('--manualSeed', default=None, type=int,help='seed')

parser.add_argument('--model_name', type=str, help='')
parser.add_argument('--num_layer', default=1, type=int, help='number of hidden layers')

parser.add_argument('--solver', type=str, default='--')
parser.add_argument('--latter_filename', default='test', type=str, help='the name related for log.txt')

# torch.cuda.empty_cache()

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
print(state)

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = 1
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
np.random.seed(args.manualSeed)

if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

def main():
    torch.cuda.empty_cache()
    if not os.path.isdir(args.ckpt):
        mkdir_p(args.ckpt)

    with open(args.ckpt + "/config.txt", 'w+') as f:
        for (k, v) in args._get_kwargs():
            f.write(k + ' : ' + str(v) + '\n')

    # Model
    if args.model_name == 'orgNN':
        if args.latter_filename == 'b1':
            model = NeurVec_b1()
        elif args.latter_filename == 'b2':
            model = NeurVec_b2()
        elif args.latter_filename == 'b3':
            model = NeurVec_b3()
        elif args.latter_filename == 'b4':
            model = NeurVec_b4()
        elif args.latter_filename == 'b5':
            model = NeurVec_b5()
        elif args.latter_filename == 'b6':
            model = NeurVec_b6()
        elif args.latter_filename == 'b7':
            model = NeurVec_b7()
        elif args.latter_filename == 'b8':
            model = NeurVec_b8()
        elif args.latter_filename == 'b9':
            model = NeurVec_b9()
        elif args.latter_filename == 'b10':
            model = NeurVec_b10()
        elif args.latter_filename == 'b11':
            model = NeurVec_b11()
        elif args.latter_filename == 'b12':
            model = NeurVec_b12()
        elif args.latter_filename == 'b13':
            model = NeurVec_b13()
        elif args.latter_filename == 'b14':
            model = NeurVec_b14()
        elif args.latter_filename == 'b15':
            model = NeurVec_b15()
        elif args.latter_filename == 'b19':
            model = NeurVec_b19()
        elif args.latter_filename == 'b20':
            model = NeurVec_b20()
        elif args.latter_filename == 'b21':
            model = NeurVec_b21()
        elif args.latter_filename == 'b22':
            model = NeurVec_b22()
        elif args.latter_filename == 'b23':
            model = NeurVec_b23()
        elif args.latter_filename == 'b24':
            model = NeurVec_b24()
        elif args.latter_filename == 'b25':
            model = NeurVec_b25()
        elif args.latter_filename == 'b26':
            model = NeurVec_b26()
        elif args.latter_filename == 'b27':
            model = NeurVec_b27()
        elif args.latter_filename == 'b28':
            model = NeurVec_b28()
        elif args.latter_filename == 'b29':
            model = NeurVec_b29()
        elif args.latter_filename == 'shuffle_true':
            model = NeurVec_shuffle_true()
        elif args.latter_filename == 'morelayer':
            model = NeurVec_morelayer()
        elif args.latter_filename == 'swish':
            model = NeurVec_swish()
        elif args.latter_filename == 'norm':
            model = NeurVec_norm()
        elif args.latter_filename == 'higher_rational':
            model = NeurVec_higher_rational()
        elif args.latter_filename == 'c1':
            model = NeurVec_c1()
        elif args.latter_filename == 'c2':
            model = NeurVec_c2()
        elif args.latter_filename == 'c3':
            model = NeurVec_c3()
        elif args.latter_filename == 'shuffle_true_jump':
            model = NeurVec_shuffle_true_jump()
        elif args.latter_filename == 'shuffle_true_mlp':
            model = NeurVec_shuffle_true_mlp()        
        elif args.latter_filename == 'shuffle_true_nihe':
            model = NeurVec_shuffle_true_nihe()
        elif args.latter_filename == 'shuffle_true_double':
            model = NeurVec_shuffle_true_double()
        elif args.latter_filename == 'shuffle_true_channelwise':
            model = NeurVec_shuffle_true_channelwise()
        elif args.latter_filename == 'shuffle_true_4channelwise':
            model = NeurVec_shuffle_true_4channelwise()
        elif args.latter_filename == 'shuffle_true_channelwiseFine':
            model = NeurVec_shuffle_true_channelwiseFine()
        elif args.latter_filename == 'shuffle_true_groupchannelwise':
            model = NeurVec_shuffle_true_groupchannelwise()
        elif args.latter_filename == 'shuffle_true_channelwise_nolinear':
            model = NeurVec_shuffle_true_channelwise_nolinear()
        elif args.latter_filename == 'shuffle_true_channelwise_pingyi':
            model = NeurVec_shuffle_true_channelwise_pingyi()
        elif args.latter_filename == 'shuffle_true_SinChannelwise':
            model = NeurVec_shuffle_true_SinChannelwise()
        elif args.latter_filename == 'shuffle_true_SirenChannelwise':
            model = NeurVec_shuffle_true_SirenChannelwise()
        elif args.latter_filename == 'shuffle_true_SirenChannelwise_wise':
            model = NeurVec_shuffle_true_SirenChannelwise_wise()
        elif args.latter_filename == 'shuffle_true_gated':
            model = NeurVec_shuffle_true_gated()
        elif args.latter_filename == 'shuffle_true_RAFs':
            model = NeurVec_shuffle_true_RAFs()
        elif args.latter_filename == 'shuffle_true_gaussian':
            model = NeurVec_shuffle_true_gaussian()
        elif args.latter_filename == 'shuffle_true_SinChannelwise_reparam1':
            model = NeurVec_shuffle_true_SinChannelwise_reparam1()
        elif args.latter_filename == 'shuffle_true_SinChannelwise_reparam2':
            model = NeurVec_shuffle_true_SinChannelwise_reparam2()
        elif args.latter_filename == 'shuffle_true_SinChannelwise_reparam3':
            model = NeurVec_shuffle_true_SinChannelwise_reparam3()
        elif args.latter_filename == 'shuffle_true_SinChannelwise_reparam4':
            model = NeurVec_shuffle_true_SinChannelwise_reparam4()
        elif args.latter_filename == 'shuffle_true_SinChannelwise_bias':
            model = NeurVec_shuffle_true_SinChannelwise_bias()
        elif args.latter_filename == 'shuffle_true_Siren0_noise0':
            model = NeurVec_shuffle_true_Siren0_noise0()
        elif args.latter_filename == 'shuffle_true_Siren1_noise0':
            model = NeurVec_shuffle_true_Siren1_noise0()
        elif args.latter_filename == 'shuffle_true_Siren2_noise0':
            model = NeurVec_shuffle_true_Siren2_noise0()
        elif args.latter_filename == 'shuffle_true_Siren3_noise0':
            model = NeurVec_shuffle_true_Siren3_noise0()
        elif args.latter_filename == 'shuffle_true_SinChannelwise_withlinear':
            model = NeurVec_shuffle_true_SinChannelwise_withlinear()
        elif args.latter_filename == 'shuffle_true_SinChannelwise_mogai1':
            model = NeurVec_shuffle_true_SinChannelwise_mogai1()
        elif args.latter_filename == 'shuffle_true_SinChannelwise_mogai2':
            model = NeurVec_shuffle_true_SinChannelwise_mogai2()
        elif args.latter_filename == 'shuffle_true_SinChannelwise_mogai3':
            model = NeurVec_shuffle_true_SinChannelwise_mogai3()
        elif args.latter_filename == 'shuffle_true_RAFsFine':
            model = NeurVec_shuffle_true_RAFsFine()
        elif args.latter_filename == 'shuffle_true_SinChannelwise_mogai5':
            model = NeurVec_shuffle_true_SinChannelwise_mogai5()
        elif args.latter_filename == 'shuffle_true_wholeRAFs':
            model = NeurVec_shuffle_true_wholeRAFs()
        elif args.latter_filename == 'shuffle_true_Quadratic':
            model = NeurVec_shuffle_true_Quadratic()
        elif args.latter_filename == 'shuffle_true_Laplacian':
            model = NeurVec_shuffle_true_Laplacian()
        elif args.latter_filename == 'shuffle_true_Supergaussian':
            model = NeurVec_shuffle_true_Supergaussian()
        elif args.latter_filename == 'shuffle_true_Expsin':
            model = NeurVec_shuffle_true_Expsin()
        elif args.latter_filename == 'shuffle_true_singleLaplacian':
            model = NeurVec_shuffle_true_singleLaplacian()
        elif args.latter_filename == 'shuffle_true_singleLaplacian_reparam1':
            model = NeurVec_shuffle_true_singleLaplacian_reparam1()
        elif args.latter_filename == 'shuffle_true_singleLaplacian_reparam2':
            model = NeurVec_shuffle_true_singleLaplacian_reparam2()
        elif args.latter_filename == 'shuffle_true_singleLaplacian_reparam3':
            model = NeurVec_shuffle_true_singleLaplacian_reparam3()
        elif args.latter_filename == 'shuffle_true_singleLaplacian_reparam4':
            model = NeurVec_shuffle_true_singleLaplacian_reparam4()
        elif args.latter_filename == 'shuffle_true_channelshareAndnolinearnobias':
            model = NeurVec_shuffle_true_channelshareAndnolinearnobias()

    else:
        model = None
    model = model.cuda()
    cudnn.benchmark = True

    print('Total params: %.6f' % (sum(p.numel() for p in model.parameters())))

    """
    Define Residual Methods and Optimizer
    """
    criterion = nn.MSELoss()
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    title = ''
    logger = Logger(os.path.join(args.ckpt, 'log.txt'), title=title)
    logger.set_names(['Learning Rate', '10000*Train Mse', '10000*Test Mse'])
    # Train and test
    training_set = Dataset_training(data_dir=args.train_dir, train_T=args.T_train, seg_num=args.length)
    trainloader = data.DataLoader(training_set, shuffle=True, batch_size=args.batch_size, num_workers=args.workers)

    testing_set = Dataset_testing(data_dir=args.test_dir)
    testloader = data.DataLoader(testing_set, shuffle=False, batch_size=200, num_workers=args.workers)

    current_iters = 0
    # 数据：(500, 300000, 4) # T_train = 10
    iters_per_epoch = len(trainloader) # 每个epoch中的迭代次数
    print('the length of iters_per_epoch is : %d', iters_per_epoch)
    zero_time = time.time()

    # print('??? len')
    # print(iters_per_epoch)  #74850000 = 300000 * args.length(499) / batch size(2) 
    # print(1)
    for epoch in range(args.epoch): # 1000
        train_mse = AverageMeter()
        test_mse = AverageMeter()
        ## training
        # 取 5% 的数据
        # print(model.error.nn[1].a5)
        for trajectories in trainloader:
            lr = cosine_lr(optimizer, args.lr, current_iters, iters_per_epoch * args.epoch)
            train_loss = train(trajectories, model, criterion, optimizer, epoch, current_iters, len(trainloader), lr)
            train_mse.update(train_loss, trajectories.size(0))
            current_iters += 1
        ## testing
        for trajectories in testloader:
            test_loss = test(trajectories, model, criterion, epoch)
            test_mse.update(test_loss, trajectories.size(0))
        # print(train_mse.avg)
        # print('000000000000000000000000000000000000000000000000')
        logger.append([lr, 10000*train_mse.avg, 10000*test_mse.avg])
        epoch_time = time.time() - zero_time
        print('--------------------')
        print('it spends %.2f minutes for epoch %d',epoch_time/60, epoch)
    # save model
        if (epoch+1) % 3 == 0 or epoch==199:
            save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                            checkpoint=args.ckpt,filename='ckpt'+'_'+str(epoch)+'.pth.tar')
    logger.append([0,0,0])
    logger.append([0,0,(time.time()-zero_time)/60])
    logger.close()

def train(trajectories, model, criterion, optimizer, epoch, current_iters, total_iters, lr):
    suffix = ''

    # switch to train mode
    model.train()
    start_time = time.time()

    # data
    trajectories = torch.FloatTensor(trajectories).cuda() # Bs*T*dim
    true_traj = trajectories.permute(1,0,2) # T*Bs*dim

    volatile = False

    coarse = args.train_coarse
    predict_traj = numerically_integrate_rk4(true_traj[0], model, args.T_train, args.dt, volatile, coarse=coarse)
    # print(predict_traj.shape)       # [10, 1, 4] -> [T, Bs, dim]
    # print(predict_traj[1:].shape)
    # print(true_traj.shape)          # [10, 1, 4]
    # print(true_traj[1:].shape)
    loss = criterion(predict_traj[1:], true_traj[1:])*1000000

    # compute gradient and do optim step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time = time.time() - start_time

    suffix += 'Training epoch: {epoch} iter: {iter:.0f} / {total_iters:.0f}|lr: {lr:.4f}| Batch: {bt:.2f}s | training MSE: {mse: .8f} |'.format(bt=batch_time,
            iter=current_iters, lr=lr, mse=loss.item(), epoch=epoch, total_iters=total_iters)
    print(suffix)

    return loss.item()

def test(trajectories, model, criterion, epoch):
    suffix = ''

    # switch to evaluation mode
    model.eval()
    start_time = time.time()

    # data
    trajectories = torch.FloatTensor(trajectories).cuda() # Bs*T*dim
    true_traj = trajectories.permute(1,0,2) # T*Bs*dim

    volatile = True
    coarse = args.train_coarse

    predict_traj = numerically_integrate_rk4(true_traj[0], model, args.T_test, args.dt, volatile, coarse=coarse)

    # print(predict_traj.size(), true_traj.size())
    loss = criterion(predict_traj, true_traj[:args.T_test])

    # measure elapsed time
    batch_time = time.time() - start_time
    # print('max idx: ', torch.mean((predict_traj - true_traj[:args.T_test]) ** 2, dim=(0, 2)).argmax())
    # print('max val: ', torch.mean((predict_traj - true_traj[:args.T_test]) ** 2, dim=(0, 2)).max())
    # print([v.item() for v in torch.mean((predict_traj-true_traj[:args.T_test])**2, dim=(0,2))])
    suffix += 'Testing epoch: {epoch} | Batch: {bt:.2f}s | test MSE: {mse: .8f} |'.format(bt=batch_time, mse=loss.item(), epoch=epoch)
    print(suffix)

    return loss.item()

def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def cosine_lr(opt, base_lr, e, epochs):
    lr = 0.5 * base_lr * (math.cos(math.pi * e / epochs) + 1)
    for param_group in opt.param_groups:
        param_group["lr"] = lr
    return lr

if __name__ == '__main__':
    main()

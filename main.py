import os
import time
import datetime
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
import argparse
import shutil

from get_model import get_model
from get_data import get_dataset
from new_utils import train, test, save_arguments_to_file

def parse_option():
    parser = argparse.ArgumentParser('GNN-SE')
    # parser.add_argument('--model_type', type=str, default="gnn", help='gnn or se')
    parser.add_argument('--model_type', type=str, default="gnn", help="'gnn', 'se', 'eca', 'cbam', 'sa', 'vanilla'")
    parser.add_argument('--model', type=str, default="resnet18", help='resnet18 or resnet50 or alexnet')
    parser.add_argument('--data', type=str, default="imagenet1k", help='imagenet100 or cifar10')
    parser.add_argument('--data_path', type=str, default="/mnt/SSD_2/Alok/IISc/parikshit/dataset/imagenet1k/data1k", help='path to data')
    parser.add_argument('--n_epochs', type=int, default=100, help='num of epochs')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--results_dir', type=str, default="./results",
                        help='path to results dir')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--lr', type=float, default=0.1, help='LR')
    parser.add_argument('--gnn_lr', type=float, default=0.1, help='GNN LR')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='decay for optim')
    # parser.add_argument('--scale_features', type=int, default=0, help='1 to scale and 0 to not scale')
    parser.add_argument('--scheduler_step_num', type=int, default=30, help='num of epochs')
    parser.add_argument('--save_weights', action='store_true', help='save weights or not')
    parser.add_argument('--use_spp', action='store_true', help='save weights or not')
    parser.add_argument('--batch_size', type=int, default=256, help='BS')
    parser.add_argument('--comments', type=str, default="", help='use this to distinguish two weights/text files')

    # parser.add_argument('--lr', default=0.1, help='')
    # parser.add_argument('--batch_size', type=int, default=256, help='')
    parser.add_argument('--num_workers', type=int, default=16, help='')

    parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--dist-url', default='tcp://10.72.33.229:1234', type=str, help='')#127.0.0.1
    parser.add_argument('--dist-backend', default='nccl', type=str, help='')
    parser.add_argument('--rank', default=0, type=int, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')

    args = parser.parse_args()
    gpu_devices = ','.join([str(id) for id in args.gpu_devices])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    if args.comments != "":
        print(f"Comments added -> {args.comments}")
    else:
        print("No comments given")
    os.makedirs(args.results_dir, exist_ok=True)
    return args


def save_checkpoint(state, is_best, filename, file_path):
    full_path = os.path.join(file_path,filename)
    torch.save(state, full_path)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def main():
    args = parse_option()
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    ngpus_per_node = torch.cuda.device_count()
    print("Use GPU: {} for training".format(args.gpu))
    args.rank = args.rank * ngpus_per_node + gpu

    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    torch.distributed.barrier()

    print('==> Making model..')
    model = get_model(args.model, args.model_type,1000, args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.num_workers = int(args.num_workers / ngpus_per_node)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    print('==> Preparing data..')
    if args.data == "imagenet100":
        args.data_path = "/mnt/SSD_2/Alok/IISc/parikshit/dataset/sampled_imagenet100"
    dataset_train, dataset_test = get_dataset(args.data, args.data_path)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size,
                              shuffle=(train_sampler is None), num_workers=args.num_workers,
                              sampler=train_sampler)

    test_loader = DataLoader(dataset_test, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parms for model: {total_params}")
    #raise
    args.model_params = total_params

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    ### resume model training
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    ###

    # lr_lambda = lambda epoch: 0.1 ** (epoch // args.scheduler_step_num)
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    file_path = None
    results_dir = args.results_dir + f"/{args.data}/{args.model}/{args.model_type}"
    os.makedirs(results_dir, exist_ok=True)
    if args.model_type == "gnn":
        file_path = results_dir + f"/{args.model_type}_lr{args.lr}_{args.comments}.txt"
    else:
        file_path = results_dir + f"/{args.model_type}_lr{args.lr}.txt"
    args.file_path = file_path
    save_arguments_to_file(args, file_path)

    for epoch in range(args.start_epoch, args.n_epochs):
        train_sampler.set_epoch(epoch)
        train(train_loader, model, criterion, optimizer, epoch, args)
        if (epoch + 1) % 2 == 0:
            test(test_loader, model, criterion, epoch, args)
        adjust_learning_rate(optimizer, epoch, args, args.scheduler_step_num, 0.1)

        if args.save_weights and (epoch+1)%2 == 0:
            print("Reached here -------------")
            save_checkpoint({
                'epoch': epoch + 1,
                # 'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename=f'{args.model_type}_lr{args.lr}_{args.comments}_checkpoint_{epoch+1}.pth.tar',file_path=results_dir)
    


def adjust_learning_rate(optimizer, epoch, opt, interval=30, reduction_factor=0.1):
    if epoch % interval == 0 and epoch != 0:
        for param_group in optimizer.param_groups:
            # Only adjust the learning rate for the CNN's parameter group
            # if param_group['lr'] > opt.gnn_lr:  # assuming CNN's lr is higher than GNN's lr
            param_group['lr'] *= reduction_factor


if __name__=='__main__':
    torch.cuda.empty_cache()
    main()

import _init_path
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import os
import argparse
import logging
from functools import partial

from lib.net.point_rcnn import PointRCNN

import lib.net.train_functions as train_functions

from lib.datasets.kitti_rcnn_dataset import KittiRCNNDataset
from lib.config import cfg, cfg_from_file, save_config_to_file, cfg_from_list
import tools.train_utils.train_utils as train_utils
from tools.train_utils.fastai_optim import OptimWrapper
from tools.train_utils import learning_schedules_fastai as lsf
from tools.train_utils import common_utils

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--cfg_file', type=str, default='cfgs/LI_Fusion_with_attention_use_ce_loss.yaml',
                    help='specify the config for training')
parser.add_argument("--train_mode", type=str, default='rpn', required=True, help="specify the training mode")
parser.add_argument("--batch_size", type=int, default=16, required=True, help="batch size for training")
parser.add_argument("--epochs", type=int, default=200, required=True, help="Number of epochs to train for")

parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
parser.add_argument("--ckpt_save_interval", type=int, default=5, help="number of training epochs")
parser.add_argument('--output_dir', type=str, default=None, help='specify an output directory if needed')
parser.add_argument('--mgpus', action='store_true', default=False, help='whether to use multiple gpu')

parser.add_argument("--ckpt", type=str, default=None, help="continue training from this checkpoint")
parser.add_argument("--rpn_ckpt", type=str, default=None, help="specify the well-trained rpn checkpoint")

parser.add_argument("--gt_database", type=str, default=None,
                    help='generated gt database for augmentation')
parser.add_argument("--rcnn_training_roi_dir", type=str, default=None,
                    help='specify the saved rois for rcnn training when using rcnn_offline mode')
parser.add_argument("--rcnn_training_feature_dir", type=str, default=None,
                    help='specify the saved features for rcnn training when using rcnn_offline mode')

parser.add_argument('--train_with_eval', action='store_true', default=False,
                    help='whether to train with evaluation')
parser.add_argument("--rcnn_eval_roi_dir", type=str, default=None,
                    help='specify the saved rois for rcnn evaluation when using rcnn_offline mode')
parser.add_argument("--rcnn_eval_feature_dir", type=str, default=None,
                    help='specify the saved features for rcnn evaluation when using rcnn_offline mode')
parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                    help='set extra config keys if needed')
parser.add_argument('--model_type', type=str, default='base', help='model type')
parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
args = parser.parse_args()


def create_logger(log_file, local_rank=0):
    print('local_rank in create_logger: ', local_rank)
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.DEBUG if local_rank == 0 else logging.ERROR, format=log_format,
                        filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def create_dataloader(logger, training, dist=False):
    DATA_PATH = os.path.join('../', 'data')

    # create dataloader
    if training:
        dataset = KittiRCNNDataset(root_dir=DATA_PATH, npoints=cfg.RPN.NUM_POINTS, split=cfg.TRAIN.SPLIT,
                                   mode='TRAIN',
                                   logger=logger,
                                   classes=cfg.CLASSES,
                                   rcnn_training_roi_dir=args.rcnn_training_roi_dir,
                                   rcnn_training_feature_dir=args.rcnn_training_feature_dir,
                                   gt_database_dir=args.gt_database)
    else:
        dataset = KittiRCNNDataset(root_dir=DATA_PATH, npoints=cfg.RPN.NUM_POINTS, split=cfg.TRAIN.VAL_SPLIT,
                                   mode='EVAL',
                                   logger=logger,
                                   classes=cfg.CLASSES,
                                   rcnn_eval_roi_dir=args.rcnn_eval_roi_dir,
                                   rcnn_eval_feature_dir=args.rcnn_eval_feature_dir)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
    dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True,
                            num_workers=args.workers,
                            shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
                            drop_last=True, sampler=sampler)
    return dataset, dataloader, sampler


def create_optimizer(model):
    if cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                              momentum=cfg.TRAIN.MOMENTUM)
    elif cfg.TRAIN.OPTIMIZER == 'adam_onecycle':
        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

        optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(model), wd=cfg.TRAIN.WEIGHT_DECAY, true_wd=True, bn_wd=True
        )

        # fix rpn: do this since we use costomized optimizer.step
        if cfg.RPN.ENABLED and cfg.RPN.FIXED:
            for param in model.rpn.parameters():
                param.requires_grad = False
    else:
        raise NotImplementedError

    return optimizer


def create_scheduler(optimizer, total_steps, last_epoch):
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in cfg.TRAIN.DECAY_STEP_LIST:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * cfg.TRAIN.LR_DECAY
        return max(cur_decay, cfg.TRAIN.LR_CLIP / cfg.TRAIN.LR)

    def bnm_lmbd(cur_epoch):
        cur_decay = 1
        for decay_step in cfg.TRAIN.BN_DECAY_STEP_LIST:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * cfg.TRAIN.BN_DECAY
        return max(cfg.TRAIN.BN_MOMENTUM * cur_decay, cfg.TRAIN.BNM_CLIP)

    if cfg.TRAIN.OPTIMIZER == 'adam_onecycle':
        lr_scheduler = lsf.OneCycle(
            optimizer, total_steps, cfg.TRAIN.LR, list(cfg.TRAIN.MOMS), cfg.TRAIN.DIV_FACTOR, cfg.TRAIN.PCT_START
        )
    else:
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

    bnm_scheduler = train_utils.BNMomentumScheduler(model, bnm_lmbd, last_epoch=last_epoch)
    return lr_scheduler, bnm_scheduler


if __name__ == "__main__":
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    print(cfg.TRAIN.RPN_TRAIN_WEIGHT, cfg.TRAIN.RCNN_TRAIN_WEIGHT)
    # input()
    cfg.LOCAL_RANK = args.local_rank
    print('local rank: ', args.local_rank)
    cfg.TAG = os.path.splitext(os.path.basename(args.cfg_file))[0]
    print('launcher: ', args.launcher)
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True
        print('devices_ids: %d/%d=%d' % (
            cfg.LOCAL_RANK, torch.cuda.device_count(), cfg.LOCAL_RANK % torch.cuda.device_count()))

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus
    print('args.batch_size: ', args.batch_size)
    if args.train_mode == 'rpn':
        cfg.RPN.ENABLED = True
        cfg.RCNN.ENABLED = False
        root_result_dir = os.path.join('../', 'output', 'rpn', cfg.TAG)
    elif args.train_mode == 'rcnn':
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = cfg.RPN.FIXED = True
        root_result_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG)
    elif args.train_mode == 'rcnn_online':
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = True
        cfg.RPN.FIXED = False
        root_result_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG)
    elif args.train_mode == 'rcnn_offline':
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = False
        root_result_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG)
    else:
        raise NotImplementedError

    if args.output_dir is not None:
        root_result_dir = args.output_dir
    if cfg.LOCAL_RANK == 0:
        os.makedirs(root_result_dir, exist_ok=True)
        # copy important files to backup
        backup_dir = os.path.join(root_result_dir, 'backup_files')
        os.makedirs(backup_dir, exist_ok=True)
        os.system('cp *.py %s/' % backup_dir)
        os.system('cp -r train_utils/ %s/' % backup_dir)
        os.system('cp -r ../lib/ %s/' % backup_dir)
        os.system('cp %s %s/' % (args.cfg_file, backup_dir))

    log_file = os.path.join(root_result_dir, 'log_train.txt')
    logger = create_logger(log_file, cfg.LOCAL_RANK)
    logger.info('**********************Start logging**********************')

    # log to file
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))

    save_config_to_file(cfg, logger=logger)

    # tensorboard log
    print(root_result_dir)
    tb_log = SummaryWriter(logdir=os.path.join(root_result_dir, 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # create dataloader & network & optimizer
    train_set, train_loader, train_sampler = create_dataloader(logger, training=True, dist=dist_train)
    if args.train_with_eval:
        test_set, test_loader, sampler = create_dataloader(logger, training=False, dist=dist_train)
    # model = PointRCNN(num_classes=train_loader.dataset.num_class, use_xyz=True, mode='TRAIN')
    fn_decorator = train_functions.model_joint_fn_decorator()

    model = PointRCNN(num_classes=train_loader.dataset.num_class, use_xyz=True, mode='TRAIN')
    model.cuda()
    optimizer = create_optimizer(model)

    model.train()
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model,find_unused_parameters=True, check_reduction=True,
                                                    device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    # model.cuda()

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.ckpt is not None:
        pure_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        it, start_epoch = train_utils.load_checkpoint(pure_model, optimizer, filename=args.ckpt, logger=logger)
        last_epoch = start_epoch + 1

    lr_scheduler, bnm_scheduler = create_scheduler(optimizer, total_steps=len(train_loader) * args.epochs,
                                                   last_epoch=last_epoch)

    if args.rpn_ckpt is not None:
        pure_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        total_keys = pure_model.state_dict().keys().__len__()
        train_utils.load_part_ckpt(pure_model, filename=args.rpn_ckpt, logger=logger, total_keys=total_keys)

    if cfg.TRAIN.LR_WARMUP and cfg.TRAIN.OPTIMIZER != 'adam_onecycle':
        lr_warmup_scheduler = train_utils.CosineWarmupLR(optimizer, T_max=cfg.TRAIN.WARMUP_EPOCH * len(train_loader),
                                                         eta_min=cfg.TRAIN.WARMUP_MIN)
    else:
        lr_warmup_scheduler = None

    # start training
    logger.info('**********************Start training**********************')
    ckpt_dir = os.path.join(root_result_dir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    trainer = train_utils.Trainer(
        model,
        # train_functions.model_joint_fn_decorator(),
        fn_decorator,
        optimizer,
        ckpt_dir=ckpt_dir,
        lr_scheduler=lr_scheduler,
        bnm_scheduler=bnm_scheduler,
        # model_fn_eval=train_functions.model_joint_fn_decorator(),
        model_fn_eval=fn_decorator,
        tb_log=tb_log,
        eval_frequency=1,
        lr_warmup_scheduler=lr_warmup_scheduler,
        warmup_epoch=cfg.TRAIN.WARMUP_EPOCH,
        grad_norm_clip=cfg.TRAIN.GRAD_NORM_CLIP
    )

    trainer.train(
        it,
        start_epoch,
        args.epochs,
        train_loader,
        # test_loader,
        ckpt_save_interval=args.ckpt_save_interval,
        lr_scheduler_each_iter=(cfg.TRAIN.OPTIMIZER == 'adam_onecycle')
    )

    logger.info('**********************End training**********************')

import logging
import os
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import tqdm
import torch.optim.lr_scheduler as lr_sched
import math

logging.getLogger(__name__).addHandler(logging.StreamHandler())
cur_logger = logging.getLogger(__name__)


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch = -1,
            setter = set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError("Class '{}' is not a PyTorch nn Module".format(type(model).__name__))

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch = None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))


class CosineWarmupLR(lr_sched._LRScheduler):
    def __init__(self, optimizer, T_max, eta_min = 0, last_epoch = -1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 - math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]


def checkpoint_state(model = None, optimizer = None, epoch = None, it = None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, (torch.nn.DataParallel,torch.nn.parallel.DistributedDataParallel)):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return { 'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state }


def save_checkpoint(state, filename = 'checkpoint'):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


def load_checkpoint(model = None, optimizer = None, filename = 'checkpoint', logger = cur_logger):
    if os.path.isfile(filename):
        logger.info("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch'] if 'epoch' in checkpoint.keys() else -1
        it = checkpoint.get('it', 0.0)
        if model is not None and checkpoint['model_state'] is not None:
            # check module prefix
            checkpoint_state = checkpoint['model_state']
            if len(checkpoint_state.keys())>0 and 'module.' in list(checkpoint_state.keys())[0]:
                new_state_dict={}
                for k in checkpoint_state.keys():
                    new_state_dict[k[7:]]=checkpoint_state[k]
                model.load_state_dict(new_state_dict,strict=True)
            else:
                model.load_state_dict(checkpoint_state,strict=True)
        else:
            raise FileNotFoundError('')
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        logger.info("==> Done")
    else:
        raise FileNotFoundError

    return it, epoch


def load_part_ckpt(model, filename, logger = cur_logger, total_keys = -1):
    if os.path.isfile(filename):
        logger.info("==> Loading part model from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model_state = checkpoint['model_state']

        update_model_state = { key: val for key, val in model_state.items() if key in model.state_dict() }
        state_dict = model.state_dict()
        state_dict.update(update_model_state)
        model.load_state_dict(state_dict)

        update_keys = update_model_state.keys().__len__()
        if update_keys == 0:
            raise RuntimeError
        logger.info("==> Done (loaded %d/%d)" % (update_keys, total_keys))
    else:
        raise FileNotFoundError


class Trainer(object):
    def __init__(self, model, model_fn, optimizer, ckpt_dir, lr_scheduler, bnm_scheduler,
                 model_fn_eval, tb_log, eval_frequency = 1, lr_warmup_scheduler = None, warmup_epoch = -1,
                 grad_norm_clip = 1.0,rank =0,train_sampler=None):
        self.model, self.model_fn, self.optimizer, self.lr_scheduler, self.bnm_scheduler, self.model_fn_eval = \
            model, model_fn, optimizer, lr_scheduler, bnm_scheduler, model_fn_eval

        self.ckpt_dir = ckpt_dir
        self.eval_frequency = eval_frequency
        self.tb_log = tb_log
        self.lr_warmup_scheduler = lr_warmup_scheduler
        self.warmup_epoch = warmup_epoch
        self.grad_norm_clip = grad_norm_clip
        self.rank = rank
        print('Trainer rank: ',self.rank)
        self.train_sampler =None

    def _train_it(self, batch):
        self.model.train()

        self.optimizer.zero_grad()
        loss, tb_dict, disp_dict = self.model_fn(self.model, batch)

        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
        self.optimizer.step()

        return loss.item(), tb_dict, disp_dict

    def eval_epoch(self, d_loader):
        self.model.eval()

        eval_dict = { }
        total_loss = count = 0.0

        # eval one epoch
        for i, data in tqdm.tqdm(enumerate(d_loader, 0), total = len(d_loader), leave = False, desc = 'val'):
            self.optimizer.zero_grad()

            loss, tb_dict, disp_dict = self.model_fn_eval(self.model, data)

            total_loss += loss.item()
            count += 1
            for k, v in tb_dict.items():
                eval_dict[k] = eval_dict.get(k, 0) + v

        # statistics this epoch
        for k, v in eval_dict.items():
            eval_dict[k] = eval_dict[k] / max(count, 1)

        cur_performance = 0
        if 'recalled_cnt' in eval_dict:
            eval_dict['recall'] = eval_dict['recalled_cnt'] / max(eval_dict['gt_cnt'], 1)
            cur_performance = eval_dict['recall']
        elif 'iou' in eval_dict:
            cur_performance = eval_dict['iou']

        return total_loss / count, eval_dict, cur_performance

    def train(self, start_it, start_epoch, n_epochs, train_loader, test_loader = None, ckpt_save_interval = 5,
              lr_scheduler_each_iter = False):
        eval_frequency = self.eval_frequency if self.eval_frequency > 0 else 1

        it = start_it
        with tqdm.trange(start_epoch, n_epochs, desc = 'epochs',dynamic_ncols=True,leave=(self.rank==0)) as tbar:

            for epoch in tbar:
                if self.train_sampler is not None:
                    self.train_sampler.set_epoch(epoch)
                if self.lr_scheduler is not None and self.warmup_epoch <= epoch and (not lr_scheduler_each_iter):
                    self.lr_scheduler.step(epoch)

                if self.bnm_scheduler is not None:
                    self.bnm_scheduler.step(it)
                    if self.tb_log is not None:
                        self.tb_log.add_scalar('bn_momentum', self.bnm_scheduler.lmbd(epoch), it)

                # train one epoch
                if self.rank == 0:
                    pbar = tqdm.tqdm(total=len(train_loader),leave=False, desc='train', dynamic_ncols=True)
                    pbar.set_postfix(dict(total_it=it))
                dataloader_iter = iter(train_loader)
                for cur_it in range(len(train_loader)):
                    try:
                        batch = next(dataloader_iter)
                    except StopIteration:
                        dataloader_iter = iter(train_loader)
                        batch = next(dataloader_iter)
                        print('new iters')

                    if lr_scheduler_each_iter:
                        self.lr_scheduler.step(it)
                        cur_lr = float(self.optimizer.lr)
                        if self.tb_log is not None:
                            self.tb_log.add_scalar('learning_rate', cur_lr, it)
                    else:
                        if self.lr_warmup_scheduler is not None and epoch < self.warmup_epoch:
                            self.lr_warmup_scheduler.step(it)
                            cur_lr = self.lr_warmup_scheduler.get_lr()[0]
                        else:
                            cur_lr = self.lr_scheduler.get_lr()[0]

                    loss, tb_dict, disp_dict = self._train_it(batch)
                    it += 1
                    if self.rank==0:
                        disp_dict.update({ 'loss': loss, 'lr': cur_lr })

                        # log to console and tensorboard
                        pbar.update()
                        pbar.set_postfix(dict(total_it = it))
                        tbar.set_postfix(disp_dict)
                        tbar.refresh()

                        if self.tb_log is not None:
                            self.tb_log.add_scalar('train_loss', loss, it)
                            self.tb_log.add_scalar('learning_rate', cur_lr, it)
                            for key, val in tb_dict.items():
                                self.tb_log.add_scalar('train_' + key, val, it)

                # save trained model
                trained_epoch = epoch + 1
                if trained_epoch % ckpt_save_interval == 0 and self.rank==0:
                    ckpt_name = os.path.join(self.ckpt_dir, 'checkpoint_epoch_%d' % trained_epoch)
                    save_checkpoint(
                            checkpoint_state(self.model, self.optimizer, trained_epoch, it), filename = ckpt_name,
                    )

                if self.rank ==0:
                    pbar.close()


        return None


def load_joint_data_to_gpu(data,cfg):
    sample_id, pts_rect, pts_features, pts_input = \
        data['sample_id'], data['pts_rect'], data['pts_features'], data['pts_input']

    inputs = torch.from_numpy(pts_input).cuda(non_blocking=True).float()

    input_data = {'pts_input': inputs}
    # img feature
    if cfg.LI_FUSION.ENABLED:
        pts_origin_xy, img = data['pts_origin_xy'], data['img']
        pts_origin_xy = torch.from_numpy(pts_origin_xy).cuda(non_blocking=True).float()
        img = torch.from_numpy(img).cuda(non_blocking=True).float().permute((0, 3, 1, 2))
        input_data['pts_origin_xy'] = pts_origin_xy
        input_data['img'] = img

    if cfg.RPN.USE_RGB or cfg.RCNN.USE_RGB:
        pts_rgb = data['rgb']
        pts_rgb = torch.from_numpy(pts_rgb).cuda(non_blocking=True).float()
        input_data['pts_rgb'] = pts_rgb
    return input_data
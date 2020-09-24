import os
import glob
import re
import time
import numpy as np
import argparse
import torch

from lib.config import cfg
from tensorboardX import SummaryWriter

from tools.train_utils import train_utils
from tools.eval_rcnn import get_no_evaluated_ckpt
import tqdm
from lib.utils.bbox_transform import decode_bbox_target
from torch.nn import functional as F
from lib.utils.iou3d import iou3d_utils
from lib.utils import kitti_utils
from tools.eval_rcnn import save_kitti_format
from tools.kitti_object_eval_python.evaluate import evaluate as kitti_evaluate
from tools.eval_rcnn import pose_process
from tools.train_utils import common_utils
import datetime

import pickle


def paser_args():
    np.random.seed(1024)  # set the same seed

    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument('--cfg_file', type=str, default='cfgs/default.yml', help='specify the config for evaluation')
    parser.add_argument("--eval_mode", type=str, default='rpn', required=True, help="specify the evaluation mode")

    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--test', action='store_true', default=False, help='evaluate without ground truth')
    parser.add_argument("--ckpt", type=str, default=None, help="specify a checkpoint to be evaluated")
    parser.add_argument("--rpn_ckpt", type=str, default=None,
                        help="specify the checkpoint of rpn if trained separated")
    parser.add_argument("--rcnn_ckpt", type=str, default=None,
                        help="specify the checkpoint of rcnn if trained separated")

    parser.add_argument('--batch_size', type=int, default=1, help='batch size for evaluation')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument("--extra_tag", type=str, default='default', help="extra tag for multiple evaluation")
    parser.add_argument('--output_dir', type=str, default=None, help='specify an output directory if needed')
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="specify a ckpt directory to be evaluated if needed")

    parser.add_argument('--save_result', action='store_true', default=False, help='save evaluation results to files')
    parser.add_argument('--save_rpn_feature', action='store_true', default=False,
                        help='save features for separately rcnn training and evaluation')

    parser.add_argument('--random_select', action='store_true', default=True,
                        help='sample to the same number of points')
    parser.add_argument('--start_epoch', default=0, type=int, help='ignore the checkpoint smaller than this epoch')
    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument("--rcnn_eval_roi_dir", type=str, default=None,
                        help='specify the saved rois for rcnn evaluation when using rcnn_offline mode')
    parser.add_argument("--rcnn_eval_feature_dir", type=str, default=None,
                        help='specify the saved features for rcnn evaluation when using rcnn_offline mode')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--model_type', type=str, default='base', help='model type')

    args = parser.parse_args()
    return args


def eval_one_epoch_dist(cfg, model, dataloader, epoch_id, logger, args,
                        result_dir=None):
    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    mode = 'TEST' if args.test else 'EVAL'
    num_gpus = torch.cuda.device_count()
    local_rank = cfg.LOCAL_RANK % num_gpus
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False
    )
    final_output_dir = os.path.join(result_dir, 'final_result', 'data')
    if local_rank == 0:
        os.makedirs(final_output_dir, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    # dataset = dataloader.dataset
    # class_names = dataset.class_names
    det_annos = []

    roi_output_dir = os.path.join(result_dir, 'roi_result', 'data')
    refine_output_dir = os.path.join(result_dir, 'refine_result', 'data')
    rpn_output_dir = os.path.join(result_dir, 'rpn_result', 'data')
    if args.local_rank == 0:
        os.makedirs(rpn_output_dir, exist_ok=True)
        os.makedirs(roi_output_dir, exist_ok=True)
        os.makedirs(refine_output_dir, exist_ok=True)

    model.eval()

    # Local variants.
    thresh_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    total_recalled_bbox_list, total_gt_bbox = [0] * 5, 0
    total_roi_recalled_bbox_list = [0] * 5
    cnt = max_num = rpn_iou_avg = 0
    cnt = final_total = total_cls_acc = total_cls_acc_refined = total_rpn_iou = 0
    MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        input_data_dict = train_utils.load_joint_data_to_gpu(batch_dict, cfg)
        sample_id = batch_dict['sample_id']
        batch_size = len(sample_id)
        with torch.no_grad():
            ret_dict = model(input_data_dict)
            final_total, total_gt_bbox, total_rpn_iou, disp_dict = pose_process(MEAN_SIZE, args, batch_size, batch_dict,
                                                                                final_output_dir, final_total, mode,
                                                                                refine_output_dir, ret_dict,
                                                                                roi_output_dir,
                                                                                rpn_output_dir, sample_id, thresh_list,
                                                                                total_gt_bbox,
                                                                                total_recalled_bbox_list,
                                                                                total_roi_recalled_bbox_list,
                                                                                total_rpn_iou)

        if cfg.LOCAL_RANK == 0:
            # Only show the disp_dict on rank 0.
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()


    rank, world_size = common_utils.get_dist_info()
    final_total, total_gt_bbox, total_rpn_iou = common_utils.merge_results_dist(
        [final_total, total_gt_bbox, total_rpn_iou], world_size, tmpdir='/dev/shm/tempdir/')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    split_file = os.path.join(dataloader.dataset.imageset_dir, '..', '..', 'ImageSets',
                              dataloader.dataset.split + '.txt')
    split_file = os.path.abspath(split_file)
    image_idx_list = [x.strip() for x in open(split_file).readlines()]
    empty_cnt = 0
    for k in range(image_idx_list.__len__()):
        cur_file = os.path.join(final_output_dir, '%s.txt' % image_idx_list[k])
        if not os.path.exists(cur_file):
            with open(cur_file, 'w') as temp_f:
                pass
            empty_cnt += 1
            logger.info('empty_cnt=%d: dump empty file %s' % (empty_cnt, cur_file))

    ret_dict = {'empty_cnt': empty_cnt}

    logger.info('-------------------performance of epoch %s---------------------' % epoch_id)
    logger.info(str(datetime.now()))

    avg_rpn_iou = (total_rpn_iou / max(cnt, 1.0))
    avg_cls_acc = (total_cls_acc / max(cnt, 1.0))
    avg_cls_acc_refined = (total_cls_acc_refined / max(cnt, 1.0))
    avg_det_num = (final_total / max(len(image_idx_list), 1.0))
    logger.info('final average detections: %.3f' % avg_det_num)
    logger.info('final average rpn_iou refined: %.3f' % avg_rpn_iou)
    logger.info('final average cls acc: %.3f' % avg_cls_acc)
    logger.info('final average cls acc refined: %.3f' % avg_cls_acc_refined)
    ret_dict['rpn_iou'] = avg_rpn_iou
    ret_dict['rcnn_cls_acc'] = avg_cls_acc
    ret_dict['rcnn_cls_acc_refined'] = avg_cls_acc_refined
    ret_dict['rcnn_avg_num'] = avg_det_num

    for idx, thresh in enumerate(thresh_list):
        cur_roi_recall = total_roi_recalled_bbox_list[idx] / max(total_gt_bbox, 1.0)
        logger.info('total roi bbox recall(thresh=%.3f): %d / %d = %f' % (thresh, total_roi_recalled_bbox_list[idx],
                                                                          total_gt_bbox, cur_roi_recall))
        ret_dict['rpn_recall(thresh=%.2f)' % thresh] = cur_roi_recall

    for idx, thresh in enumerate(thresh_list):
        cur_recall = total_recalled_bbox_list[idx] / max(total_gt_bbox, 1.0)
        logger.info('total bbox recall(thresh=%.3f): %d / %d = %f' % (thresh, total_recalled_bbox_list[idx],
                                                                      total_gt_bbox, cur_recall))
        ret_dict['rcnn_recall(thresh=%.2f)' % thresh] = cur_recall

    if cfg.TEST.SPLIT != 'test':
        logger.info('Averate Precision:')
        name_to_class = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
        ap_result_str, ap_dict = kitti_evaluate(dataloader.dataset.label_dir, final_output_dir, label_split_file=split_file,
                                                current_class=name_to_class[cfg.CLASSES])
        logger.info(ap_result_str)
        ret_dict.update(ap_dict)

    logger.info('result is saved to: %s' % result_dir)
    return ret_dict

    logger.info('****************Evaluation done.*****************')
    return ret_dict


def repeat_eval_ckpt(model, test_loader, args, root_result_dir, logger, ckpt_dir):
    # evaluated ckpt record
    root_result_dir = os.path.join(root_result_dir, 'eval', 'eval_all_' + args.extra_tag)
    os.makedirs(root_result_dir, exist_ok=True)
    ckpt_record_file = os.path.join(root_result_dir, 'eval_list_%s.txt' % cfg.TEST.SPLIT)

    log_file = os.path.join(root_result_dir, 'log_eval_all_%s.txt' % cfg.TEST.SPLIT)

    # tensorboard log
    if cfg.LOCAL_RANK == 0:
        tb_log = SummaryWriter(
            log_dir=os.path.join(root_result_dir, ('tensorboard_%s' % cfg.DATA_CONFIG.DATA_SPLIT['test'])))
    total_time = 0
    first_eval = True

    while True:
        # check whether there is checkpoint which is not evaluated
        cur_epoch_id, cur_ckpt = get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args)
        if cur_epoch_id == -1 or int(float(cur_epoch_id)) < args.start_epoch:
            wait_second = 120
            if cfg.LOCAL_RANK == 0:
                print('Wait %s seconds for next check (progress: %.1f / %d minutes): %s \r'
                      % (wait_second, total_time * 1.0 / 60, args.max_waiting_mins, ckpt_dir), end='', flush=True)
            time.sleep(wait_second)
            total_time += 120
            if total_time > args.max_waiting_mins * 60 and (first_eval is False):
                break
            continue

        total_time = 0
        first_eval = False

        train_utils.load_checkpoint(model, filename=cur_ckpt)
        model.cuda()

        # start evaluation
        cur_result_dir = os.path.join(root_result_dir, ('epoch_%s' % cur_epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test'])
        tb_dict = eval_one_epoch_dist(
            cfg, model, test_loader, cur_epoch_id, logger,
            result_dir=cur_result_dir
        )

        if cfg.LOCAL_RANK == 0:
            for key, val in tb_dict.items():
                tb_log.add_scalar(key, val, cur_epoch_id)

        # record this epoch which has been evaluated
        with open(ckpt_record_file, 'a') as f:
            print('%s' % cur_epoch_id, file=f)
        logger.info('Epoch %s has been evaluated' % cur_epoch_id)

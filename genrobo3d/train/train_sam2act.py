import os
import sys
import json
import argparse
import time
from collections import defaultdict
from tqdm import tqdm
import copy
from functools import partial
import sam2act.mvt.mvt_sam2 as mvt_sam2
import warnings
from torch.nn.parallel import DistributedDataParallel as DDP
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

from genrobo3d.train.utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from genrobo3d.train.utils.save import ModelSaver, save_training_meta
from genrobo3d.train.utils.misc import NoOp, set_dropout, set_random_seed
from genrobo3d.train.utils.distributed import set_cuda, wrap_model, all_gather,get_local_rank
from genrobo3d.train.utils.rvt_utils import get_model_size,load_cfgs,save_agent,load_agent

from genrobo3d.train.optim import get_lr_sched, get_lr_sched_decay_rate
from genrobo3d.train.optim.misc import build_optimizer

from genrobo3d.configs.default import get_config

from genrobo3d.train.datasets.loader import build_dataloader
from genrobo3d.train.datasets.motion_planner_dataset import (
    MotionPlannerDataset, base_collate_fn_partial, ptv3_collate_fn_partial
)

# from genrobo3d.models.pct_motion_planner import PCTMotionPlanner
from genrobo3d.models.motion_planner_ptv3 import (
    MotionPlannerPTV3AdaNorm, MotionPlannerPTV3CA
)
from genrobo3d.models.sam2act_agent import SAM2Act_Agent2 as SAM2Act_Agent

from sam2act.utils.peract_utils import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
    DATA_FOLDER,
    DATA_FOLDER_MEM,
)

DATASET_FACTORY = {
    'sam2act': (MotionPlannerDataset, ptv3_collate_fn_partial),
    'MotionPlannerPTV3AdaNorm': (MotionPlannerDataset, ptv3_collate_fn_partial),
    'MotionPlannerPTV3CA':  (MotionPlannerDataset, ptv3_collate_fn_partial)
}

MODEL_FACTORY = {
    'sam2act': SAM2Act_Agent,
    'MotionPlannerPTV3AdaNorm': MotionPlannerPTV3AdaNorm,
    'MotionPlannerPTV3CA': MotionPlannerPTV3CA,
}

def get_logdir(cmd_args, exp_cfg):
    exp = exp_cfg.exp_id + '_' + exp_cfg.exp_name
    log_dir = os.path.join(cmd_args.log_dir, exp)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def main(config,cmd_args):
    config.defrost()
    rank = get_local_rank()
    config.local_rank = rank
    default_gpu = (rank==0)
    world_size = config.world_size
    rank %= world_size
    ddp = world_size > 1
    device = f"cuda:{rank}"
    # ddp_utils.setup(rank, world_size=len(devices), port=port)

    exp_cfg,mvt_cfg = load_cfgs(cmd_args)
    log_dir = get_logdir(cmd_args, exp_cfg)
    if ddp:
        print(f"Running DDP on rank {rank}.")

    seed = config.SEED
    if config.local_rank != -1:
        seed += rank

    set_random_seed(seed)

    # load data training set
    dataset_class, dataset_collate_fn = DATASET_FACTORY[config.MODEL.model_class]
    dataset_collate_fn = partial(dataset_collate_fn, config.MODEL.action_config.max_traj_len)

    trn_dataset = dataset_class(**config.TRAIN_DATASET)
    
    LOGGER.info(f'#num_train: {len(trn_dataset)}')
    trn_dataloader, pre_epoch = build_dataloader(
        trn_dataset, dataset_collate_fn, True, config
    )
    if config.VAL_DATASET.use_val:
        val_dataset = dataset_class(**config.VAL_DATASET)
        LOGGER.info(f"#num_val: {len(val_dataset)}")
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config.TRAIN.val_batch_size, shuffle=False,
            num_workers=config.TRAIN.n_workers, pin_memory=True, collate_fn=dataset_collate_fn
        )
    else:
        val_dataloader = None

    LOGGER.info(f'#num_steps_per_epoch: {len(trn_dataloader)}')
    if config.TRAIN.num_train_steps is None:
        config.TRAIN.num_train_steps = len(trn_dataloader) * config.TRAIN.num_epochs
    else:
        # assert config.TRAIN.num_epochs is None, 'cannot set num_train_steps and num_epochs at the same time.'
        config.TRAIN.num_epochs = int(np.ceil(config.TRAIN.num_train_steps / len(trn_dataloader)))
        
    if config.TRAIN.gradient_accumulation_steps > 1:
        config.TRAIN.num_train_steps *= config.TRAIN.gradient_accumulation_steps
        config.TRAIN.num_epochs *= config.TRAIN.gradient_accumulation_steps

    # setup loggers
    if default_gpu:
        save_training_meta(config)
        # TB_LOGGER.create(os.path.join(config.output_dir, 'logs'))
        if config.tfboard_log_dir is None:
            output_dir_tokens = config.output_dir.split('/')
            config.tfboard_log_dir = os.path.join(output_dir_tokens[0], 'TFBoard', *output_dir_tokens[1:])
        TB_LOGGER.create(config.tfboard_log_dir)
        add_log_to_file(os.path.join(config.output_dir, 'logs', 'log.txt'))
    else:
        LOGGER.disabled = True

    # Prepare model
    if config.MODEL.model_class == "sam2act":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        sam2act = mvt_sam2.MVT_SAM2(
            renderer_device=device,
            rank=rank,
            **mvt_cfg,
        ).to(device)
        if rank == 0:
            get_model_size(sam2act)
        if ddp:
            sam2act = DDP(sam2act, device_ids=[device], find_unused_parameters=True)

        agent = SAM2Act_Agent(
            network=sam2act,
            image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
            add_lang=mvt_cfg.add_lang,
            stage_two=mvt_cfg.stage_two,
            rot_ver=mvt_cfg.rot_ver,
            scene_bounds=SCENE_BOUNDS,
            cameras=CAMERAS,
            log_dir=f"{log_dir}/test_run/",
            cos_dec_max_step=config.TRAIN.num_train_steps,
            **exp_cfg.peract,
            **exp_cfg.rvt,
        )
        agent.build(training=True, device=device)
    else:
        assert False, "Incorrect agent"
    # Fix parameters not needed
    # if config.TRAIN.freeze_params.encoder:
    #     for param_name, param in model.named_parameters():
    #         if param_name.startswith('mae_encoder') and 'decoder_block' not in param_name:
    #                 param.requires_grad = False

    # LOGGER.info("Model: nweights %d nparams %d" % (model.num_parameters))
    # LOGGER.info("Model: trainable nweights %d nparams %d" % (model.num_trainable_parameters))
    
    config.freeze()

    # Load from checkpoint
    model_checkpoint_file = config.checkpoint#name of loaded check point

    if model_checkpoint_file is not None and config.TRAIN.resume_training:
        epoch,steps = load_agent(model_checkpoint_file, agent, only_epoch=False)
        restart_epoch = epoch + 1
        global_step = steps
        LOGGER.info(f'Load the model checkpoint from {model_checkpoint_file} at epoch{restart_epoch}')
        #更新restart_epoch，更新global_step
    else:
        # to compute training statistics
        restart_epoch = 0
        global_step = 0
    if ddp:    
        dist.barrier()
    agent.train()

    if default_gpu:
        pbar = tqdm(initial=global_step, total=config.TRAIN.num_train_steps)
    else:
        pbar = NoOp()

    LOGGER.info(f"***** Running training with {config.world_size} GPUs *****")
    LOGGER.info("  Batch size = %d", config.TRAIN.train_batch_size if config.local_rank == -1 
                else config.TRAIN.train_batch_size * config.world_size)
    LOGGER.info("  Accumulate steps = %d", config.TRAIN.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", config.TRAIN.num_train_steps)

    # optimizer.zero_grad()
    # optimizer.step()

    running_metrics = {}

    best_val_step, best_val_metric = None, np.inf
    
    for epoch_id in range(restart_epoch, config.TRAIN.num_epochs):
        if global_step >= config.TRAIN.num_train_steps:
            break

        # In distributed mode, calling the set_epoch() method at the beginning of each epoch
        pre_epoch(epoch_id)
        
        for step, batch in enumerate(trn_dataloader):
            # forward pass
            _, losses = agent(batch, compute_loss=True, compute_final_action=False)

            # backward pass
            # if config.TRAIN.gradient_accumulation_steps > 1:  # average loss
            #     losses['total'] = losses['total'] / config.TRAIN.gradient_accumulation_steps
            # losses['total'].backward()

            for key, value in losses.items():#加载所有loss
                TB_LOGGER.add_scalar(f'step/loss_{key}', value, global_step)
                running_metrics.setdefault(f'loss_{key}', RunningMeter(f'loss_{key}'))
                running_metrics[f'loss_{key}'](value)

            # optimizer update and logging
            if (step + 1) % config.TRAIN.gradient_accumulation_steps == 0:
                global_step += 1
                # learning rate scheduling
                # lr_decay_rate = get_lr_sched_decay_rate(global_step, config.TRAIN)
                # for kp, param_group in enumerate(optimizer.param_groups):
                #     param_group['lr'] = lr_this_step = max(init_lrs[kp] * lr_decay_rate, 1e-8)
                # TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

                # log loss
                # NOTE: not gathered across GPUs for efficiency
                TB_LOGGER.step()

                # update model params
                # if config.TRAIN.grad_norm is not None:
                #     grad_norm = torch.nn.utils.clip_grad_norm_(
                #         model.parameters(), config.TRAIN.grad_norm
                #     )
                #     TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
                # optimizer.step()
                # optimizer.zero_grad()
                pbar.update(1)

            if global_step % config.TRAIN.log_steps == 0:
                # monitor training throughput
                LOGGER.info(
                    f'==============Epoch {epoch_id} Step {global_step}===============')
                LOGGER.info(', '.join(['%s:%.4f' % (lk, lv.val) for lk, lv in running_metrics.items()]))
                LOGGER.info('===============================================')                

            if global_step % config.TRAIN.save_steps == 0:
                save_agent(agent, f"{log_dir}/model_{global_step}.pth", epoch_id,global_step)
                save_agent(agent, f"{log_dir}/model_last.pth", epoch_id,global_step)

            if (val_dataloader is not None) and (global_step % config.TRAIN.val_steps == 0):
                val_metrics = validate(agent, val_dataloader)
                LOGGER.info(f'=================Validation=================')
                metric_str = ', '.join(['%s: %.4f' % (lk, lv) for lk, lv in val_metrics.items()])
                LOGGER.info(metric_str)
                LOGGER.info('===============================================')
                if val_metrics['pos_loss'] < best_val_metric:
                    best_val_metric = val_metrics['pos_loss']
                    best_val_step = global_step
                agent.train()

            if global_step >= config.TRAIN.num_train_steps:
                break

    if global_step % config.TRAIN.save_steps != 0:
        LOGGER.info(
            f'==============Epoch {epoch_id} Step {global_step}===============')
        LOGGER.info(', '.join(['%s:%.4f' % (lk, lv.val) for lk, lv in running_metrics.items()]))
        LOGGER.info('===============================================')
        save_agent(agent, f"{log_dir}/model_{global_step}.pth", epoch_id,global_step)
        save_agent(agent, f"{log_dir}/model_last.pth", epoch_id,global_step)

        val_metrics = validate(agent, val_dataloader)
        LOGGER.info(f'=================Validation=================')
        metric_str = ', '.join(['%s: %.4f' % (lk, lv) for lk, lv in val_metrics.items()])
        LOGGER.info(metric_str)
        LOGGER.info('===============================================')
        if val_metrics['pos_loss'] < best_val_metric:
            best_val_metric = val_metrics['pos_loss']
            best_val_step = global_step

    LOGGER.info(
        f'Validation: Best loss: {best_val_metric:.4f} at step {best_val_step}'
    )

@torch.no_grad()
def validate(model, val_dataloader):
    model.eval()
    pos_loss, rot_loss, open_loss, stop_loss, total_loss, num_examples, num_batches = 0, 0, 0, 0, 0, 0, 0
    open_acc, stop_acc = 0, 0
    for batch in val_dataloader:
        pred_action, loss = model(batch, compute_loss=True)
        pred_action = pred_action.cpu()
        pred_open = torch.sigmoid(pred_action[..., -2]) > 0.5
        open_acc += (pred_open == batch['gt_trajs'][..., -1][:,0,:].cpu()).float().sum().item()
        pred_stop = torch.sigmoid(pred_action[..., -1]) > 0.5
        stop_acc += (pred_stop == batch['gt_trajs_stop'][:,0,:].cpu()).float().sum().item()
        pos_loss += loss['pos']
        rot_loss += loss['rot']
        open_loss += loss['open']
        stop_loss += loss['stop']
        total_loss += loss['total']
        num_examples += pred_action.size(0) * pred_action.size(1)
        num_batches += 1
        
    return {
        'total_loss': total_loss / num_batches, 
        'pos_loss': pos_loss / num_batches,
        'rot_loss': rot_loss / num_batches,
        'open_loss': open_loss / num_batches,
        'stop_loss': stop_loss / num_batches,
        'open_acc': open_acc / num_examples, 
        'stop_acc': stop_acc / num_examples,
    }


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--mvt_cfg_path", type=str, default="")
    parser.add_argument("--exp_cfg_path", type=str, default="")

    parser.add_argument("--mvt_cfg_opts", type=str, default="")
    parser.add_argument("--exp_cfg_opts", type=str, default="")
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line (use , to separate values in a list)",
    )
    args = parser.parse_args()

    config = get_config(args.exp_config, args.opts)

    if os.path.exists(config.output_dir) and os.listdir(config.output_dir):
        LOGGER.warning(
            "Output directory ({}) already exists and is not empty.".format(
                config.output_dir
            )
        )

    return config,args


if __name__ == '__main__':
    config,args = build_args()
    main(config,args)

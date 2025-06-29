import argparse
import datetime
import logging
import math
import random
import time
import torch
from os import path as osp
import os

from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse

import numpy as np

def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Whether to use wandb for logging. If set, it will override the logger settings in the config file.",
        default=False)

    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)
    opt['wandb'] = True

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    if args.wandb:
        opt["wandb"] = True
        if 'logger' not in opt: opt['logger'] = {}
        if 'wandb' not in opt['logger']: opt['logger']['wandb'] = {}
        opt['logger']['wandb'].setdefault('project', 'tfg-restormer-experiments')
        opt['logger']['wandb'].setdefault('name', opt['name'])
        opt['logger'].setdefault('use_tb_logger', True)
    elif 'logger' in opt and 'wandb' in opt['logger'] and opt['logger']['wandb'].get('project'):
        opt["wandb"] = True

    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])
    return opt

def init_loggers(opt):
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # if opt.get("wandb", False) and opt['logger'].get('wandb') is not None and \
    #    opt['logger']['wandb'].get('project') is not None and \
    #    ('debug' not in opt['name'] or opt.get("wandb_debug", False)):
    #     assert opt['logger'].get('use_tb_logger') is True, (
    #         'should turn on tensorboard when using wandb for basicsr.utils.init_wandb_logger')
    #     init_wandb_logger(opt)
    #     logger.info(f"Wandb logger initialized by basicsr.utils.init_wandb_logger for project '{opt['logger']['wandb']['project']}'.")

    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join('tb_logger', opt['name']))
    return logger, tb_logger

def create_train_val_dataloader(opt, logger):
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'],
                                            opt['rank'], dataset_enlarge_ratio)
            train_loader = create_dataloader(
                train_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'],
                sampler=train_sampler, seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'],
                sampler=None, seed=opt['manual_seed'])
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')
    return train_loader, train_sampler, val_loader, total_epochs, total_iters

def main():
    opt = parse_options(is_train=True)

    state_folder_path = 'experiments/{}/training_states/'.format(opt['name'])
    try:
        states = os.listdir(state_folder_path)
        states = [s for s in states if s.endswith('.state')] # Filter
    except:
        states = []

    resume_state = None
    if len(states) > 0:
        max_state_file = '{}.state'.format(max([int(x[0:-6]) for x in states]))
        resume_state = os.path.join(state_folder_path, max_state_file)
        opt['path']['resume_state'] = resume_state

    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join('tb_logger', opt['name']))

    logger, tb_logger = init_loggers(opt)

    if opt.get("wandb", False):
        import wandb
        if not wandb.run:
            wandb.init(project="tfg_restormer_experiments",name=opt["name"] , config=opt)

    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result

    if resume_state:
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        model.resume_training(resume_state)
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        model = create_model(opt)
        start_epoch = 0
        current_iter = 0

    msg_logger = MessageLogger(opt, current_iter, tb_logger)
    val_iter_freq = 1000
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, "cuda", "cpu".')

    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    iters = opt['datasets']['train'].get('iters')
    batch_size = opt['datasets']['train'].get('batch_size_per_gpu')
    mini_batch_sizes = opt['datasets']['train'].get('mini_batch_sizes')
    gt_size = opt['datasets']['train'].get('gt_size')
    mini_gt_sizes = opt['datasets']['train'].get('gt_sizes')

    groups = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])
    logger_j = [True] * len(groups)
    scale = opt.get('scale', 1)

    epoch = start_epoch
    while current_iter <= total_iters:
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        accumulated_train_loss_for_epoch = 0.0
        iterations_this_epoch = 0

        while train_data is not None:
            data_time = time.time() - data_time
            current_iter += 1
            if current_iter > total_iters:
                break
            
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            
            j = ((current_iter > groups) != True).nonzero()[0]
            bs_j = len(groups) - 1 if len(j) == 0 else j[0]
            mini_gt_size = mini_gt_sizes[bs_j]
            mini_batch_size = mini_batch_sizes[bs_j]
            
            if logger_j[bs_j]:
                logger.info(f'\n Updating Patch_Size to {mini_gt_size} and Batch_Size to {mini_batch_size * torch.cuda.device_count()} \n')
                logger_j[bs_j] = False

            lq, gt = train_data['lq'], train_data['gt']
            if mini_batch_size < batch_size:
                indices = random.sample(range(0, batch_size), k=mini_batch_size)
                lq, gt = lq[indices], gt[indices]
            if mini_gt_size < gt_size:
                x0 = int((gt_size - mini_gt_size) * random.random())
                y0 = int((gt_size - mini_gt_size) * random.random())
                x1, y1 = x0 + mini_gt_size, y0 + mini_gt_size
                lq = lq[:, :, x0:x1, y0:y1]
                current_scale = int(scale)
                gt = gt[:, :, x0 * current_scale:x1 * current_scale, y0 * current_scale:y1 * current_scale]
            
            model.feed_train_data({'lq': lq, 'gt': gt})
            model.optimize_parameters(current_iter)
            iter_time = time.time() - iter_time
            
            current_log_dict = model.get_current_log()
            if 'l_pix' in current_log_dict:
                loss_value = current_log_dict['l_pix']
                batch_size_used = lq.size(0)
                accumulated_train_loss_for_epoch += current_log_dict['l_pix']
                iterations_this_epoch += batch_size_used

            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter,
                            'lrs': model.get_current_learning_rate(),
                            'time': iter_time, 'data_time': data_time}
                log_vars.update(current_log_dict)
                msg_logger(log_vars)
                if opt.get("wandb", False) and wandb.run:
                    wandb_iter_payload = {
                        'iter': current_iter, 'epoch_context': epoch,
                        'learning_rate': model.get_current_learning_rate()[0] if model.get_current_learning_rate() else 0,
                        'time_per_iter': iter_time, 'data_time': data_time,
                    }
                    for k_log, v_log in current_log_dict.items():
                        wandb_iter_payload[f'train_{k_log}_iter'] = v_log
                    wandb.log(wandb_iter_payload, )
            
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # !!! REMOVED VALIDATION TRIGGER FROM HERE !!!
            # if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
            #    ...
            
            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
        # End of inner loop (train_data is None)

        # --- VALIDATION AT THE END OF EACH EPOCH ---
            if opt.get('val') is not None and val_loader is not None and \
               current_iter > 0 and current_iter % val_iter_freq == 0:
                logger.info(f"--- Running validation at iteration {current_iter} (Epoch {epoch}) ---")
                rgb2bgr = opt['val'].get('rgb2bgr', True)
                use_image = opt['val'].get('use_image', True)
                
                validation_results = model.validation(val_loader, current_iter, tb_logger,
                                            opt['val']['save_img'], rgb2bgr, use_image)
                
                if opt.get("wandb", False) and wandb.run and validation_results:
                    wandb_val_payload = {
                        # 'iter_for_val': current_iter, # Step is current_iter
                        'epoch_for_val_context': epoch,       
                    }
                    for metric_name, metric_value in validation_results.items():
                        wandb_val_payload[f'validation_{metric_name}'] = metric_value
                    
                    wandb.log(wandb_val_payload, step=current_iter) 
                    logger.info(f"Logged validation results to wandb at iteration {current_iter}.")
        # --- END OF EPOCH VALIDATION ---

        if opt.get("wandb", False) and wandb.run and iterations_this_epoch > 0:
            avg_train_loss_this_epoch = accumulated_train_loss_for_epoch / iterations_this_epoch
            wandb.log({
                'epoch': epoch, 
                'avg_train_loss_epoch': avg_train_loss_this_epoch,
            }, )
            logger.info(f"Epoch {epoch} completed. Avg train loss: {avg_train_loss_this_epoch:.4f}. Logged to wandb (step {current_iter}).")
        
        epoch += 1
    # End of outer loop

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)
    
    if opt.get('val') is not None and val_loader is not None: # Final validation
        logger.info("--- Running final validation ---")
        final_val_results = model.validation(val_loader, current_iter, tb_logger,
                                             opt['val']['save_img'], 
                                             opt['val'].get('rgb2bgr', True),
                                             opt['val'].get('use_image', True))
        if opt.get("wandb", False) and wandb.run and final_val_results:
            wandb_final_val_payload = {
                'iter_for_val': current_iter, 
                'epoch_for_val': epoch -1 if epoch > start_epoch else start_epoch, 
            }
            for metric_name, metric_value in final_val_results.items():
                wandb_final_val_payload[f'final_validation_{metric_name}'] = metric_value
            wandb.log(wandb_final_val_payload, )
            logger.info("Logged final validation results to wandb.")

    if tb_logger: tb_logger.close()
    if opt.get("wandb", False) and wandb.run: wandb.finish()

if __name__ == '__main__':
    main()
import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm # Ensure tqdm is imported if you plan to use pbar

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')
import wandb
import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device
        self.use_identity = use_identity
        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1,1)).item()
        r_index = torch.randperm(target.size(0)).to(self.device)
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_

class ImageCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)

        # define network
        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train: # This is True when instantiated from train.py
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            # If not in training mode, cri_pix might not be defined.
            # For validation loss calculation during training, this should be fine.
            # If this model is used purely for inference without 'train' in opt, this will be an issue.
            self.cri_pix = None 
            logger = get_root_logger()
            logger.warning('Pixel loss (cri_pix) is not defined. Validation loss cannot be computed.')
            # raise ValueError('pixel loss are None.') # Soften this for potential inference-only paths

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag: # self.mixing_flag is set in __init__
            if hasattr(self, 'mixing_augmentation') and self.gt is not None:
                 self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)
            elif not hasattr(self, 'mixing_augmentation'):
                 logger = get_root_logger()
                 logger.warning("Mixing flag is true, but mixing_augmentation is not initialized.")


    def feed_data(self, data): # For validation/testing
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device) # Keep GT on device for loss calculation

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        preds = self.net_g(self.lq)
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1] # self.output will be on device

        loss_dict = OrderedDict()
        # pixel loss
        l_pix = 0.
        if self.cri_pix is not None and hasattr(self, 'gt') and self.gt is not None:
            for pred in preds:
                # provar a restar gt lq i multiplicar ponderat
                l_pix += self.cri_pix(pred, self.gt, self.lq)
            loss_dict['l_pix'] = l_pix
            l_pix.backward()
            if self.opt['train'].get('use_grad_clip', False): # Check key existence
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
            self.optimizer_g.step()
        else:
            logger = get_root_logger()
            logger.error("cri_pix or gt not available for loss calculation during optimization.")
            # Fallback or raise error:
            loss_dict['l_pix'] = torch.tensor(0.0).to(self.device) # Dummy loss if not calculable


        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):        
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img) # self.output is set here
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq      
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
        else:
            self.net_g.eval() # Set to eval mode for testing
            with torch.no_grad():
                pred = self.net_g(img)
            if self.is_train: # If called during training's validation phase, set back to train
                 self.net_g.train()

        if isinstance(pred, list):
            pred = pred[-1]
        self.output = pred # self.output is on device

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ.get('LOCAL_RANK', '0') == '0': # Check environment variable correctly
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            # For non-master ranks, return a dictionary with default values
            # consistent with what nondist_validation would return.
            metrics_dict = {}
            if self.opt['val'].get('metrics') is not None:
                for metric_key in self.opt['val']['metrics'].keys():
                    metrics_dict[metric_key] = 0.0
            # Add default for val_loss if it's expected
            if hasattr(self, 'cri_pix') and self.cri_pix is not None:
                metrics_dict['val_loss'] = 0.0
            return metrics_dict


    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        
        # self.metric_results will store per-metric accumulated values
        self.metric_results = {} # Reset for each validation run
        if with_metrics:
            for metric_key in self.opt['val']['metrics'].keys():
                self.metric_results[metric_key] = 0.0
        
        total_val_loss = 0.
        total_val_samples = 0
        # Check if cri_pix is available (it should be if model.is_train is true)
        can_calc_val_loss = hasattr(self, 'cri_pix') and self.cri_pix is not None

        # pbar = tqdm(total=len(dataloader), unit='image') # Uncomment for progress bar

        window_size = self.opt['val'].get('window_size', 0)
        if window_size:
            test_fn = partial(self.pad_test, window_size)
        else:
            test_fn = self.nonpad_test

        cnt = 0
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data) # self.lq and self.gt are set to device
            test_fn()                # self.output is set to device

            # Calculate validation loss using self.output and self.gt (both on device)
            if can_calc_val_loss and hasattr(self, 'gt') and self.gt is not None:
                with torch.no_grad():
                    # self.gt is already on self.output.device due to feed_data
                    val_l_pix_iter = self.cri_pix(self.output, self.gt, self.lq)
                    batch_size = self.lq.size(0) # Get batch size from lq
                    total_val_loss += val_l_pix_iter.item() * batch_size
                    total_val_samples += batch_size

            # Visuals are for saving and CPU-based metrics; tensors moved to CPU
            visuals = self.get_current_visuals() 
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr) # result is on CPU
            gt_img = None
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr) # gt is on CPU
            
            lq_img = tensor2img([visuals['lq']], rgb2bgr=rgb2bgr) 
            wandb.log({
                "SR Image": wandb.Image(sr_img),
                "GT Image": wandb.Image(gt_img),
                "LQ Image": wandb.Image(lq_img)
            })

            # Tentative for out of GPU memory - original self.gt and self.output (GPU tensors) can be cleared
            # self.gt was moved to visuals['gt'] (CPU), self.output to visuals['result'] (CPU)
            # The original GPU tensors self.lq, self.gt (if it existed), self.output are overwritten or cleared by feed_data/test_fn in next iteration
            # Explicit deletion here helps if loop body is very memory intensive before next feed_data
            if hasattr(self, 'gt'): del self.gt # Delete the GPU tensor version
            del self.lq
            del self.output # Delete the GPU tensor version
            torch.cuda.empty_cache()


            if save_img:
                if self.opt['is_train']:
                    save_img_dir = osp.join(self.opt['path']['visualization'], img_name)
                    os.makedirs(save_img_dir, exist_ok=True)
                    save_img_path = osp.join(save_img_dir, f'{img_name}_{current_iter}.png')
                    if gt_img is not None:
                         save_gt_img_path = osp.join(save_img_dir, f'{img_name}_{current_iter}_gt.png')
                else:
                    save_img_dir = osp.join(self.opt['path']['visualization'], dataset_name)
                    os.makedirs(save_img_dir, exist_ok=True)
                    save_img_path = osp.join(save_img_dir, f'{img_name}.png')
                    if gt_img is not None:
                        save_gt_img_path = osp.join(save_img_dir, f'{img_name}_gt.png')
                
                imwrite(sr_img, save_img_path)
                if gt_img is not None:
                    imwrite(gt_img, save_gt_img_path)

            if with_metrics and gt_img is not None: # Metrics require GT
                opt_metric = deepcopy(self.opt['val']['metrics'])
                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    if use_image: # Use CPU images (np arrays)
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, lq_img, **opt_)
                    else: # Use CPU tensors from visuals
                        # Ensure metric function can handle CPU tensors or move them if needed
                        # visuals['result'] and visuals['gt'] are already on CPU.
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)
            cnt += 1
        # pbar.close() # if using tqdm

        # Prepare dictionary for returning all results
        # This will be logged by train.py to wandb
        final_metrics_dict = {}

        if cnt > 0: # Avoid division by zero
            # if can_calc_val_loss:
            #     avg_val_loss = total_val_loss / cnt
            #     final_metrics_dict['val_loss'] = avg_val_loss
            #     # self.metric_results is used by _log_validation_metric_values for tb_logger and console
            #     self.metric_results['val_loss'] = avg_val_loss 

            if with_metrics:
                for metric_key in self.metric_results.keys():
                    if metric_key != 'val_loss': # val_loss already averaged if calculated
                        self.metric_results[metric_key] /= cnt
                    final_metrics_dict[metric_key] = self.metric_results[metric_key]
        else: # No validation samples processed
            # if can_calc_val_loss:
            #     final_metrics_dict['val_loss'] = 0.0
            #     self.metric_results['val_loss'] = 0.0
            if with_metrics:
                for metric_key in self.opt['val']['metrics'].keys():
                     if metric_key not in final_metrics_dict: # ensure all expected metrics are present
                        final_metrics_dict[metric_key] = 0.0
                        self.metric_results[metric_key] = 0.0

        if total_val_samples > 0:
            avg_val_loss = total_val_loss / total_val_samples
        final_metrics_dict["val_loss"] = avg_val_loss
        self.metric_results['val_loss'] = avg_val_loss # Store in metric_results for logging

        # Log all values in self.metric_results (which now includes val_loss)
        # to console and tb_logger
        self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        
        # Return the dictionary for train.py to use for wandb logging
        return final_metrics_dict


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        # self.metric_results should contain all computed metrics including val_loss by now
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                # For tb_logger, it might be good to prefix with dataset_name if multiple val sets exist
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)


    def get_current_visuals(self):
        out_dict = OrderedDict()
        if hasattr(self, 'lq') and self.lq is not None: # lq might have been deleted
            out_dict['lq'] = self.lq.detach().cpu()
        if hasattr(self, 'output') and self.output is not None: # output might have been deleted
            out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt') and self.gt is not None: # gt might have been deleted or not exist
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema') and self.ema_decay > 0: # Check if net_g_ema exists
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
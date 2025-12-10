import os
import sys
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
import torchvision
import numpy as np
from itertools import cycle
from utils.utils import init_distributed_mode, epoch_saving, best_saving, AverageMeter, reduce_tensor, accuracy
from utils.logger import setup_logger
import clip

from pathlib import Path
import yaml
import pprint
from dotmap import DotMap

import datetime
import shutil
from contextlib import suppress


from datasets import Video_dataset
from modules.video_clip import video_header, ViFiCLIP
from utils.NCELoss import NCELoss, DualLoss
from utils.Augmentation import get_augmentation
from utils.solver import _optimizer, _lr_scheduler
from modules.text_prompt import text_prompt
import torch.nn.functional as F


def update_dict(dict):
    new_dict = {}
    for k, v in dict.items():
        new_dict[k.replace('module.', '')] = v
    return new_dict

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', type=str, default='clip.yaml', help='global config file')
    parser.add_argument('--log_time', default='001')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')                        
    parser.add_argument("--local-rank", type=int,
                        help='local rank for DistributedDataParallel')
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precition."
    )                        
    args = parser.parse_args()
    return args



def main(args):
    global best_prec1
    """ Training Program """
    init_distributed_mode(args)
    if args.distributed:
        print('[INFO] turn on distributed train', flush=True)
    else:
        print('[INFO] turn off distributed train', flush=True)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    working_dir = os.path.join('./exp_nce', config['data']['dataset'], config['network']['arch'] , args.log_time)


    if dist.get_rank() == 0:
        Path(working_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy(args.config, working_dir)
        shutil.copy('train_nce.py', working_dir)
        shutil.copy('clip/model.py', working_dir)
        shutil.copy('modules/video_clip.py', working_dir)

    # build logger, print env and config
    logger = setup_logger(output=working_dir,
                          distributed_rank=dist.get_rank(),
                          name=f'VideoClip')
    logger.info("------------------------------------")
    logger.info("Environment Versions:")
    logger.info("- Python: {}".format(sys.version))
    logger.info("- PyTorch: {}".format(torch.__version__))
    logger.info("- TorchVison: {}".format(torchvision.__version__))
    logger.info("------------------------------------")
    pp = pprint.PrettyPrinter(indent=4)
    logger.info(pp.pformat(config))
    logger.info("------------------------------------")
    logger.info("storing name: {}".format(working_dir))



    config = DotMap(config)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        cudnn.benchmark = True

    # fix the seed for reproducibility
    seed = config.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    design_details = {"trainer": config.prompt.trainer_type,  
                       "vision_depth": config.prompt.vision_depth,     
                        "vision_ctx": config.prompt.vision_ctx,          
                        "language_depth": config.prompt.language_depth,     
                        "language_ctx": config.prompt.language_ctx}
    
    # get fp16 model and weight
    model, clip_state_dict = clip.load(
        config.network.arch,
        design_details,
        device='cpu',jit=False,
        internal_modeling=config.network.tm,
        T=config.data.num_segments,
        dropout=config.network.drop_out,
        emb_dropout=config.network.emb_dropout,
        pretrain=config.network.init,
        joint_st = config.network.joint_st) # Must set jit=False for training  ViT-B/32
 
    # no learning prompt used
    design_details_vanilla = {"trainer": 'None',  
                       "vision_depth": 0,     
                        "vision_ctx": 0,          
                        "language_depth": 0,     
                        "language_ctx": 0}
    
    # get fp16 model and weight
    model_vanilla, _ = clip.load(
        config.network.arch_vanilla,
        design_details_vanilla,
        device='cpu',jit=False,
        internal_modeling=config.network.tm,
        T=config.data.num_segments,
        dropout=config.network.drop_out,
        emb_dropout=config.network.emb_dropout,
        pretrain=config.network.init,
        joint_st = config.network.joint_st) # Must set jit=False for training  ViT-B/32
   
    transform_train = get_augmentation(True, config)
    transform_val = get_augmentation(False, config)
   
    logger.info('train transforms: {}'.format(transform_train.transforms))
    logger.info('val transforms: {}'.format(transform_val.transforms))


    video_head = video_header(
        config.network.sim_header,
        config.network.interaction,
        model,
        clip_state_dict)

 
    if args.precision == "amp" or args.precision == "fp32":
        model = model.float()
        model_vanilla = model_vanilla.float()

    train_data = Video_dataset(
        config.data.train_root, config.data.train_list,
        config.data.label_list, num_segments=config.data.num_segments,
        modality=config.data.modality,
        image_tmpl=config.data.image_tmpl, random_shift=config.data.random_shift,
        transform=transform_train, dense_sample=config.data.dense)
  
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)                       
    train_loader = DataLoader(train_data,
        batch_size=config.data.batch_size, num_workers=config.data.workers,
        sampler=train_sampler, drop_last=True)

    val_data = Video_dataset(
        config.data.val_root, config.data.val_list, config.data.label_list,
        random_shift=False, num_segments=config.data.num_segments,
        modality=config.data.modality,
        image_tmpl=config.data.image_tmpl,
        transform=transform_val, dense_sample=config.data.dense, val=True)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    val_loader = DataLoader(val_data,
        batch_size=config.data.batch_size,num_workers=config.data.workers,
        sampler=val_sampler, drop_last=False)

    if config.network.gaa:
        transform_train_ng = get_augmentation(True, config, gaa=True)
 
        gaa_data = Video_dataset(
            config.prompt.gaa_train_root, config.prompt.gaa_train_list,
            config.prompt.gaa_label, num_segments=config.data.num_segments,
            modality=config.data.modality,
            image_tmpl=config.data.image_tmpl, random_shift=config.data.random_shift,
            transform=transform_train_ng, dense_sample=config.data.dense)
        # The default batch_size is 1. If it is greater than 1, you need to use .mean() for dimension reduction in forward
        gaa_sampler = torch.utils.data.distributed.DistributedSampler(gaa_data)
        gaa_loader = DataLoader(
            gaa_data,
            batch_size=1, 
            num_workers=config.data.workers,
            sampler=gaa_sampler,
            drop_last=True
        )
    else:
        gaa_loader = None
    loss_type = config.solver.loss_type
    if loss_type == 'CE':
        criterion = torch.nn.CrossEntropyLoss()
    elif loss_type == 'NCE':
        criterion = NCELoss()
    elif loss_type == 'DS':
        criterion = DualLoss()
    else:
        raise NotImplementedError

    class_names = [class_name for i, class_name in train_data.classes]
    model = ViFiCLIP(config, model, class_names, design_details)

    start_epoch = config.solver.start_epoch
    
    if config.pretrain:
        if os.path.isfile(config.pretrain):
            logger.info("=> loading checkpoint '{}'".format(config.pretrain))
            checkpoint = torch.load(config.pretrain, map_location='cpu')

            new_state_dict = {}
            for key, value in checkpoint['model_state_dict'].items():
            # Skip these keys (your original filtering logic)
                if any(k in key.lower() for k in ["prompt_learner.complete_text_embeddings", 
                                                "token_prefix", 
                                                "token_suffix"]):
                    continue          
                # Handle visual branch (image encoder)
                if key.startswith('visual.'):
                    new_key = key.replace('visual.', 'image_encoder.')
                    new_state_dict[new_key] = value

                # Handle text branch (transformer blocks)
                elif key.startswith('transformer.'):
                    new_key = 'text_encoder.' + key
                    new_state_dict[new_key] = value
                
                # Handle other text-related keys
                elif any(k in key for k in ['token_embedding', 'positional_embedding', 'ln_final', 'text_projection']):
                    new_key = 'text_encoder.' + key
                    new_state_dict[new_key] = value
                
                # Keep other keys unchanged
                else:
                    new_state_dict[key] = value      

            model.load_state_dict(new_state_dict, strict=False)
            video_head.load_state_dict(checkpoint['fusion_model_state_dict'], strict=False)
            del checkpoint
        else:
            logger.info("=> no checkpoint found at '{}'".format(config.resume))
    
    if config.resume:
        if os.path.isfile(config.resume):
            logger.info("=> loading checkpoint '{}'".format(config.resume))
            checkpoint = torch.load(config.resume, map_location='cpu')
            model.load_state_dict(update_dict(checkpoint['model_state_dict']))
            video_head.load_state_dict(update_dict(checkpoint['fusion_model_state_dict']))
            start_epoch = checkpoint['epoch'] + 1
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(config.evaluate, checkpoint['epoch']))
            del checkpoint
        else:
            logger.info("=> no checkpoint found at '{}'".format(config.pretrain))

    classes, _, _ = text_prompt(gaa_data)
    _, _, text_dict = text_prompt(train_data)
    n_class = text_dict[0].size(0)
  
    if config.network.freeze == 'P':
        logger.info("Turning off gradients in both the image and the text encoder")
        for name, param in model.named_parameters():
            if "VPT" in name:
                param.requires_grad_(True)
                print(name, 'True') 
            else:
                param.requires_grad_(False)
                
    elif config.network.freeze == 'VT':
        logger.info("Turning on gradients for COMPLETE GA2-CLIP model")
        for name, param in model.named_parameters():
            param.requires_grad_(True)
            print(name, 'True') 
    
    elif config.network.freeze == 'V':
        logger.info("Turning on gradients for VISION side the GA2-CLIP model")
        for name, param in model.named_parameters():
            if "image" not in name and "logit_scale" not in name:
                param.requires_grad_(False)    
               
    elif config.network.freeze == 'T':
        logger.info("Turning on gradients for TEXT side the GA2-CLIP model")
        for name, param in model.named_parameters():
            if "image" in name:
                param.requires_grad_(False)
                print(name, 'False')
    else:
        raise NotImplementedError('freeze type must be V, T, VT or P')
    
    for name, param in video_head.named_parameters():
            param.requires_grad_(True)
            print(name, 'True') 
        
    optimizer = _optimizer(config, model, video_head)
    lr_scheduler = _lr_scheduler(config, optimizer)

    if args.distributed:
        model = DistributedDataParallel(model.cuda(), device_ids=[args.gpu], find_unused_parameters=False)
    
        if config.network.sim_header == "None":
            video_head_nomodule = video_head
        else:
            video_head = DistributedDataParallel(video_head.cuda(), device_ids=[args.gpu])
            video_head_nomodule = video_head.module
        
    scaler = GradScaler() if args.precision == "amp" else None

    best_prec1 = 0.0
    if config.solver.evaluate:
        logger.info(("===========evaluate==========="))
        prec1 = validate(
            start_epoch,
            val_loader, classes, device, 
            model, video_head, config, n_class, logger)
        return


    for epoch in range(start_epoch, config.solver.epochs):

        if args.distributed:
            train_loader.sampler.set_epoch(epoch)        
            gaa_loader.sampler.set_epoch(epoch)

        train(model, video_head, train_loader, optimizer, criterion, scaler,
         epoch, device, lr_scheduler, config, classes, logger, model_vanilla, gaa_loader)
       
        if (epoch+1) % config.logging.eval_freq == 0:  # and epoch>0
            if config.logging.skip_epoch is not None and epoch in config.logging.skip_epoch:
                continue
    
            prec1 = validate(epoch, val_loader, classes, device, model, video_head, config, n_class, logger,  model_vanilla)
         
            if dist.get_rank() == 0:
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                logger.info('Testing: {}/{}'.format(prec1,best_prec1))
                logger.info('Saving:')
                filename = "{}/last_model.pt".format(working_dir)

                epoch_saving(epoch, model.module, video_head_nomodule, optimizer, filename)
                if is_best:
                    best_saving(working_dir, epoch, model.module, video_head_nomodule, optimizer)


def train(model, video_head, train_loader, optimizer, criterion, scaler,
          epoch, device, lr_scheduler, config, classes, logger, model_vanilla, gaa_loader):
    """ train a epoch """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    video_head.train()

    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    end = time.time()
    if gaa_loader is not None:
        gaa_iter = iter(cycle(gaa_loader))
   
    for i,(images, list_id) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if config.solver.type != 'monitor':
            if (i + 1) == 1 or (i + 1) % 10 == 0:
                lr_scheduler.step(epoch + i / len(train_loader))
        if gaa_loader is not None:
            dist.barrier()
            try:       
                gaa_image, _ = next(gaa_iter)      
            except StopIteration:    
                gaa_iter = (cycle(gaa_loader))
                gaa_image, _ = next(gaa_iter)
            image_cat = torch.cat([images, gaa_image], dim=0)
            images = image_cat.view((-1,config.data.num_segments,3) + image_cat.size()[-2:])  # b+1t 3 h w
      
        b,t,c,h,w = images.size()
        images= images.view(-1,c,h,w) # omit the Image.fromarray if the images already in PIL format, change this line to images=list_image if using preprocess inside the dataset class
        texts = classes # n_cls 77
        
        model_vanilla.eval()
        with torch.no_grad():     
            text_features_vanilla, cls_features_vanilla = model_vanilla.encode_text(texts, return_token=True)

        with autocast():
          
            if config.solver.loss_type == 'CE':
              
                image_embedding, cls_embedding, text_embedding, logit_scale = model(images, list_id, return_token=True, training=False)     
                image_embedding = image_embedding.view(b,t,-1)       
                logits = video_head(image_embedding, text_embedding, cls_embedding, text_features_vanilla.to(device), cls_features_vanilla.to(device), training=True)
                loss = criterion(logit_scale * logits, list_id.to(device))          
            else:    
                raise NotImplementedError

            # loss regularization
            loss = loss / config.solver.grad_accumulation_steps
          
        if scaler is not None:
            # back propagation
            scaler.scale(loss).backward()
        
            if (i + 1) % config.solver.grad_accumulation_steps == 0:
                scaler.step(optimizer)  
                scaler.update()  
                optimizer.zero_grad()  # reset gradient
                
        else:
            # back propagation
            loss.backward()
            
            if (i + 1) % config.solver.grad_accumulation_steps == 0:
                optimizer.step()  # update param
                optimizer.zero_grad()  # reset gradient
        
        losses.update(loss.item(), logits.size(0))
      
        batch_time.update(time.time() - end)
        end = time.time()                

        cur_iter = epoch * len(train_loader) +  i
        max_iter = config.solver.epochs * len(train_loader)
        eta_sec = batch_time.avg * (max_iter - cur_iter + 1)
        eta_sec = str(datetime.timedelta(seconds=int(eta_sec)))        

        if i % config.logging.print_freq == 0:
            logger.info(('Epoch: [{0}][{1}/{2}], lr: {lr:.2e}, eta: {3}\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                             epoch, i, len(train_loader), eta_sec, batch_time=batch_time, data_time=data_time, loss=losses,
                             lr=optimizer.param_groups[-1]['lr'])))  # TODO


def validate(epoch, val_loader, classes, device, model, video_head, config, n_class, logger, model_vanilla):
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    video_head.eval()
    model_vanilla.eval()
    
    with torch.no_grad():
  
        for i, (image, class_id) in enumerate(val_loader):
          
            image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
            b,t,c,h,w = image.size()
                  
            class_id = class_id.to(device)
            image_input = image.to(device).view(-1, c, h, w)

            text_features_vanilla, cls_features_vanilla = model_vanilla.encode_text(classes, return_token=True)
            image_embedding, cls_embedding, text_embedding, logit_scale = model(image_input, class_id, return_token=True)                         
            image_embedding = image_embedding.view(b,t,-1)      
          
            logits =  video_head(image_embedding, text_embedding, cls_embedding, text_features_vanilla.to(device), cls_features_vanilla.to(device))

            similarity = logit_scale * logits.view(b, -1, n_class).softmax(dim=-1)  # [bs, n_frames, n_cls]
            similarity = similarity.mean(dim=1, keepdim=False)  # [bs, n_cls]
           
            prec = accuracy(similarity, class_id, topk=(1, 5))
            prec1 = reduce_tensor(prec[0])
            prec5 = reduce_tensor(prec[1])

            top1.update(prec1.item(), class_id.size(0))
            top5.update(prec5.item(), class_id.size(0))

            if i % config.logging.print_freq == 0:
                logger.info(
                    ('Test: [{0}/{1}]\t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                         i, len(val_loader), top1=top1, top5=top5)))
             
    logger.info(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5)))
    
    return top1.avg

if __name__ == '__main__':
    args = get_parser() 
    main(args)
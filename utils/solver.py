import torch.optim as optim

from utils.lr_scheduler import WarmupMultiStepLR, WarmupCosineAnnealingLR


def construct_DiST_optimizer(model, config):
    temporal_net_normal_params = []
    temporal_net_bias_params = []
    temporal_net_no_wd_params = []
    others = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        elif "visual" in name:
          
            if name.endswith('cls_token') or name.endswith('positional_embedding'):
                temporal_net_no_wd_params.append(p)       
                   
            else:
                if 'bias' in name or len(p.shape) == 1:
                    temporal_net_bias_params.append(p)
                else:
                    temporal_net_normal_params.append(p)
        else:
            others.append(p)

    optim_params = []

    if len(temporal_net_no_wd_params) > 0:
        optim_params.append({"params": temporal_net_no_wd_params, 'weight_decay': 0, "lr": 10.0 * config.solver.lr})
       
    if len(temporal_net_normal_params) > 0:
        optim_params.append({"params": temporal_net_normal_params , "lr": 10.0 * config.solver.lr})
        optim_params.append({"params": temporal_net_bias_params, 'weight_decay': 0, "lr": 10.0 * config.solver.lr})
    if len(temporal_net_normal_params) > 0:
        optim_params.append({"params": others , "lr": config.solver.lr})

    return optim_params

def _optimizer(config, model, video_head):
    if config.solver.optim == 'adam':
        optimizer = optim.Adam([{'params': model.parameters()},  
         {'params': video_head.parameters(), 'lr': config.solver.lr}],
                               lr=config.solver.lr * config.solver.clip_ratio, betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=0.2)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
        print('Adam')
    elif config.solver.optim == 'sgd':

        optimizer = optim.SGD([{'params': model.parameters()},  
         {'params': video_head.parameters(), 'lr': config.solver.lr}],
                              config.solver.lr * config.solver.clip_ratio,
                              momentum=config.solver.momentum,
                              weight_decay=config.solver.weight_decay)
        print('SGD')
    elif config.solver.optim == 'adamw':
        vision_params = []
        text_params = []
        for name, param in model.named_parameters():
            if 'visual.' in name:
                vision_params.append(param)
            else:
                text_params.append(param)        

        # print('[INFO] number of visual parameters:', len(vision_params), flush=True)
        # print('[INFO] number of textual parameters:', len(text_params), flush=True)
        # optim_params = construct_DiST_optimizer(model, config)
        # optim_params.append({
        #     'params': video_head.parameters(),
        #     'lr': config.solver.lr
        # })
        # optimizer = optim.AdamW(optim_params,
        #                         betas=(0.9, 0.999), lr=config.solver.lr, eps=1e-8,
        #                         weight_decay=config.solver.weight_decay)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
        
        optimizer = optim.AdamW([{'params': model.parameters(), 'lr': config.solver.lr * config.solver.clip_ratio},
                                 {'params': video_head.parameters(), 'lr': config.solver.lr}],
                                betas=(0.9, 0.999), lr=config.solver.lr, eps=1e-8,
                                weight_decay=config.solver.weight_decay)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
    else:
        raise ValueError('Unknown optimizer: {}'.format(config.solver.optim))
   
    return optimizer


def _lr_scheduler(config, optimizer):
    if config.solver.type == 'cosine':
        lr_scheduler = WarmupCosineAnnealingLR(
            optimizer,
            config.solver.epochs,
            warmup_epochs=config.solver.lr_warmup_step
        )
    elif config.solver.type == 'multistep':
        if isinstance(config.solver.lr_decay_step, list):
            milestones = config.solver.lr_decay_step
        elif isinstance(config.solver.lr_decay_step, int):
            milestones = [
                config.solver.lr_decay_step * (i + 1)
                for i in range(config.solver.epochs //
                               config.solver.lr_decay_step)]
        else:
            raise ValueError("error learning rate decay step: {}".format(type(config.solver.lr_decay_step)))
        lr_scheduler = WarmupMultiStepLR(
            optimizer,
            milestones,
            warmup_epochs=config.solver.lr_warmup_step
        )
    else:
        raise ValueError('Unknown lr scheduler: {}'.format(config.solver.type))
    return lr_scheduler



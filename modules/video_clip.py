import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.checkpoint import checkpoint
from clip import clip

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



class TemporalTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        self.grad_checkpointing = False

    def forward(self, x: torch.Tensor):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x)
            else:
                x = r(x)
        return x


class video_header(nn.Module):
    def __init__(self, vid_head, interaction, clip_model, clip_state_dict):
        super().__init__()
        self.vid_header = vid_head
        self.interaction = interaction   
        self.device = clip_model.dtype
        self.alpha = 1.0
        assert vid_head in ["None", "Transf"]
       
        if self.vid_header == "Transf":
            embed_dim = clip_state_dict["text_projection"].shape[1]

            context_length = clip_state_dict["positional_embedding"].shape[0]
            vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
            transformer_width = clip_state_dict["ln_final.weight"].shape[0]
            transformer_heads = transformer_width // 64

            transformer_layers = len(
                set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

            self.frame_position_embeddings = nn.Embedding(context_length, embed_dim)
       
            self.transformer = TemporalTransformer(width=embed_dim, layers=6, heads=transformer_heads)
            print('layer=6')
        

        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def agg_video_feat(self, x):
        
        b, t, c = x.size()
        x = x.contiguous()
        if self.vid_header == "None":
            pass

        elif self.vid_header == "Transf":
            x_original = x
            seq_length = t
            position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
  
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
      
            x = x + frame_position_embeddings

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = x.type(x_original.dtype) + x_original

        else:
            raise ValueError('Unknown temporal modeling header: {}'.format(self.vid_header))
        return x

    
    def get_logits(self, vid_emb, text_emb, cls_emb, v_cls_emb, v_text_emb, training):
        vid_emb = self.agg_video_feat(vid_emb)  # b t c
     
        if self.interaction == 'DP':
            vid_emb = vid_emb.mean(dim=1, keepdim=False)  # b c
            vid_emb = vid_emb / vid_emb.norm(dim=-1, keepdim=True)
            cls_emb = cls_emb / cls_emb.norm(dim=-1, keepdim=True)
      
            logit = vid_emb @ cls_emb.t()  
           
        elif self.interaction == 'VCS':  # video concept spotting
            cls_emb = cls_emb / cls_emb.norm(dim=-1, keepdim=True)
            vid_emb = vid_emb / vid_emb.norm(dim=-1, keepdim=True)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            
            sims = torch.einsum('awd,btd->abwt', [text_emb, vid_emb])

            att_weight_v = F.softmax(sims/0.01, dim=-1) # abwt
            att_weight_v = att_weight_v.mean(dim=-2)  # abt
          
            v_att = torch.einsum('abt,btd->abd', [att_weight_v, vid_emb])
            # new loss      
            t2v_logits = torch.einsum('abd,ad->ab',[v_att, cls_emb])
        
            logit = t2v_logits.transpose(1, 0)

        elif self.interaction == 'DPS':  # video concept spotting

            vid_emb = vid_emb.mean(dim=1, keepdim=False)  # b c
      
            cls_emb = cls_emb / cls_emb.norm(dim=-1, keepdim=True)
            vid_emb = vid_emb / vid_emb.norm(dim=-1, keepdim=True)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            v_cls_emb = v_cls_emb / v_cls_emb.norm(dim=-1, keepdim=True)
      
            if training:
                # use explicit regularization
                logit_forward = vid_emb[0:-1,:] @ cls_emb.t() 
                logti_neg = vid_emb[-1:, :] @ v_cls_emb.t()
                logit = logit_forward + self.alpha * logti_neg

                return logit
            
                # use margin loss
                #base_sim = vid_vanilla @ cls_emb.t() 
                #max_base_sim = base_sim.max(dim=1).values
                #diff = max_base_sim - logti_neg + self.margin

                # softplus
                #score = F.softplus(diff).mean()
                # clamp
                #score = torch.clamp(diff, min=0).mean()

                #return logit, score
            
            else:
                logit = vid_emb @ cls_emb.t()
                return logit
       
    
    def forward(self, vid_emb, text_emb, cls_emb, v_cls_emb, v_text_emb, training=False):
       
        logits = self.get_logits(vid_emb, text_emb, cls_emb, v_cls_emb, v_text_emb, training)
      
        return logits


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, return_token=False):
      
        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
 
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
   
        text_token = x @ self.text_projection   # eg, [400 77 512]
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
     
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection # eg, [400 512]
      
        if return_token:
            return x, text_token 
        else:
            return x, None


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, design_details):
        super().__init__()
        dtype = clip_model.dtype
        self.use_prompt_stage = True
        ctx_init = "a video of a person"
        ZS_evaluation = False

        # Only the shallow version is provided here
        if cfg.prompt.use_hard_prompt:
          
            prompt_learner = torch.load(cfg.hard_prompt_pretrain, map_location='cuda')["model_state_dict"] 
            width = clip_model.ln_final.weight.shape[0]
            self.proj_b2n_VPT = nn.Sequential(
                nn.Linear(2 * width, width),
                nn.ReLU(),
                nn.Dropout(cfg.prompt.dropout))
            #for k,v in prompt_learner.items():
            #    print(k)
           
            self.ctx_txt_vpt = prompt_learner["prompt_learner.ctx_VPT"].clone() 
            self.ctx_txt_vpt.requires_grad_(False)

        if ZS_evaluation:
            text_aug = f"{{}}"
           
            tokenized_prompts = torch.cat([clip.tokenize(text_aug.format(c), context_length=77) for c in classnames])       
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype).cuda()
            self.register_buffer("complete_text_embeddings", embedding)
            self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        elif self.use_prompt_stage:
            
            n_cls = len(classnames)
            # Make sure Language depth >= 1
            # assert cfg.TRAINER.ViFi_CLIP.PROMPT_DEPTH_TEXT >= 1, "In VL prompting, Language prompt depth should be >=1" \
            #                                             "\nPlease use VPT trainer if you want to learn only vision " \
            #                                             "branch  "
            n_ctx = design_details["language_ctx"]
            ctx_dim = clip_model.ln_final.weight.shape[0]
      
            if ctx_init and n_ctx <= 4:

                # use given words to initialize context vectors
                ctx_init = ctx_init.replace("_", " ")
                n_ctx = n_ctx
                prompt = clip.tokenize(ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
                prompt_prefix = ctx_init
            else:      
                # random initialization
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)

                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)
         
            self.ctx_VPT = nn.Parameter(ctx_vectors)
          
            classnames = [name.replace("_", " ") for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
            
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)

            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            
            # These token vectors will be saved when in save_model(),
            # but they should be ignored in load_model() as we want to use
            # those computed using the current class names
            self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
            self.n_cls = n_cls
            self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        else:
            # No prompting
            ctx_init = ctx_init.replace("_", " ")
            prompt_prefix = ctx_init
         
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
   
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
            with torch.no_grad():
             
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
         
            self.register_buffer("complete_text_embeddings", embedding)

            self.tokenized_prompts = tokenized_prompts  # torch.Tensor
       
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )
        
        return prompts


    def forward(self):
        if self.use_prompt_stage:
            ctx = self.ctx_VPT
           
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            prefix = self.token_prefix
            suffix = self.token_suffix
            
            # use hard prompt
            vanilla_ctx = self.ctx_txt_vpt.expand(ctx.shape[0], -1, -1)
         
            ctx_mid = torch.cat([vanilla_ctx, ctx], dim=-1)
            ctx_mix = self.proj_b2n_VPT(ctx_mid)
            prompts = self.construct_prompts(ctx_mix, prefix, suffix)
            # no use
            #prompts = self.construct_prompts(ctx, prefix, suffix)
        else:
            prompts = self.complete_text_embeddings
           
            
        return prompts

    

class ViFiCLIP(nn.Module):
    def __init__(self, cfg, clip_model, classnames, design_details):
        super().__init__()
   
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model, design_details)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
     
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, list_id, return_token=False, training=False):
        tokenized_prompts = self.tokenized_prompts
 
        prompts = self.prompt_learner()

        cls_feat, text_feats = self.text_encoder(prompts, tokenized_prompts, return_token)
        image_feats = self.image_encoder(image.type(self.dtype))
       
        return image_feats, cls_feat, text_feats, self.logit_scale.exp()
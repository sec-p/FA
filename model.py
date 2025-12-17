import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Normalize
from torch import autograd
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets


import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from torch.distributions import Bernoulli


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, N_CTX = 16, CTX_INIT = "a photo of a", CSC = False, CLASS_TOKEN_POSITION = "end",cfg = None):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = N_CTX
        ctx_init = CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        device = cfg['device']
        

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).cuda()
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]

            if CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = ctx_vectors.repeat(n_cls, 1, 1)
            prompt_prefix = ctx_init

        else:
            # random initialization
            if CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype).to(device)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype).to(device)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # ----- normal optimized
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        # -----


        print(self.ctx.shape)  # 当为"a photo of a"时为[4, 512]，csc为True时为[1000, 4, 512]
        print("------------------")        

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(clip_model.positional_embedding.device)
        # print('tokenized:',tokenized_prompts.shape) #[len(classnames), 77]

        
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            # print('embedding:',embedding.shape)  # [len(classnames), 77, 512]

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        # print('prompts.shape',prompts.shape)  # [len(classnames), ctx_sum, transformer.width] 如[1000, 77, 512]
        return prompts

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts.type(self.dtype) + self.positional_embedding.to(dtype = self.dtype, device = prompts.device)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, _, _, _ = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

def get_original_text_features(prompt_learner, text_encoder, clip_model):
    device = next(text_encoder.parameters()).device

    if clip_model.dtype == torch.float16:
        text_encoder = text_encoder.cuda()

    with torch.no_grad():
        prompts = prompt_learner()
        tokenized_prompts = prompt_learner.tokenized_prompts

        original_text_features = text_encoder(prompts.cuda(), tokenized_prompts.cuda())

    text_encoder = text_encoder.to(device)
    return original_text_features.to(device)

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, 
                 clip_model,
                 CTX_INIT = "a photo of a", 
                 single = False, 
                 n_ctx = 16,
                 csc = False):
        super().__init__()
        
        self.prompt_learner = PromptLearner(classnames, clip_model, CTX_INIT=CTX_INIT, N_CTX=n_ctx, CSC = csc, cfg = cfg)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts 
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.single = single
        self.cfg = cfg
        self.classnum = len(classnames)

        self.text_features_classify_original = get_original_text_features(self.prompt_learner, self.text_encoder, clip_model)

        repeat_sum_num = int(cfg['K']) 

        if repeat_sum_num > 0:
            text_features_classify_original = self.text_features_classify_original.repeat(1, 1)
        else:
            text_features_classify_original = self.text_features_classify_original.repeat(0, 1)
        
        self.text_features_classify_original = text_features_classify_original
        self.repeat_sum_num = repeat_sum_num
        # print(f"self.text_features_classify_original.shape: {self.text_features_classify_original.shape}")

        # gating configuration
        self.use_rl = bool(cfg.get('use_rl', False))
        self.lambda_sparsity = float(cfg.get('lambda_sparsity', 1e-3))
        self.lambda_minimal = float(cfg.get('lambda_minimal', 1e-2))

        # gating network: produce per-token probability for local features
        # determine local feature dim dynamically if possible
        try:
            local_dim = self.image_encoder.output_dim
        except Exception:
            # fallback to CLIP transformer width
            local_dim = clip_model.ln_final.weight.shape[0]

        gate_hidden = int(cfg.get('gate_hidden', 256))

        self.gating = nn.Sequential(
            nn.Linear(local_dim * 2, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        # baseline for policy gradient
        self.register_buffer('baseline', torch.tensor(0.0))

    def forward(self, image, labels=None):        
        # image_features = self.image_encoder(image.type(self.dtype))
        image_features, local_image_features = self.image_encoder(image.type(self.dtype))


        prompts = self.prompt_learner().to(image.device)   #[len(classnames), ctx_sum, transformer.width]
        tokenized_prompts = self.tokenized_prompts.to(image.device)  #[len(classnames), 77]
        text_features = self.text_encoder(prompts, tokenized_prompts).type(self.dtype)

        text_features_classify_final = torch.cat((text_features, self.text_features_classify_original), dim=0)


        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)
        local_image_features = local_image_features / (local_image_features.norm(dim=-1, keepdim=True) + 1e-8)
        text_features_classify_final = text_features_classify_final / (text_features_classify_final.norm(dim=-1, keepdim=True) + 1e-8)

        logit_scale = self.logit_scale.exp()

        # global logits
        logits = logit_scale * image_features @ text_features_classify_final.transpose(-1, -2)

        # gating
        rl_info = {}
        aux_loss = {}

        B, T, C = local_image_features.shape
        device = local_image_features.device

        if labels is not None:
            labels = labels.to(device)

        # build class text bank (assume first self.classnum correspond to classes)
        if text_features_classify_final.shape[0] >= self.classnum:
            class_text_bank = text_features_classify_final[: self.classnum]
        else:
            class_text_bank = text_features_classify_final

        if labels is None:
            with torch.no_grad():
                pseudo = logits.argmax(dim=1)
            selected_text = class_text_bank[pseudo]
        else:
            selected_text = class_text_bank[labels]

        selected_text_exp = selected_text.unsqueeze(1).expand(-1, T, -1)  # [B, T, C]
        gate_input = torch.cat([local_image_features, selected_text_exp], dim=-1)  # [B, T, 2C]
        gate_input_flat = gate_input.view(B * T, -1)
        gate_logits = self.gating(gate_input_flat).view(B, T)

        if self.use_rl:
            probs = torch.sigmoid(gate_logits)
            m = Bernoulli(probs=probs)
            mask_sample = m.sample()
            log_prob = m.log_prob(mask_sample).sum(dim=1)
            mask_apply = mask_sample.detach()
            mask_ratio = mask_apply.mean(dim=1)
            rl_info['log_prob'] = log_prob
            rl_info['mask'] = mask_sample
            rl_info['mask_ratio'] = mask_ratio
        else:
            probs = torch.sigmoid(gate_logits)
            mask_apply = probs
            mask_ratio = mask_apply.mean()
            aux_loss['sparsity'] = mask_ratio

        gated_local = local_image_features * mask_apply.unsqueeze(-1)

        logits_local = logit_scale * gated_local @ text_features_classify_final.T

        # compute negative logits for minimal set constraint
        if labels is not None:
            logits_local_cls = logits_local[:, :, : self.classnum]
            all_idx = torch.arange(self.classnum, device=device)
            negative_list = []
            for i in range(B):
                neg_idx = torch.cat([all_idx[:labels[i]], all_idx[labels[i]+1:]]) if self.classnum>1 else torch.tensor([], device=device, dtype=torch.long)
                if neg_idx.numel() == 0:
                    negative_list.append(torch.zeros((T,0), device=device))
                else:
                    negative_list.append(logits_local_cls[i,:, neg_idx])
            negative_logits = torch.stack([neg for neg in negative_list], dim=0)  # [B, T, classnum-1]
        else:
            negative_logits = logits_local[:, :, : self.classnum]

        if not self.use_rl:
            minimal_loss = torch.mean(negative_logits ** 2)
            aux_loss['minimal'] = minimal_loss
        else:
            with torch.no_grad():
                prob_all = F.softmax(logits_local, dim=-1)
            prob_neg = prob_all[:, :, : self.classnum]
            prob_neg_mean_list = []
            for i in range(B):
                if labels is None:
                    gt = logits[i].argmax()
                else:
                    gt = labels[i]
                mask_idx = torch.ones(self.classnum, device=device, dtype=torch.bool)
                mask_idx[gt] = False
                if mask_idx.sum() == 0:
                    prob_neg_mean_list.append(torch.tensor(0.0, device=device))
                else:
                    prob_neg_mean_list.append(prob_neg[i,:, mask_idx].mean())
            confusion_penalty = torch.stack(prob_neg_mean_list).mean()
            rl_info['confusion_penalty'] = confusion_penalty

        # preserve previous repeat logic
        if self.repeat_sum_num - 1 > 0:
            batch_size, token_len, sum_template_len = logits_local.shape
            logits_repeat_template = logits.view(batch_size, 2, self.classnum)[:,1,:]
            logits_repeat_context = logits_repeat_template.repeat(1, self.repeat_sum_num - 1)
            logits = torch.cat((logits, logits_repeat_context), dim=1)

            logits_local_repeat_template = logits_local.view(batch_size, token_len, 2, self.classnum)[:,:,1,:]
            logits_local_repeat_context = logits_local_repeat_template.repeat(1, 1, self.repeat_sum_num - 1)
            logits_local = torch.cat((logits_local, logits_local_repeat_context), dim=2)

        return logits, logits_local, aux_loss, rl_info
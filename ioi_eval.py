import sys
sys.path.append('limber')
sys.path.append('EasyTransformer')
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import DataLoader
import numpy as np
from transformer_lens import HookedTransformer
from EasyTransformer.easy_transformer.ioi_dataset import IOIDataset

from limber.use_limber import simple_load_model

from load_mscoco import COCOImageDataset
from limber_lens import ablate_img_block_hook, ablate_text_block_hook, ablate_txt2img_attn, overwrite_rand


batch_size = 256
total_size = batch_size * 16
torch.set_grad_enabled(False)
torch.cuda.set_device(1)
config_path = 'limber/configs/clip_linear.yml'
model = simple_load_model(config_path, limber_proj_path='limber/limber_weights/clip_linear/proj.ckpt')
model = model.cuda().half()
print("Loaded model")

hooked_model = HookedTransformer.from_pretrained("gpt-j-6B", default_prepend_bos=False, tokenizer=model.tokenizer, device='cuda:1', dtype='float16')
ablate_layers = 28
ablate_hooks = []
for i in range(ablate_layers):
    ablate_hooks.append((f'blocks.{i}.hook_attn_out', ablate_img_block_hook))
    ablate_hooks.append((f'blocks.{i}.hook_mlp_out', ablate_img_block_hook))
    # ablate_hooks.append((f'blocks.{i}.hook_attn_out', ablate_text_block_hook))
    # ablate_hooks.append((f'blocks.{i}.hook_mlp_out', ablate_text_block_hook))
    # ablate_hooks.append((f'blocks.{i}.attn.hook_attn_scores', ablate_txt2img_attn))
    # ablate_hooks.append((f'blocks.{i}.hook_resid_pre', overwrite_rand))

ioi_ds = IOIDataset('mixed', N=total_size, tokenizer=model.tokenizer)
print("Loaded IOI")

coco_ds = COCOImageDataset(f'/media/andrelongon/DATA/mscoco2017_val', transform=model.transforms)
print("Loaded COCO")
dataloader = DataLoader(coco_ds, batch_size=batch_size, shuffle=True, drop_last=True)
data_iter = iter(dataloader)

acc = 0
for i in range(0, total_size, batch_size):
    imgs = None
    try:
        imgs, _ = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        imgs, _ = next(data_iter)

    text_embeds = model.word_embedding(ioi_ds.toks[i:i+batch_size].cuda())
    text_norms = torch.mean(torch.linalg.vector_norm(text_embeds, dim=-1), dim=1)

    imgs = imgs.cuda().half()
    img_embeds = model.image_prefix(imgs)
    img_norms = torch.mean(torch.linalg.vector_norm(img_embeds, dim=-1), dim=1)
    norm_ratios = text_norms / (img_norms + 1e-10)
    img_embeds = torch.mul(img_embeds, norm_ratios[:, None, None].expand(-1, img_embeds.shape[1], img_embeds.shape[2]))

    # rand_embeds = torch.normal(0, 1, size=img_embeds.shape).cuda()
    # rand_norms = torch.mean(torch.linalg.vector_norm(rand_embeds, dim=-1), dim=1)
    # norm_ratios = text_norms / (rand_norms + 1e-10)
    # rand_embeds = torch.mul(rand_embeds, norm_ratios[:, None, None].expand(-1, img_embeds.shape[1], img_embeds.shape[2]))

    embeds = torch.cat([img_embeds, text_embeds], dim=1)
    logits = hooked_model.run_with_hooks(embeds, prepend_bos=False, start_at_layer=0, stop_at_layer=None, return_type='logits', fwd_hooks=ablate_hooks)

    IO_logits = logits[
        torch.arange(batch_size),
        ioi_ds.word_idx["end"][i:i+batch_size] + img_embeds.shape[1],
        ioi_ds.io_tokenIDs[i:i+batch_size],
    ]
    S_logits = logits[
        torch.arange(batch_size),
        ioi_ds.word_idx["end"][i:i+batch_size] + img_embeds.shape[1],
        ioi_ds.s_tokenIDs[i:i+batch_size],
    ]

    acc += torch.nonzero((IO_logits - S_logits) > 0).shape[0]

print(f'Final Accuracy:  {acc / total_size}')
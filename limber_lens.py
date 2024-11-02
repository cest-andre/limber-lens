import sys
sys.path.append('limber')

from image_input import ImageInput
import os
from limber_gptj import LimberGPTJ
import torch
import numpy as np
from PIL import Image

from limber.use_limber import simple_load_model
from transformer_lens import HookedTransformer


image_seq_len = 144

def ablate_img_block_hook(value, hook):
    value[:, :image_seq_len, :] = 0.
    return value


def ablate_text_block_hook(value, hook):
    value[:, image_seq_len:, :] = 0.
    return value


def ablate_txt2img_attn(value, hook):
    value[:, :, image_seq_len:, :image_seq_len] = -float('inf')
    return value


def overwrite_rand(value, hook):
    mean_image_norm = 168.125  #  mean projector out norm across MSCOCO 2017 val for Limber+CLIP
    mean_text_norm = 1.988  #  mean embedding weight norms for GPTJ
    rand_embeds = torch.normal(0, 1, size=value[:, :image_seq_len, :].shape).cuda()
    rand_norms = torch.mean(torch.linalg.vector_norm(rand_embeds, dim=-1), dim=1)

    norm_ratios = mean_image_norm / (rand_norms + 1e-10)
    rand_embeds = torch.mul(rand_embeds, norm_ratios[:, None, None].expand(-1, rand_embeds.shape[1], rand_embeds.shape[2]))

    value[:, :image_seq_len, :] = rand_embeds

    return value


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    config_path = 'limber/configs/clip_linear.yml'
    model = simple_load_model(config_path, limber_proj_path='limber/limber_weights/clip_linear/proj.ckpt')
    print("Loaded model")
    model = model.cuda().half()
    imginp = ImageInput(r'/home/ajl_onion123/mscoco2017_val/val2017/000000015746.jpg')
    embeds = model.preprocess_inputs([imginp, 'A picture of'])
    model.cpu()
    
    model = HookedTransformer.from_pretrained("gpt-j-6B", default_prepend_bos=False, tokenizer=model.tokenizer, device='cuda:0', dtype='float16')

    #  TODO:  Ensure image token block ablation works!  Can I run_with_cache with the hooks and print out resid pre/post for all the blocks?
    num_layers = 27
    block_hooks = []
    for i in range(num_layers):
        print(f'\nABLATE FIRST {i+1} LAYERS')
        block_hooks.append((f'blocks.{i}.hook_attn_out', ablate_img_block_hook))
        block_hooks.append((f'blocks.{i}.hook_mlp_out', ablate_img_block_hook))
        embed_clone = torch.clone(embeds)
        answer = []
        for j in range(10):
            logits = model.run_with_hooks(embed_clone, prepend_bos=False, start_at_layer=0, stop_at_layer=None, return_type='logits', fwd_hooks=block_hooks)
            if j == 0:
                print("Top Logits on first pass")
                # print(torch.topk(logits[0, -1], 16))
                print(model.to_string(torch.topk(logits[0, -1], 16)[1]))

            next_token = torch.argmax(logits[0, -1], dim=-1).cuda()
            if next_token.item() == model.tokenizer.eos_token_id:
                break

            answer.append(next_token.item())
            embed_clone = torch.cat((embed_clone, model.input_to_embed(next_token[None].cuda())[0]), dim=1)

        print("Final Description")
        print(model.to_string(answer))
import sys
sys.path.append('limber')
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import DataLoader
import numpy as np
import clip
from transformer_lens import HookedTransformer

from limber.use_limber import simple_load_model

from load_mscoco import COCOImageDataset
from limber_lens import ablate_img_block_hook, ablate_text_block_hook, ablate_txt2img_attn


batch_size = 128
torch.set_grad_enabled(False)
config_path = 'limber/configs/clip_linear.yml'
model = simple_load_model(config_path, limber_proj_path='limber/limber_weights/clip_linear/proj.ckpt')
model = model.cuda().half()
print("Loaded model")
text_embeds = model.tokenizer.encode('A picture of', return_tensors="pt").cuda()
text_embeds = model.word_embedding(text_embeds)
text_norm = torch.mean(torch.linalg.vector_norm(text_embeds, dim=-1))

# all_text_norms = torch.mean(torch.linalg.vector_norm(model.word_embedding.state_dict()['weight'], dim=-1))
# print(f'Mean Text Norm: {all_text_norms}')

clip_model = clip.load('ViT-L/14@336px', device='cpu')[0].half()
clip_resize = T.Resize(336, interpolation=T.InterpolationMode.BICUBIC)

hooked_model = HookedTransformer.from_pretrained("gpt-j-6B", default_prepend_bos=False, tokenizer=model.tokenizer, device='cpu', dtype='float16')

coco_ds = COCOImageDataset(f'/media/andrelongon/DATA/mscoco2017_val', transform=model.transforms)
print("Loaded COCO")
dataloader = DataLoader(coco_ds, batch_size=batch_size, shuffle=False, drop_last=False)

all_embeds = []
# all_clip_norms = []
# all_projector_norms = []
print("Computing embeds")
for i, (imgs, gt_capts) in enumerate(dataloader):
    imgs = imgs.cuda().half()

    #   Compute encoder and projector output norms.
    # img_embeds = model.image_prefix.encode(imgs)
    # clip_norm = torch.mean(torch.linalg.vector_norm(img_embeds, dim=-1))
    # all_clip_norms.append(clip_norm.cpu())
    # img_embeds = model.image_prefix.project(img_embeds)
    # proj_norm = torch.mean(torch.linalg.vector_norm(img_embeds, dim=-1))
    # all_projector_norms.append(proj_norm.cpu())

    img_embeds = model.image_prefix(imgs)
    img_norms = torch.mean(torch.linalg.vector_norm(img_embeds, dim=-1), dim=1)
    norm_ratio = text_norm / (img_norms + 1e-10)
    img_embeds = torch.mul(img_embeds, norm_ratio[:, None, None].expand(-1, img_embeds.shape[1], img_embeds.shape[2]))
    embeds = torch.cat([img_embeds, text_embeds.expand(imgs.shape[0], -1, -1)], dim=1)
    all_embeds.append(embeds.cpu())

# print(f'Mean Text Norm: {all_text_norms}')
# print(f'Mean CLIP Encoding Norm: {torch.mean(torch.tensor(all_clip_norms))}')
# print(f'Mean Projector Out Norm: {torch.mean(torch.tensor(all_projector_norms))}')

model.cpu()
hooked_model.cuda()
clip_model.cuda()
num_layers = 28
ablate_hooks = []
for i in range(num_layers):
    ablate_hooks.append((f'blocks.{i}.hook_attn_out', ablate_img_block_hook))
    ablate_hooks.append((f'blocks.{i}.hook_mlp_out', ablate_img_block_hook))
    ablate_hooks.append((f'blocks.{i}.attn.hook_attn_scores', ablate_txt2img_attn))
    all_pred_capts = []
    for embeds in all_embeds:
        embeds = embeds.cuda()
        answer = []
        logits = hooked_model.run_with_hooks(embeds, prepend_bos=False, start_at_layer=0, stop_at_layer=None, return_type='logits', fwd_hooks=ablate_hooks)
        pred_capts = hooked_model.to_string(torch.topk(logits[:, -1], 16)[1].cuda())
        all_pred_capts.append(pred_capts)

    # hooked_model.cpu()
    all_pred_scores = torch.tensor([])
    all_gt_scores = torch.tensor([])
    for j, (imgs, gt_capts) in enumerate(dataloader):
        imgs = imgs.cuda().half()
        pred_capts = all_pred_capts[j]

        imgs = clip_resize(imgs)
        imgs_out = F.normalize(clip_model.encode_image(imgs).cuda())
        text_out = F.normalize(clip_model.encode_text(clip.tokenize(pred_capts).cuda()))
        gt_out = F.normalize(clip_model.encode_text(clip.tokenize(gt_capts).cuda()))

        pred_scores = 100*torch.sum(torch.mul(imgs_out, text_out), -1)
        all_pred_scores = torch.cat((all_pred_scores, pred_scores.cpu()))
        gt_scores = 100*torch.sum(torch.mul(imgs_out, gt_out), -1)
        all_gt_scores = torch.cat((all_gt_scores, gt_scores.cpu()))

        # embed_clone = torch.clone(embeds)
        # batch_eos = torch.zeros(batch_size, dtype=torch.bool)
        # for j in range(57):
        #     logits = hooked_model.run_with_hooks(embed_clone, prepend_bos=False, start_at_layer=0, stop_at_layer=None, return_type='logits')#, fwd_hooks=block_hooks)
        #     # if j == 0:
        #     #     print("Top Logits on first pass")
        #     #     # print(torch.topk(logits[0, -1], 16))
        #     #     print(hooked_model.to_string(torch.topk(logits[0, -1], 16)[1]))

        #     next_token = torch.argmax(logits[:, -1], dim=-1).cuda()
        #     batch_eos[next_token == hooked_model.tokenizer.eos_token_id] = True
        #     if torch.all(batch_eos).item():
        #         break

        #     answer.append(next_token)
        #     next_embeds = hooked_model.input_to_embed(next_token[:, None])[0]
        #     embed_clone = torch.cat((embed_clone, next_embeds), dim=1)

        # answer = torch.stack(answer, dim=1)
        # out_capt = hooked_model.to_string(answer)
        # out_capt = [s.split(hooked_model.tokenizer.eos_token)[0] for s in out_capt]
        # print("Final Description")
        # print(out_capt)

    print(f"\nFirst {i+1} Layers Ablated")
    mean_pred = torch.mean(all_pred_scores).item()
    mean_gt = torch.mean(all_gt_scores).item()
    print(f"Mean Pred Score:  {mean_pred}\nPred / GT:  {mean_pred / mean_gt}")

    # np.save(f'/media/andrelongon/DATA/limber_lens_results/clipscores/first{i+1}_img_out_img2txt_attn_ablate_scores.npy', all_pred_scores.numpy())
    np.save(f'/media/andrelongon/DATA/limber_lens_results/clipscores/reduced_norm_first{i+1}_img_out_img2txt_attn_ablate_scores.npy', all_pred_scores.numpy())
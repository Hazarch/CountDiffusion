import os
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import warnings
warnings.filterwarnings("ignore")  # ignore warning
import re
import argparse
from datetime import datetime
from tqdm import tqdm
import torch
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from transformers import T5EncoderModel, T5Tokenizer

from diffusion.model.utils import prepare_prompt_ar
from diffusion import IDDPM, DPMS, SASolverSampler
from tools.download import find_model
from diffusion.model.nets import PixArtMS_XL_2, PixArt_XL_2
from diffusion.data.datasets import get_chunks
from diffusion.data.datasets.utils import *

from omegaconf import OmegaConf
from tools.sam import sam
import json
from torchvision import transforms


def get_args():
    parser = argparse.ArgumentParser()

    # pixart control
    parser.add_argument(
        "--pipeline_load_from", default='/data1/cache/pixart/pixart_sigma_sdxlvae_T5_diffusers',
        type=str, help="Download for loading text_encoder, "
                       "tokenizer and vae from https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
    )
    parser.add_argument('--model_path', default='/data1/cache/pixart/PixArt-Sigma-XL-2-1024-MS.pth', type=str)
    parser.add_argument('--cfg_scale', default=4.5, type=float)
    parser.add_argument('--seed', default=3633, type=int)
    parser.add_argument('--step', default=-1, type=int)
    parser.add_argument('--save_path', default='outputs', type=str)

    # CountDiffusion control
    parser.add_argument('--ori', action='store_true', help="run pixart without CountDiffusion")
    parser.add_argument('--txt_file', default=None, type=str, help='When txt_file seted, prompts and token are invailable. instances in scripts/samples.json')
    parser.add_argument('--prompts', default='Three apples and two cat on the grass', type=str)
    parser.add_argument(
        '--token', 
        default={
            "apples": 3,
            "cat": 2
        }, 
        type=dict)
    parser.add_argument('--max_revers_guidance', default=15, type=int, help='max steps applying revers guidance')
    parser.add_argument('--MCS', action='store_true', help='Multi_Class_Strategy')
    parser.add_argument('--save_mid', action='store_true', help='save the intermediate result')

    parser.add_argument('--smooth_attentions', default=True, type=bool)
    parser.add_argument('--sigma', default=0.5, help='smooth_attentions control') 
    parser.add_argument('--kernel_size', default=3, help='smooth_attentions control')     

    parser.add_argument('--P', default=0.8, help='topk(P), range from 0 to 1')
    parser.add_argument('--intensity_scale', default=10, type=int)
    parser.add_argument('--intensity_scale_decay', default=(1.0, 0.8))

    parser.add_argument('--device_num', default=-1, type=int)

    return parser.parse_args()


def set_env(seed=0):
    if args.device_num != -1:
        torch.cuda.set_device(args.device_num)
    torch.manual_seed(seed)

def visualize(items, sample_steps, cfg_scale):

    for chunk in tqdm(list(get_chunks(items, 1)), unit='batch'):

        bs = 1
        prompts = []
        prompt_clean, _, hw, ar, custom_hw = prepare_prompt_ar(chunk[0], base_ratios, device=device, show=False)  # ar for aspect ratio
        if args.image_size == 1024:
            latent_size_h, latent_size_w = int(hw[0, 0] // 8), int(hw[0, 1] // 8)
        else:
            hw = torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device).repeat(bs, 1)
            ar = torch.tensor([[1.]], device=device).repeat(bs, 1)
            latent_size_h, latent_size_w = latent_size, latent_size
        prompts.append(prompt_clean.strip())

        caption_token = tokenizer(prompts, max_length=max_sequence_length, padding="max_length", truncation=True,
                                  return_tensors="pt").to(device)
        caption_embs = text_encoder(caption_token.input_ids, attention_mask=caption_token.attention_mask)[0]
        emb_masks = caption_token.attention_mask

        caption_embs = caption_embs[:, None]
        null_y = null_caption_embs.repeat(len(prompts), 1, 1)[:, None]
        print(f'finish embedding')

        # find index of tokens
        args.indices_to_correct= dict()
        remove_ids = [1, 7, 3, 15]     # [</s>, s, , e]
        for key in args.token.keys():
            key_token = tokenizer(key, max_length=300, padding="do_not_pad", truncation=True, return_tensors="pt").to(device).input_ids[0].tolist()
            key_token = [token for token in key_token if token not in remove_ids]
            if len(key_token) == 1:
                token_id = key_token[0]
                index = caption_token.input_ids[0].tolist().index(token_id)
            else:
                print(f"'{key}' is tokenized to {[tokenizer.decode(i) for i in key_token]}, we will take the mean of their attention maps")
                index = []
                caption_token_list = caption_token.input_ids[0].tolist()
                for i in key_token:
                    index.append(caption_token_list.index(i))
            args.indices_to_correct.update({key: index})

        with torch.no_grad():

            # Create sampling noise:
            n = len(prompts)
            z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device)
            model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
            dpm_solver = DPMS(model.forward_with_dpmsolver,
                                condition=caption_embs,
                                uncondition=null_y,
                                cfg_scale=cfg_scale,
                                model_kwargs=model_kwargs)
            
            # Detection stage
            first_samples, intermediates = dpm_solver.sample(
                z,
                steps=sample_steps,
                order=2,
                skip_type="time_uniform",
                method="multistep",
                return_intermediate=True,
                args=args,
            )

            # Correction stage
            if args.ori:
                samples = first_samples
            else:
                samples = first_samples.to(weight_dtype)
                samples = vae.decode(samples / vae.config.scaling_factor).sample
                pil = transform(samples[0])
                pred_bbox_dict = sam_model(pil, tags=list(args.token.keys()), save_file=None)
                if args.save_mid:
                    save_name = prompts[0][:100].strip()
                    save_name = save_name[:-1] if save_name.endswith('.') else save_name
                    save_path = os.path.join(save_root, f"{save_name}_intermediate.jpg")
                    save_image(samples[0], save_path, nrow=1, normalize=True, value_range=(-1, 1))

                samples = dpm_solver.sample(
                    z,
                    steps=sample_steps,
                    order=2,
                    skip_type="time_uniform",
                    method="multistep",
                    args=args,
                    store_intermediates=intermediates,
                    pred_bbox_dict=pred_bbox_dict
                )

        samples = samples.to(weight_dtype)
        samples = vae.decode(samples / vae.config.scaling_factor).sample
        torch.cuda.empty_cache()
        # Save images:
        os.umask(0o000)  # file permission: 666; dir permission: 777
        for i, sample in enumerate(samples):
            save_name = prompts[i][:100].strip()
            save_name = save_name[:-1] if save_name.endswith('.') else save_name
            save_path = os.path.join(save_root, f"{save_name}.jpg")
            # print("Saving path: ", save_path)
            save_image(sample, save_path, nrow=1, normalize=True, value_range=(-1, 1))


if __name__ == '__main__':
    args = get_args()
    # Setup PyTorch:
    seed = args.seed
    set_env(seed)
    device = "cuda" # only support cuda

    sam_config = 'configs/sam_config.yaml'

    # only support fixed latent size currently
    args.image_size = 1024      # only support 1024 for now
    latent_size = args.image_size // 8
    max_sequence_length = 300
    pe_interpolation = {256: 0.5, 512: 1, 1024: 2, 2048: 4}     # trick for positional embedding interpolation
    sample_steps = args.step if args.step != -1 else 30
    assert sample_steps > args.max_revers_guidance, ValueError('sample_steps must be biger than max_revers_guidance')
    weight_dtype = torch.float16
    print(f"Inference with {weight_dtype}")

    # model setting
    model = PixArtMS_XL_2(
        input_size=latent_size,
        pe_interpolation=pe_interpolation[args.image_size],
        model_max_length=max_sequence_length,
    ).to(device)

    print("Generating sample from ckpt: %s" % args.model_path)
    state_dict = find_model(args.model_path)
    if 'pos_embed' in state_dict['state_dict']:
        del state_dict['state_dict']['pos_embed']
    missing, unexpected = model.load_state_dict(state_dict['state_dict'], strict=False)
    print('Missing keys: ', missing)
    print('Unexpected keys', unexpected)
    model.eval()
    model.to(weight_dtype)
    base_ratios = eval(f'ASPECT_RATIO_{args.image_size}_TEST')

    # pixart-Sigma vae link: https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers/tree/main/vae
    vae = AutoencoderKL.from_pretrained(f"{args.pipeline_load_from}/vae").to(device).to(weight_dtype)

    # load Grounded SAM
    sam_model = sam(OmegaConf.load(sam_config).sam, device=device)

    tokenizer = T5Tokenizer.from_pretrained(args.pipeline_load_from, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(args.pipeline_load_from, subfolder="text_encoder").to(device)

    null_caption_token = tokenizer("", max_length=max_sequence_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
    null_caption_embs = text_encoder(null_caption_token.input_ids, attention_mask=null_caption_token.attention_mask)[0]

    work_dir = args.save_path

    transform = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        transforms.ToPILImage()
    ])

    # data setting
    if args.txt_file is not None:
        with open(args.txt_file, 'r') as f:
            items = [item.strip() for item in f.readlines()]
    else:
        items = args.prompts if isinstance(args.prompts, list) else [args.prompts]
        

    # img save setting
    img_save_dir = os.path.join(work_dir, 'vis')
    os.umask(0o000)  # file permission: 666; dir permission: 777
    os.makedirs(img_save_dir, exist_ok=True)

    save_root = os.path.join(img_save_dir, f"{datetime.now().date()}_step{sample_steps}_seed{seed}")
    os.makedirs(save_root, exist_ok=True)
    
    if args.txt_file is not None:
        inputs = json.load(open(args.txt_file))
        for one_input in inputs:
            items = [one_input[0]]
            args.token = one_input[1]
            visualize(items, sample_steps, args.cfg_scale)
    else:
        visualize(items, sample_steps, args.cfg_scale)

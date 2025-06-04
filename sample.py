import argparse
import numpy as np
import random, os
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.utils import save_image
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as torch_transforms
import torch.nn as nn
import tqdm

from sampler import SD3EulerDC, SDXLEulerDC, SD1EulerDC
from dataset.datasets import get_target_dataset
import json

INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose([
        torch_transforms.Resize(size, interpolation=interpolation),
        torch_transforms.CenterCrop(size),
        _convert_image_to_rgb,
        torch_transforms.ToTensor(),
        torch_transforms.Normalize([0.5], [0.5])
    ])
    return transform

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # sampling config
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--img_size', type=int, default=1024, choices=[256,512,768,1024])
    parser.add_argument('--NFE', type=int, default=28)
    parser.add_argument('--cfg_scale', type=float, default=1.0, help='0 for null prompt, 1 for only using conditional prompt')
    parser.add_argument('--batch_size', type=int, default=1)
    # path
    parser.add_argument('--load_dir', type=str, default=None, help="replace it with your checkpoint")
    parser.add_argument('--save_dir', type=str, default=None, help="default savedir is set to under load_dir")
    parser.add_argument('--datadir', type=str, default='', required=True, help='data path')
    # model config
    parser.add_argument('--model', type=str, default='sd3', choices=['sd3', 'sdxl', 'sd1.5'], help='Model to use')
    parser.add_argument('--use_dc', action='store_true', default=False)
    parser.add_argument('--use_dc_t', type=str, default=False, help='use t dependent')
    parser.add_argument('--n_dc_tokens', type=int, default=4)
    parser.add_argument('--n_dc_layers', type=int, default=5, help='sd3')
    parser.add_argument('--apply_dc', nargs='+', type=str, default=[True, True, False], help='sdxl, sd1.5')
    # one sample generation
    parser.add_argument('--prompt', type=str, default="")
    parser.add_argument('--save_name', type=str, default="image_sd3")
    # set generation
    parser.add_argument('--num', type=int, default=-1, help='number of sampling images. -1 for whole dataset')
    parser.add_argument('--dataset', type=str, nargs='+', default=None, choices=['coco'])
    
    args = parser.parse_args()
    set_seed(args.seed)

    args.apply_dc = [str2bool(x) for x in args.apply_dc]
    args.use_dc_t = str2bool(args.use_dc_t)

    interpolation = INTERPOLATIONS['bilinear']
    transform = get_transform(interpolation, 1024)

    # load model
    if args.model == 'sd3':
        sampler = SD3EulerDC(n_dc_tokens=args.n_dc_tokens, use_8bit=False, use_dc_t=args.use_dc_t, n_dc_layers=args.n_dc_layers)
    elif args.model == 'sdxl':
        sampler = SDXLEulerDC(n_dc_tokens=args.n_dc_tokens, use_8bit=False, use_dc_t=args.use_dc_t, apply_dc=args.apply_dc)
    elif args.model == 'sd1.5':
        sampler = SD1EulerDC(n_dc_tokens=args.n_dc_tokens, use_8bit=False, use_dc_t=args.use_dc_t, apply_dc=args.apply_dc)
    else:
        raise ValueError('args.model should be one of [sd3, sdxl, sd1.5]')
    
    # load tokens
    if args.load_dir is not None:
        sampler.denoiser.dc_tokens = torch.load(os.path.join(args.load_dir, f'dc_tokens.pth'), map_location='cuda', weights_only=True)
        if args.use_dc_t:
            sampler.denoiser.dc_t_prompts = torch.load(os.path.join(args.load_dir, f'dc_t_tokens.pth'), map_location='cuda')
    
    # sample set
    if args.dataset is not None:
        # save dir
        config=f'{"-".join(args.dataset)}-cfg{args.cfg_scale}-dc{args.use_dc}-dct{args.use_dc_t}-nfe{args.NFE}'
        if args.save_dir is not None:
            args.load_dir = args.save_dir
        savedir = os.path.join(args.load_dir, config)
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        train_datasets = []
        for ds in args.dataset:
            train_datasets.append(get_target_dataset(ds, args.datadir, train=False, transform=transform))

        train_dataset = ConcatDataset(train_datasets)
        num = args.num if args.num != -1 else len(train_dataset)
        train_dataset = Subset(train_dataset, list(range(num)))
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4)
        pbar = tqdm.tqdm(train_dataloader)
        i=0
        results = []
        for _, label in pbar:
            if os.path.exists(os.path.join(savedir, f'{i+args.batch_size:04d}.png')):
                i+=1
                continue
            img = sampler.sample(label, NFE=args.NFE, img_shape=(args.img_size, args.img_size), cfg_scale=args.cfg_scale, use_dc=args.use_dc, batch_size=len(label))
            for bi in range(img.shape[0]):
                imgname = f'{i:04d}.png'
                save_image(img[bi], os.path.join(savedir, imgname), normalize=True)
                results.append({"prompt": label[bi], "img_path": imgname})
                pbar.set_description(f'SD Sampling [{i}/{num}]')
                i+=1
        
        # save config
        if os.path.exists(os.path.join(args.load_dir, f"results-{config}.json")):
            with open(os.path.join(args.load_dir, f"results-{config}.json"), 'r', encoding='utf-8') as file:
                results_all = json.load(file)
                if isinstance(results_all, list):
                    results_all.extend(results)
                else:
                    results_all = [results_all] + results
        else:
            results_all = results
        with open(os.path.join(args.load_dir, f"results-{config}.json"), "w", encoding="utf-8") as file:
            json.dump(results_all, file, indent=4)  # `indent=4` makes the JSON more readable

    # sample image
    else:
        # save dir
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        img = sampler.sample([args.prompt], NFE=args.NFE, img_shape=(args.img_size, args.img_size), cfg_scale=args.cfg_scale, use_dc=args.use_dc, batch_size=1)
        save_image(img, os.path.join(args.save_dir, f'{args.save_name}.png'), normalize=True)
    
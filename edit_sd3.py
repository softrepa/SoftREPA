import argparse
from pathlib import Path
import random
import os

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image

from sampler import SD3EulerFE

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# def load_img(img_path: Path, img_size:Tuple[int, int]=img_shape) -> torch.Tensor:
def load_img(img_path: Path, img_size=None) -> torch.Tensor:
    if img_size is None:
        tf = transforms.Compose([
            transforms.ToTensor()
        ])
    else:
        tf = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])
    img=Image.open(img_path).convert('RGB')
    size=img.size[::-1]
    img = tf(img).unsqueeze(0)
    return img, size

def resize_img(img, img_size=(1024,1024)):
    tf = transforms.Resize(img_size)
    return tf(img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--method', type=str, default="flowedit")
    # flowedit parameter
    parser.add_argument('--NFE', type=int, default=50)
    parser.add_argument('--n_start', type=int, default=30, help='33 for flowedit, 30 for softrepa')
    parser.add_argument('--src_cfg_scale', type=float, default=3.5, help='for flowedit src cfg scale')
    parser.add_argument('--tar_cfg_scale', type=float, default=9.0, help='for flowedit tgt cfg scale, 13.5 for default, 7 for ours')
    # softrepa parameter
    parser.add_argument('--use_dc', action='store_true', default=False, help='True for softrepa, False for flowedit')
    parser.add_argument('--use_dc_t', action='store_true', default=False, help='time dependent tokens')
    parser.add_argument('--n_dc_tokens', type=int, default=4, help='the number of soft text tokens')
    parser.add_argument('--n_dc_layers', type=int, default=5, help='the number of layers to use soft text tokens')
    parser.add_argument('--load_dir', type=str, default='tokens/sd3/', help='softrepa parameters loading path')
    # image 
    parser.add_argument('--tgt_prompt', type=str, default="The image features a small, fluffy gray rabbit sitting on a wooden surface, possibly a tree branch or a fence. The rabbit is holding a carrot in its paws, which it might be eating or preparing to eat.\n")
    parser.add_argument('--src_prompt', type=str, default="The image features a small, fluffy brown squirrel sitting on a wooden surface, possibly a tree branch or a fence. The squirrel is holding a nut in its paws, which it might be eating or preparing to eat.\n")
    parser.add_argument('--input_path', type=Path, default=Path("./samples/0001.png"), help='src image path')
    parser.add_argument('--img_size', type=int, default=1024, help='image size, -1 for preserve src image size')
    parser.add_argument('--workdir', type=Path, default="edited/", help='savedir')
    
    args = parser.parse_args()

    args.workdir = Path(args.workdir)
    if not os.path.exists(args.workdir):
            os.makedirs(args.workdir)
    set_seed(args.seed)

    sampler = SD3EulerFE()
    sampler.ch_transformer(n_dc_layers=args.n_dc_layers, n_dc_tokens=args.n_dc_tokens, use_dc_t=args.use_dc_t, device='cuda')
    if args.use_dc:
        sampler.denoiser.dc_tokens = torch.load(os.path.join(args.load_dir, f'dc_tokens.pth'), map_location='cuda')
        if args.use_dc_t:
            sampler.denoiser.dc_t_tokens = torch.load(os.path.join(args.load_dir, f'dc_t_tokens.pth'), map_location='cuda')

    input_path = args.input_path
    basename = os.path.basename(input_path).replace('.png', '')

    src_img, src_size = load_img(args.input_path) if args.img_size==-1 else load_img(args.input_path, img_size=(args.img_size, args.img_size))
    src_img = src_img * 2.0 - 1.0
    img_shape = src_img.shape[-2:]

    if args.use_dc:
        writepath=args.workdir.joinpath(f'{basename}-softrepa.png')
        output = sampler.sample(src_img, [args.src_prompt], [args.tgt_prompt],
                                NFE=args.NFE, img_shape=img_shape,
                                n_start=args.n_start, src_cfg_scale=args.src_cfg_scale, tar_cfg_scale=args.tar_cfg_scale, use_dc=args.use_dc)
        save_image(output, writepath, normalize=True)
    else:
        writepath=args.workdir.joinpath(f'{basename}-flowedit.png')
        output = sampler.sample(src_img, [args.src_prompt], [args.tgt_prompt],
                                NFE=args.NFE, img_shape=img_shape,
                                n_start=args.n_start, src_cfg_scale=args.src_cfg_scale, tar_cfg_scale=args.tar_cfg_scale)
        save_image(output, writepath, normalize=True)

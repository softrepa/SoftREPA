import argparse
import numpy as np
import random, os
import torch
import tqdm
from torchvision import transforms
from PIL import Image

from dataset.datasets import get_target_dataset
import ImageReward as RM
import json
from eval_utils import PickScore, HPSv2
from pytorch_fid import fid_score
import lpips

# Define image transformation (Ensure 299x299 and uint8 format)
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Resize to InceptionV3 input size
    transforms.ToTensor(),  # Convert to tensor
    lambda x: (x * 255).byte(),  # Convert float32 (0-1) to uint8 (0-255)
])

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
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--load_name', type=str, default=None)
    parser.add_argument('--datadir', type=str, default=None, help='source dataset dir')
    parser.add_argument('--dataset', type=str, default="coco", choices=["coco"])
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--num', type=int, default=-1, help='-1 for all data')
    parser.add_argument(
        "--benchmark",
        default="ImageReward-v1.0, CLIP, PickScore, HPS, FID, LPIPS",
        type=str,
        help="ImageReward-v1.0, Aesthetic, BLIP or CLIP, FID splitted with comma(,) if there are multiple benchmarks.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    if args.num==-1: num='all'
    else: num=args.num

    if args.overwrite:
        with open(os.path.join(args.load_dir, f"metric-{args.load_name}-{num}.json"), "r", encoding="utf-8") as file:
            results = json.load(file)
    else:
        with open(os.path.join(args.load_dir, f"results-{args.load_name}.json"), "r", encoding="utf-8") as file:
            results = json.load(file)

    benchmark_types = args.benchmark.split(",")
    benchmark_types = [x.strip() for x in benchmark_types]
    if args.overwrite:
        with open(os.path.join(args.load_dir, f"benchmark_metric-{args.load_name}-{num}.json"), "r", encoding="utf-8") as file:
            benchmark_results = json.load(file)
    else:
        benchmark_results = {}

    for benchmark_type in benchmark_types:
        print('Benchmark Type: ', benchmark_type)
        model = None
        reward_list = []
        if benchmark_type == "ImageReward-v1.0":
            model = RM.load(name=benchmark_type, device="cuda")
        elif benchmark_type == "PickScore":
            model = PickScore(device="cuda")
        elif benchmark_type == "HPS":
            model = HPSv2()
        elif benchmark_type == 'CLIP':
            model = RM.load_score(
                name=benchmark_type, device="cuda"
            )
        elif benchmark_type == 'FID':
            continue
        elif benchmark_type == 'LPIPS':
            continue
        else:
            raise NotImplementedError(f"Unknown benchmark type: {benchmark_type}")
        
        with torch.no_grad():
            if args.num != -1: pbar = tqdm.tqdm(range(args.num))
            else: pbar = tqdm.tqdm(range(len(results)))
            for i in pbar:
                prompt = results[i]["prompt"]
                img_path = os.path.join(args.load_dir, f'{args.load_name}', results[i]["img_path"])

                if benchmark_type in ["ImageReward-v1.0", "PickScore", "HPS"]:
                    rewards = model.score(prompt, [img_path])

                elif benchmark_type in ['CLIP']:
                    _, rewards = model.inference_rank(prompt, [img_path])
                
                if isinstance(rewards, list):
                    rewards = float(rewards[0])

                results[i][benchmark_type] = rewards
                reward_list.append(rewards)
        reward_list = np.array(reward_list)
        benchmark_results[benchmark_type] = reward_list.mean()
        print(f"{benchmark_type}: {benchmark_results[benchmark_type]}")
    

    if 'FID' in benchmark_types:
        gen_path = str(os.path.join(args.load_dir, f'{args.load_name}'))
        real_path = str(os.path.join(args.datadir, 'coco', 'val2017-resized'))
        fid_score = fid_score.calculate_fid_given_paths([gen_path, real_path], 50, 'cuda', 2048).item()
        benchmark_results['FID'] = fid_score

    if 'LPIPS' in benchmark_types:
        lpips_model = lpips.LPIPS(net='vgg').to("cuda")
        lpips_list = []
        if args.num != -1: pbar = tqdm.tqdm(range(args.num))
        else: pbar = tqdm.tqdm(range(len(results)))
        with torch.no_grad():
            for i in pbar:
                img_path = os.path.join(args.load_dir, f'{args.load_name}', results[i]["img_path"])
                gt_path = os.path.join(args.datadir, 'coco', 'val2017-resized', results[i]["img_path"])
                
                img1 = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to("cuda")
                img2 = transform(Image.open(gt_path).convert("RGB")).unsqueeze(0).to("cuda")

                lpips_score = lpips_model(img1, img2).item()
                lpips_list.append(lpips_score)
                results[i]['LPIPS'] = lpips_score
        benchmark_results['LPIPS'] = np.mean(lpips_list)
        print('LPIPS: ', np.mean(lpips_list))
    
    with open(os.path.join(args.load_dir, f"metric-{args.load_name}-{num}.json"), "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)  # `indent=4` makes the JSON more readable
    
    with open(os.path.join(args.load_dir, f"benchmark_metric-{args.load_name}-{num}.json"), "w", encoding="utf-8") as file:
        json.dump(benchmark_results, file, indent=4)  # `indent=4` makes the JSON more readable

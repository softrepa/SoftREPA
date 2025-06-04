import argparse
import numpy as np
import os
import yaml, json
import os.path as osp
import pandas as pd
import tqdm
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image

from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from dataset.datasets import get_target_dataset, EncodedDataset

from sampler import SD3EulerDC, SDXLEulerDC, SD1EulerDC
import torchvision.transforms as torch_transforms
from torchvision.transforms.functional import InterpolationMode

from util import setup, cleanup, setup_for_distributed, set_seed, save_on_master
import ImageReward as RM
from eval_utils import PickScore, HPSv2
import torch.distributed as dist

device = "cuda" if torch.cuda.is_available() else "cpu"

# def setup():
#     dist.init_process_group("nccl")
#     rank = dist.get_rank()
#     local_rank = rank  # Now matches the index inside CUDA_VISIBLE_DEVICES
#     torch.cuda.set_device(local_rank)  # Set correct GPU
#     setup_for_distributed(local_rank == 0)
#     return local_rank, dist.get_world_size()

# def cleanup():
#     dist.destroy_process_group()

INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
        
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


def eval(args, model, target_dataset, eval_run_folder, rank, **sample_cfg):
    pbar_eval = tqdm.tqdm(range(rank, args.num_eval, dist.get_world_size()))
    eval_results = [{"prompt": data[-1], "img_path": f'{i:04d}.png'} for i,data in enumerate(target_dataset)]
    with torch.no_grad():
        for vi in pbar_eval:
            _, encoded, label = target_dataset[vi]
            with autocast(enabled=args.dtype == 'float16'):
                img = model.sampler.sample("", prompt_emb=[x.to("cuda") for x in encoded], null_prompt_emb=[x.to("cuda") for x in model.null_embs], **sample_cfg)
            save_image(img, osp.join(eval_run_folder,f'{vi:04d}.png'), normalize=True)
            pbar_eval.set_description(f'SD Evaluation Sampling [{vi}/{args.num_eval}]')

    benchmark_types = args.benchmark.split(",")
    benchmark_types = [x.strip() for x in benchmark_types]
    benchmark_results = {}
    for benchmark_type in benchmark_types:
        print('Benchmark Type: ', benchmark_type)
        eval_model = None
        reward_list = []
        if benchmark_type == "ImageReward-v1.0":
            eval_model = RM.load(name=benchmark_type, device="cuda")
        elif benchmark_type == "PickScore":
            eval_model = PickScore(device="cuda")
        elif benchmark_type == "HPS":
            eval_model = HPSv2()
        elif benchmark_type == "CLIP":
            eval_model = RM.load_score(
                name=benchmark_type, device="cuda"
            )

        with torch.no_grad():
            for vi in range(args.num_eval):
                prompt = eval_results[vi]["prompt"]
                img_path = os.path.join(eval_run_folder, eval_results[vi]["img_path"])

                if benchmark_type in ["ImageReward-v1.0", "PickScore", "HPS"]:
                    rewards = eval_model.score(prompt, [img_path])
                else:
                    _, rewards = eval_model.inference_rank(prompt, [img_path])
                
                if isinstance(rewards, list):
                    rewards = float(rewards[0])

                reward_list.append(rewards)
        reward_list = np.array(reward_list)
        benchmark_results[benchmark_type] = reward_list.mean()
    return benchmark_results


class ContrastiveLoss(nn.Module):
    def __init__(self, temp=0.07, scale=4.0, device='cuda', dweight=0):
        super().__init__()
        self.device = device
        self.temp = torch.nn.Parameter(torch.tensor(temp).to(self.device))
        self.scale = torch.nn.Parameter(torch.tensor(scale).to(self.device))
        self.dweight = dweight

    def get_mask(self, shape=None): # label: [b,], shape: (b, n_p)
        mask = torch.zeros(shape, device=self.device)
        n_b, n_p = shape
        index = torch.arange(n_b, device=self.device)
        mask[index, index] = 1
        return mask # (b, n_p)
        
    def forward(self, errors, labels=None):
        # compute mask
        if labels is None:
            labels = self.get_mask(shape=errors.shape) # (b, n_p)
        # compute logits
        logits = self.scale * torch.exp(-errors/self.temp)
        # compute loss
        loss = F.cross_entropy(logits, labels)
        loss += self.dweight * errors[list(range(labels.shape[0])), labels].mean()
        return loss


class SoftREPA(nn.Module):
    def __init__(self, sampler, device='cuda', dtype='float16'):
        super().__init__()
        self.sampler = sampler
        self.device = device
        self.dtype = dtype
        self.null_embs = self.sampler.encode_prompt([""])

    def forward(self, latent, prompt_embs, t, use_dc=True, img_shape=(512, 512)):
        if self.dtype == 'float16':
            latent = latent.half()

        # append unconditional generation
        n_b,_,h,w = latent.shape
        n_p, n_tkn, n_dim = prompt_embs[0].shape[-3:]
        batch_latent = torch.cat([latent]*n_p, 0)
        batch_pidxs = torch.arange(n_p, device="cuda").unsqueeze(0).repeat(n_b,1).transpose(0,1).contiguous().reshape(-1)

        # set noise and timestep
        self.sampler.set_noise(img_shape=img_shape, batch_size=1)
        batch_nidxs = torch.zeros(n_p*n_b, device="cuda").long().contiguous()
        v, pred_v = self.sampler.error(batch_latent, batch_nidxs, batch_pidxs, prompt_embs, t, use_dc=use_dc)
        error = F.mse_loss(v, pred_v, reduction='none').mean(dim=(1, 2, 3))
        
        return error.reshape(n_p, n_b).transpose(0,1) #(b, n_p)


def main():
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument('--dataset', nargs='+', type=str, default=['coco'], choices=['coco'], help='Dataset to use')
    parser.add_argument('--target_dataset', type=str, default='coco', choices=['coco'], help='Dataset to use')
    parser.add_argument('--interpolation', type=str, default='bicubic', help='Resize interpolation type')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of workers to split the dataset across')

    # run args
    parser.add_argument('--model', type=str, default='sd3', choices=['sd3', 'sdxl', 'sd1.5'], help='Model to use')
    parser.add_argument('--n_dc_tokens', type=int, default=4, help='the number of learnable dc tokens')
    parser.add_argument('--n_dc_layers', type=int, default=5, help='the number of layers to append dc_tokens (sd3)')
    parser.add_argument('--apply_dc', nargs='+', type=str, default=[True, True, False], help='down/mid/up layers of unet (sd1.5, sdxl)')
    parser.add_argument('--use_dc_t', action='store_true', default=False, help='t dependent tokens')
    parser.add_argument('--max_t', type=int, default=1000)
    parser.add_argument('--min_t', type=int, default=0)
    parser.add_argument('--dweight', type=float, default=0, help='weight of diffusion score matching loss')

    # text embedding encoding batchsize before distributed training
    parser.add_argument('--batch_size_for_encodings', type=int, default=16)

    # training args
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3, help='train learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--img_size', type=int, default=512, choices=(256, 512, 768, 1024), help='training image size')
    parser.add_argument('--batch_size', '-b', type=int, default=4)
    parser.add_argument('--nrows', type=int, default=1, help='number of rows per backward')
    parser.add_argument('--dtype', type=str, default='float16', choices=('float16', 'float32'),
                        help='Model data type to use')
    parser.add_argument('--device', type=str, default='cuda', choices=('cuda', 'cpu'))

    # save/eval args
    parser.add_argument('--logdir', type=str, default='./data', help='path for save checkpoint')
    parser.add_argument('--datadir', type=str, default='', required=True, help='data path')
    parser.add_argument('--num_iter', type=int, default=2500, help='number of iterations before validation')
    parser.add_argument('--num_eval', type=int, default=50, help='number of generating images during validation')
    parser.add_argument('--benchmark', default="ImageReward-v1.0, CLIP, PickScore", type=str,
                        help="ImageReward-v1.0, Aesthetic, BLIP or CLIP, PickScore, HPS splitted with comma(,) if there are multiple benchmarks.")
    parser.add_argument('--note', type=str, default=None, help='note for saving path')

    # multi gpus
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--use_8bit', action='store_true', default=False, help='Use 8bit quantization for T5 and transformer.')
    
    args = parser.parse_args()
    args.apply_dc = [str2bool(x) for x in args.apply_dc]

    # setup
    set_seed(42)
    rank, world_size = setup()

    args.device = "cuda"

    # make run output folder
    name = f"dist-{args.model}"
    if args.img_size != 512:
        name += f'_{args.img_size}'
    name += f'_np{args.n_dc_tokens}'
    name += f'_nl{args.n_dc_layers}'
    name += f'_usedct{args.use_dc_t}'
    if args.note != None:
        name += f'_{args.note}'
    run_folder = osp.join(args.logdir, "-".join(args.dataset), name)
    os.makedirs(run_folder, exist_ok=True)
    print(f'Run folder: {run_folder}')

    # save arguments to a YAML file
    with open(os.path.join(run_folder, 'config.yaml'), 'w') as f:
        yaml.dump(vars(args), f)
    print('Arguments saved to config.yaml')

    needs_encoding = {d: not os.path.exists(os.path.join(f"encodings_{args.model}", d)) for d in args.dataset}

    # set up dataset for train
    interpolation = INTERPOLATIONS[args.interpolation]
    transform = get_transform(interpolation, args.img_size)

    train_datasets = {}
    dataloaders = {}
    for ds in args.dataset:
        if needs_encoding[ds]:
            train_dataset = get_target_dataset(ds, args.datadir, train=True, transform=transform)

            # if torch.cuda.device_count()>1 and not args.separate_gpus:
            distributed_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=False)
            dataloader = torch.utils.data.DataLoader(
                        train_dataset, 
                        sampler=distributed_sampler,
                        batch_size=args.batch_size_for_encodings,
                        num_workers=args.n_workers)
            train_datasets[ds] = train_dataset
            dataloaders[ds] = dataloader
    
    # set up dataset for eval
    target_dataset = get_target_dataset(args.target_dataset, args.datadir, train=False, transform=transform)

    # load pretrained models
    if args.model == 'sd3':
        sampler = SD3EulerDC(n_dc_tokens=args.n_dc_tokens, n_dc_layers=args.n_dc_layers, device="cpu", use_dc_t=args.use_dc_t, use_8bit=args.use_8bit)
        sample_cfg = {'NFE':28, 'img_shape':(1024,1024), 'cfg_scale':4, 'use_dc':True}
        sampler.text_enc_1.to("cuda")
        sampler.text_enc_2.to("cuda")
        sampler.text_enc_3.to("cuda")
        sampler.vae.to("cuda")
    elif args.model == 'sdxl':
        sampler = SDXLEulerDC(n_dc_tokens=args.n_dc_tokens, use_8bit=args.use_8bit, device="cpu", use_dc_t=args.use_dc_t, apply_dc=args.apply_dc)
        sample_cfg = {'NFE':30, 'img_shape':(1024,1024), 'cfg_scale':7.0, 'use_dc':True} 
        sampler.text_enc.to("cuda")
        sampler.text_enc_2.to("cuda")
        sampler.vae.to("cuda")
        sampler.denoiser.to("cuda")
    elif args.model == 'sd1.5':
        sampler = SD1EulerDC(n_dc_tokens=args.n_dc_tokens, use_8bit=args.use_8bit, device="cpu", use_dc_t=args.use_dc_t, apply_dc=args.apply_dc)
        sample_cfg = {'NFE':30, 'img_shape':(512,512), 'cfg_scale':7.0, 'use_dc':True} 
        sampler.text_enc.to("cuda")
        sampler.vae.to("cuda")
        sampler.denoiser.to("cuda")
        with torch.no_grad():
            null_embs = sampler.encode_prompt([""])[0][0, :1].expand(args.n_dc_tokens, -1)
            sampler.initialize_dc(null_embs)

    # make target dataset
    with torch.no_grad():
        encoded_targets = []
        pbar_eval = tqdm.tqdm(range(args.num_eval))

        for vi in pbar_eval:
            img, label = target_dataset[vi]
            prompt_emb = sampler.encode_prompt([label])
            encoded_targets.append((img, [pe.to("cpu") for pe in prompt_emb], label))
        
        target_dataset = encoded_targets

    dist.barrier()

    # encoding dataset
    for ds in args.dataset:
        if needs_encoding[ds]:
            print(f"Encoding: {ds}")
            if not os.path.exists(os.path.join(f"encodings_{args.model}", f"{ds}_rank{rank}")):
                os.makedirs(os.path.join(f"encodings_{args.model}", f"{ds}_rank{rank}"), exist_ok=True)
                with torch.no_grad():
                    print("Start Encoding...")
                    pbar = tqdm.tqdm(dataloaders[ds])
                    didx = 0
                    for i, (image, label) in enumerate(pbar):
                        image = image.to("cuda")
                        if args.dtype == "float16":
                            image = image.half()
                        latent = sampler.encode(image)
                        prompt_embs = sampler.encode_prompt(label)
                        for bidx in range(latent.shape[0]):
                            torch.save(latent[bidx], os.path.join(f"encodings_{args.model}", f"{ds}_rank{rank}", f"latent_{didx}.pt"))
                            torch.save(prompt_embs[0][bidx], os.path.join(f"encodings_{args.model}", f"{ds}_rank{rank}", f"prompt_emb_{didx}.pt"))
                            if len(prompt_embs)==2:
                                torch.save(prompt_embs[1][bidx], os.path.join(f"encodings_{args.model}", f"{ds}_rank{rank}", f"pooled_prompt_emb_{didx}.pt"))
                            didx += 1

            dist.barrier()
            if rank == 0:
                didx = 0
                os.makedirs(os.path.join(f"encodings_{args.model}", ds), exist_ok=True)
                for r in range(dist.get_world_size()):
                    for eidx in range(len(glob.glob(osp.join(f"encodings_{args.model}", f"{ds}_rank{r}", "latent_*.pt")))):
                        os.rename(os.path.join(f"encodings_{args.model}", f"{ds}_rank{r}", f"latent_{eidx}.pt"),
                                os.path.join(f"encodings_{args.model}", ds, f"latent_{didx}.pt"))
                        os.rename(os.path.join(f"encodings_{args.model}", f"{ds}_rank{r}", f"prompt_emb_{eidx}.pt"),
                                os.path.join(f"encodings_{args.model}", ds, f"prompt_emb_{didx}.pt"))
                        if len(prompt_embs)==2:
                            os.rename(os.path.join(f"encodings_{args.model}", f"{ds}_rank{r}", f"pooled_prompt_emb_{eidx}.pt"),
                                    os.path.join(f"encodings_{args.model}", ds, f"pooled_prompt_emb_{didx}.pt"))
                        didx += 1

    dist.barrier()
    print("Initiallizing dataset")

    encoded_datasets = []
    for ds in args.dataset:
        encoded_dataset = EncodedDataset(os.path.join(f"encodings_{args.model}", ds), model=args.model)
        encoded_datasets.append(encoded_dataset)
    
    encoded_dataset = ConcatDataset(encoded_datasets)

    distributed_sampler = DistributedSampler(encoded_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = torch.utils.data.DataLoader(
                    encoded_dataset, 
                    sampler=distributed_sampler,
                    batch_size=args.batch_size,
                    num_workers=args.n_workers,
                    # shuffle=True,
                    )

    print("Constructing model")
    model = SoftREPA(sampler, device=args.device, dtype=args.dtype)
    scaler = GradScaler() if args.dtype == 'float16' else None
    loss_criterion = ContrastiveLoss(device=sampler.denoiser.device)
    if args.model == 'sd3':
        del sampler.text_enc_1
        del sampler.text_enc_2
        del sampler.text_enc_3
        sampler.vae.to("cpu", non_blocking=True)
        sampler.denoiser.to("cuda")
        model_device=sampler.denoiser.device
    elif args.model == 'sdxl':
        del sampler.text_enc
        del sampler.text_enc_2
        sampler.vae.to("cpu", non_blocking=True)
        sampler.denoiser.to("cuda")
        model_device=sampler.denoiser.device
    elif args.model == 'sd1.5':
        del sampler.text_enc
        sampler.vae.to("cpu", non_blocking=True)
        sampler.denoiser.to("cuda")
        model_device=sampler.denoiser.device

    # set requires grad
    print("Enable DDP")
    # update_param_set = []
    for name, param in model.sampler.denoiser.named_parameters():
        if 'dc' in name:
            param.requires_grad = True
            print(f'param: {name} requires grad [True]')
            # update_param_set.append({'params':param})
        else:
            param.requires_grad = False
    model.sampler.denoiser = nn.parallel.DistributedDataParallel(model.sampler.denoiser, device_ids=[rank])
    print("Set Optimizer")
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.sampler.denoiser.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.wd)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-5) # T_0=20, T_mult=2

    # multi gpu
    print('Using {} GPUs'.format(torch.cuda.device_count()))
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.sampler.denoiser.parameters() if p.requires_grad)}")
    model.sampler.denoiser.train()

    save_dict = {'epoch':[], 'loss':[], 'lr':[]}
    benchmark_types = args.benchmark.split(",")
    benchmark_types = [x.strip() for x in benchmark_types]
    for bch in benchmark_types:
        save_dict[bch] = []
    # train
    iteration = 0
    global_loss = 0.0
    for ep in range(args.epochs):
        dataloader.sampler.set_epoch(ep)
        pbar = tqdm.tqdm(dataloader)
        for i, (latent, prompt_embs) in enumerate(pbar):
            iteration += 1
            optimizer.zero_grad()
            latent = latent.to("cuda")
            prompt_embs = [pe.to('cuda') for pe in prompt_embs]
            aggregated_loss = 0
            t = torch.randint(args.min_t, args.max_t, (1,), device=args.device)
            for j in range(0, latent.shape[0], args.nrows):
                with torch.autocast("cuda", enabled=args.dtype == 'float16'):
                    errors = model(latent[j:min(latent.shape[0], j+args.nrows)], prompt_embs, t.long(), img_shape=(args.img_size, args.img_size))
                    loss = loss_criterion(errors, torch.tensor(list(range(j, min(latent.shape[0], j+args.nrows))), dtype=torch.long, device="cuda")) / latent.shape[0]
                    aggregated_loss += loss.item()
                scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_loss += loss.item()
            pbar.set_description(f'Loss: {aggregated_loss:.4f}')

            # validation
            # if iteration % args.num_iter == 0 and rank==0: 
            if iteration % args.num_iter == 0: 
                model.sampler.vae.to("cuda")
                model.sampler.vae.eval()
                model.sampler.denoiser.eval()
                it_ep = iteration // args.num_iter
                # save model
                global_loss /= args.num_iter
                print(f'Epoch {ep} Iteration {iteration}: Loss: {global_loss:.4f}')
                save_on_master(sampler.denoiser.module.dc_tokens, osp.join(run_folder, f'dc_tokens_{it_ep}.pth'))
                if args.use_dc_t:
                    save_on_master(sampler.denoiser.module.dc_t_tokens, osp.join(run_folder, f'dc_t_tokens_{it_ep}.pth'))
               
                # evaluate
                with torch.no_grad():
                    eval_run_folder = osp.join(run_folder, f'val_samples_{it_ep}')
                    if rank==0:
                        os.makedirs(eval_run_folder, exist_ok=True)

                    benchmark_results = eval(args, model, target_dataset, eval_run_folder, rank, **sample_cfg)

                    for bch,result in benchmark_results.items():
                        save_dict[bch].append(result) 
                    if rank==0:
                        save_dict['epoch'].append(it_ep)
                        save_dict['loss'].append(global_loss)
                        current_lr = optimizer.param_groups[0]['lr']
                        save_dict['lr'].append(current_lr)
                        df = pd.DataFrame(save_dict)
                        df.to_csv(os.path.join(run_folder, 'run.csv'), index=False)

                model.sampler.denoiser.train()
                model.sampler.vae.to("cpu")

                # reset loss
                global_loss = 0.0
                lr_scheduler.step()
            dist.barrier()

        pbar.close()

    print(f'Training complete. Saving model to {run_folder}')
    cleanup()

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()

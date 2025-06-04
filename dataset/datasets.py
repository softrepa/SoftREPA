import json
import os, glob
import os.path as osp
import torch
from torchvision import datasets
from PIL import Image
from torch.utils.data import Dataset
import torchvision.datasets as datasets


class COCODataset(Dataset):
    def __init__(self, root, train, annFile, transform=None):
        self.root = root
        self.train = train

        self.dataset = datasets.CocoCaptions(root=root, annFile=annFile, transform=transform)

    def __getitem__(self, index):
        img, captions = self.dataset[index]
        if self.train:
            cindex = torch.randint(0, len(captions), (1,)).item()
        else:
            cindex = torch.zeros((1,)).type(torch.int32)
        caption = captions[cindex]

        return img, caption

    def __len__(self):
        return len(self.dataset)
        # return 100

class EncodedDataset(Dataset):
    def __init__(self, encoding_dir, model='sd3'):
        super().__init__()
        self.encoding_dir = encoding_dir
        self.data_len = len(glob.glob(osp.join(self.encoding_dir, "latent_*.pt")))
        self.model = model
    
    def __len__(self):
        return self.data_len
        
    def __getitem__(self, index):
        latent = torch.load(osp.join(self.encoding_dir, f"latent_{index}.pt"), map_location="cpu")
        if len(latent.shape) == 4:
            latent = latent[0]
        latent = latent.detach()

        prompt_emb = torch.load(osp.join(self.encoding_dir, f"prompt_emb_{index}.pt"), map_location="cpu")
        if len(prompt_emb.shape) == 3:
            prompt_emb = prompt_emb[0]
        prompt_emb = prompt_emb.detach()
        if self.model != 'sd1.5':
            pooled_prompt_emb = torch.load(osp.join(self.encoding_dir, f"pooled_prompt_emb_{index}.pt"), map_location="cpu")
            if len(pooled_prompt_emb.shape) == 2:
                pooled_prompt_emb = pooled_prompt_emb[0]
            pooled_prompt_emb = pooled_prompt_emb.detach()
            return latent, (prompt_emb, pooled_prompt_emb)
        else:
            return latent, (prompt_emb, )


def get_target_dataset(name: str, datadir, train=False, transform=None):
    if name == 'coco':
        # import ipdb; ipdb.set_trace();
        datapath = os.path.join(datadir, 'coco', 'train2017') if train else os.path.join(datadir, 'coco', 'val2017')
        annpath = os.path.join(datadir, 'coco', 'annotations', 'captions_train2017.json') if train else os.path.join(datadir, 'coco', 'annotations', 'captions_val2017.json')
        dataset = COCODataset(root=datapath, train=train, annFile=annpath, transform=transform,)
    else:
        raise ValueError(f"Dataset {name} not supported.")

    return dataset

# Copyright(c) 2022 Liang Zhang 
# E-Mail: <zhangliang00@ruc.edu.cn>

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json

import torch
import numpy as np
import torch.tensor as Tensor

from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import random


class TxtReader(object):
    def __init__(self, txt_path, max_sents=-1):
        self.lines = open(txt_path, 'r', encoding='utf-8').readlines()
        self.lines = [line.strip() for line in self.lines]
        if max_sents != -1:
            self.lines = self.lines[:max_sents]
    
    def __getitem__(self, idx):
        return self.lines[idx]
    
    def __len__(self):
        return len(self.lines)

class TextDataset(Dataset):
    def __init__(self, src_reader, src_toker, max_len=50):
        super().__init__()
        self.src = src_reader
        self.src_toker = src_toker
        self.max_len = max_len

    def __getitem__(self, idx):
        src_sent = self.src[idx]
        src_dict = self.src_toker(src_sent, truncation=True, max_length=self.max_len, padding='max_length')
        src_ids = src_dict['input_ids']
        src_mask = src_dict['attention_mask']
        
        # import pdb;pdb.set_trace()

        return Tensor(src_ids), Tensor(src_mask)

        # return {
        #     'src_ids': src_ids,
        #     'src_mask': src_mask,
        #     'trg_ids': trg_ids,
        #     'trg_mask': trg_mask
        # }

    def __len__(self):
        return len(self.src)

class PairTextDataset(Dataset):
    def __init__(self, src_reader, trg_reader, src_toker, trg_toker, max_len=50):
        super().__init__()
        self.src = src_reader
        self.trg = trg_reader
        self.src_toker = src_toker
        self.trg_toker = trg_toker
        self.max_len = max_len
        assert len(self.src) == len(self.trg)

    def __getitem__(self, idx):
        src_sent = self.src[idx]
        trg_sent = self.trg[idx]
        # src_dict = self.src_toker(src_sent, truncation=True, max_length=self.max_len, padding='max_length')
        src_ids = self.src_toker(src_sent, truncate=True).squeeze(dim=0)
        trg_dict = self.trg_toker(trg_sent, truncation=True, max_length=self.max_len, padding='max_length')
        # src_ids = src_dict['input_ids']
        # src_mask = src_dict['attention_mask']
        src_mask = [1]
        trg_ids = trg_dict['input_ids']
        trg_mask = trg_dict['attention_mask']
        token_type_ids = trg_dict['token_type_ids']
        
        # import pdb;pdb.set_trace()

        return src_ids, Tensor(src_mask), Tensor(trg_ids), Tensor(trg_mask), Tensor(token_type_ids), src_sent

        # return {
        #     'src_ids': src_ids,
        #     'src_mask': src_mask,
        #     'trg_ids': trg_ids,
        #     'trg_mask': trg_mask
        # }

    def __len__(self):
        return len(self.src)
    

class ImageTextDataset(Dataset):
    def __init__(self, names, json_file, image_dir, preprocess, toker=None, sentence_transformer=False, return_type='tuple'):
        self.names = np.load(names)
        self.data_dict = json.load(open(json_file, 'r', encoding='utf-8'))
        self.preprocess = preprocess
        self.toker = toker
        self.return_type = return_type
        self.imgpath2imgid = {}

        self.sentence_transformer = sentence_transformer

        self.multi_sentence_per_video = True
        self.cut_off_points = []

        self.pairs = []
        not_found = []
        for name in self.names:
            if isinstance(name, np.bytes_):
                name = name.decode('utf-8')
            # try:
            image_path = os.path.join(image_dir, name)
            # except:
                # import pdb;pdb.set_trace()
            if not os.path.isfile(image_path):
                not_found.append(name)
                continue
            self.imgpath2imgid[image_path] = name
            #     import pdb;pdb.set_trace()
            # if not os.path.isfile(image_path):
            #     import pdb;pdb.set_trace()
            # assert os.path.isfile(image_path)
            # if name not in self.data_dict:
            #     not_found.append(name)
            #     continue
            if name not in self.data_dict:
                import pdb;pdb.set_trace()
            for i, caption in enumerate(self.data_dict[name]):
                self.pairs.append((image_path, caption, i))

            self.cut_off_points.append(len(self.pairs))
        
        self.video_num = len(self.names)
        self.sentence_num = len(self.pairs)

        print(f'Total image-text pairs: {len(self.pairs)}')
        print(f'Not found image: {len(not_found)}')

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        image_path, caption, i = self.pairs[idx]
        image_id = self.imgpath2imgid[image_path]
        if self.sentence_transformer:
            # img = Image.open(image_path)
            return image_path, image_path, caption
        img = self.preprocess(Image.open(image_path))
        # import pdb;pdb.set_trace()
        if self.toker:
            input_ids = self.toker(caption, truncate=True).squeeze(dim=0)
            if self.return_type == 'tuple':
                return img, input_ids, caption
            elif self.return_type == 'dict':
                return {
                    'img': img,
                    'image_id': image_id,
                    'input_ids': input_ids,
                    'caption': caption,
                    'caption_id': image_id+'#'+str(i)
                }
        else:
            if self.return_type == 'tuple':
                return img, caption
            elif self.return_type == 'dict':
                return {
                    'img': img,
                    'image_id': image_id,
                    'caption': caption,
                    'caption_id': image_id+'#'+str(i)
                }

class MultilingualImageTextDataset(Dataset):
    def __init__(self, names, json_files, langs, image_dir, preprocess, top=-1, train_en=False):
        self.names = np.load(names)
        self.data_dict = {}
        self.langs = langs
        self.trg_langs = [lang for lang in langs if lang != 'en']
        if train_en:
            self.trg_langs = self.trg_langs + ['en']
        print(f"Target languages: {self.trg_langs}")
        for lang, json_file in zip(langs, json_files):
            self.data_dict[lang] = json.load(open(json_file, 'r', encoding='utf-8'))
        
        not_found = set()
        self.id2path = {}
        all_image_names = set(json.load(open('dataset/ConceptualCaption/existing_ids.json')))
        for name in self.names:
            img_path = os.path.join(image_dir, name)
            # if not os.path.isfile(img_path):
            #     not_found.add(name)
            if name not in all_image_names:
                not_found.add(name)
            else:
                self.id2path[name] = img_path
        del all_image_names
        print(f'Images not found: {len(not_found)}')
        self.names = [name for name in self.names if name not in not_found]
        if top > 0:
            self.names = self.names[:top]
        self.preprocess = preprocess
        self.image_num = len(self.names)
        print(f'Total images: {self.image_num}')
        
    def __len__(self):
        return self.image_num
    
    def __getitem__(self, idx):
        img_id = self.names[idx]
        # lang = random.choice(self.langs)
        lang = random.choice(self.trg_langs)
        caption_trg = self.data_dict[lang][img_id][0]
        caption_en = self.data_dict['en'][img_id][0]
        image_path = self.id2path[img_id]
        img = self.preprocess(Image.open(image_path))
        return img, caption_en, caption_trg

def get_mit_loader(args, names, json_files, langs, image_dir, preprocess, is_train, top=-1, train_en=False):
    dataset = MultilingualImageTextDataset(names, json_files, langs, image_dir, preprocess, top=top, train_en=train_en)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=is_train,
        num_workers=args.n_workers,
        drop_last=is_train
    )
    return dataset, dataloader

def get_it_loader(args, names, json_file, image_dir, preprocess, toker, is_train=True, st=False, return_type='tuple'):
    dataset = ImageTextDataset(
        names, json_file, image_dir, preprocess, toker, sentence_transformer=st, return_type=return_type
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=is_train,
        num_workers=args.n_workers,
        drop_last=is_train
    )
    return dataset, dataloader
    

def get_pair_dataloader(args, src_paths, trg_paths, src_toker, trg_toker, max_len, max_sents=-1):
    datasets = []
    assert len(src_paths) == len(trg_paths)

    for src_path, trg_path in zip(src_paths, trg_paths):
        src_reader = TxtReader(src_path, max_sents=max_sents)
        trg_reader = TxtReader(trg_path, max_sents=max_sents)
        datasets.append(PairTextDataset(src_reader, trg_reader, src_toker, trg_toker, max_len))
    pair_dataset = ConcatDataset(datasets)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(pair_dataset)

    train_dataloader = DataLoader(
        dataset=pair_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        # sampler=train_sampler,
        num_workers=args.n_workers,
        drop_last=True,
    )
    return pair_dataset, train_dataloader

def get_textloader(args, paths, toker, max_len):
    datasets = []
    for path in paths:
        reader = TxtReader(path)
        datasets.append(TextDataset(reader, toker, max_len))
    dataset = ConcatDataset(datasets)

# def multiloader(dataloaders):
#     while True:
#         batchs = []
#         for loader in dataloaders:
#             try:
#                 batchs = iter(loader)
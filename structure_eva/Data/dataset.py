import os
import random
import numpy as np
import pandas as pd
from os.path import join, exists
from torch.utils.data import Dataset, DataLoader
import multiprocessing
import torch
import pytorch_lightning as pl
from functools import partial

class DecoyDataset(Dataset):
    def __init__(self,
                 targets_fpath='',
                 feature_dir='',
                 interface_pdb_base_dir='',
                 interface_fe_base_dir='',
                 interface_emb_base_dir='',
                 process_feat=False,
                 max_length=999,
                 pool_process=4,
                 infer=False,
                 sample_num=1):

        self.samples = []
        self.target_models_dict = {}
        self.target_list = []
        self.interface_fpath_list = []

        self.feature_dir = feature_dir
        self.max_length = max_length
        self.infer = infer
        self.process_feat = process_feat
        self.sample_num = sample_num

        self.interface_pdb_base_dir = interface_pdb_base_dir
        self.interface_fe_base_dir = interface_fe_base_dir
        self.interface_emb_base_dir = interface_emb_base_dir

        # Read target/model list
        with open(targets_fpath, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                smp = parts[0]
                score = float(parts[1]) if len(parts) == 2 else 0.0

                target, model = smp.split("/")[-2:]
                self.target_models_dict.setdefault(target, []).append([smp, score])
                self.target_list.append(target) if target not in self.target_list else None

                inter_pdb_path = join(self.interface_pdb_base_dir, target, model)
                self.interface_fpath_list.append((smp, inter_pdb_path))
                self.samples.append([smp, score])

        print(f"Total samples: {len(self.samples) if infer else len(self.target_list)}")

        if process_feat:
            pool = multiprocessing.Pool(pool_process)
            fv_cmd = []
            for smp, inter in self.interface_fpath_list:
                target, model = smp.split("/")[-2:]
                feat_path = join(self.interface_fe_base_dir, target, f"{model}.features.npz")

                os.makedirs(os.path.dirname(inter), exist_ok=True)
                os.makedirs(os.path.dirname(feat_path), exist_ok=True)
                fv_cmd.append((smp, inter, feat_path))

            pool.map(self.prepare_feature_multprocess, fv_cmd)

    def __len__(self):
        return len(self.samples) if self.infer else len(self.target_list)

    def __getitem__(self, idx):
        try:
            return self.load_data_feature(idx)
        except Exception as e:
            return {"error": 0}

    def get_samples(self):
        return self.samples if self.infer else self.target_list

    def prepare_feature_multprocess(self, args):
        from structure_eva.utils.featurizer import process
        structure_fpath, inter_pdb_path, output_feat_path = args
        process((structure_fpath, inter_pdb_path, output_feat_path))

    def load_data_feature(self, idx):
        if self.infer:
            smp_fpath, quality = self.samples[idx]
        else:
            target_name = self.target_list[idx]
            smp_fpath, quality = random.sample(self.target_models_dict[target_name], self.sample_num)[0]

        target_model = "/".join(smp_fpath.split("/")[-2:])
        feat_path = join(self.interface_fe_base_dir, f"{target_model}.features.npz")
        data = np.load(feat_path, encoding='bytes', allow_pickle=True)

        if data["obt"].shape[1] > self.max_length:
            return {"name": smp_fpath}

        # 1D features
        angles = np.stack([
            np.sin(data["phi"]), np.cos(data["phi"]),
            np.sin(data["psi"]), np.cos(data["psi"])
        ], axis=-1)

        mie_mee = np.concatenate([
            angles.transpose(1, 0),
            data["obt"],
            data["prop"][:52]
        ], axis=0)  # 20:44 from BLOSUM62 of mee

        # 2D features
        orientations = np.stack([data["omega6d"], data["theta6d"], data["phi6d"]], axis=-1)
        orientations = np.concatenate([np.sin(orientations), np.cos(orientations)], axis=-1)
        euler = np.concatenate([np.sin(data["euler"]), np.cos(data["euler"])], axis=-1)
        mae = np.concatenate([
            data["tbt"].transpose(1, 2, 0),
            data["maps"],
            euler,
            orientations
        ], axis=-1)

        return {
            'mie_mee': mie_mee.astype(np.float32),
            'mae': mae.astype(np.float32),
            'mee_vidx': data["idx"].astype(np.int32),
            'mee_val': data["val"].astype(np.float32),
            'mie_feat': data["feat"][0].astype(np.float32),
            'mae_adj': data["adj"].astype(np.float32),
            'quality': quality,
            'name': smp_fpath,
        }


# =============================
# Utility: Dict Stack Mapping
# =============================
def dict_multimap(fn, dicts):
    first = dicts[0]
    return {
        k: dict_multimap(fn, [d[k] for d in dicts]) if isinstance(v, dict) else fn([d[k] for d in dicts])
        for k, v in first.items()
    }


class BatchCollator:
    def __call__(self, prots):
        return dict_multimap(partial(torch.stack, dim=0), prots)


# =============================
# Lightning DataModule
# =============================
class ScoreDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_collator = BatchCollator()

    def setup(self, stage=None):
        self.val_dataset = DecoyDataset(
            targets_fpath=self.config.val_fpath,
            interface_pdb_base_dir=self.config.interface_pdb_base_dir,
            interface_fe_base_dir=self.config.interface_fe_base_dir,
            process_feat=self.config.process_feat,
            max_length=self.config.max_length,
            pool_process=self.config.pool_process
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.config.num_workers,
        )

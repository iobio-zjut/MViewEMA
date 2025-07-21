import os
import argparse
import warnings
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from os.path import join, exists
from structure_eva.Data.dataset import DecoyDataset
from structure_eva.Model.ModelNet import Score_m

warnings.filterwarnings("ignore")

def parse_config():
    parser = argparse.ArgumentParser(description="Inference pipeline for structure ranking")

    parser.add_argument("--infer", default=True, help="Whether to run in inference mode")
    parser.add_argument("--test_fpath", type=str, required=True, help="Path to test targets file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for DataLoader")
    parser.add_argument("--only_process_feat", "-f", action="store_true", help="Only process features without inference")
    parser.add_argument("--process_feat", action="store_true", default=True, help="Whether to process features")
    parser.add_argument("--max_length", type=int, default=999, help="Max sequence length")

    return parser.parse_args()

def main():

    config = parse_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Create output folders
    pdb_dir = join(config.output, "overall_pdb")
    feat_dir = join(config.output, "overall_feature")
    tm_dir = join(config.output, "overall_tm")
    os.makedirs(pdb_dir, exist_ok=True)
    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(tm_dir, exist_ok=True)

    config.interface_pdb_base_dir = pdb_dir
    config.interface_fe_base_dir = feat_dir

    # Load dataset
    test_dataset = DecoyDataset(
        targets_fpath=config.test_fpath,
        process_feat=config.process_feat,
        max_length=config.max_length,
        interface_pdb_base_dir=config.interface_pdb_base_dir,
        interface_fe_base_dir=config.interface_fe_base_dir,
        infer=config.infer,
    )

    # Feature-only mode
    if config.only_process_feat:
        print(f'Feature processing completed in {time.time() - start:.2f}s')
        return

    valid_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
    )

    # Load model
    model = Score_m()
    state_dict = torch.load(config.ckpt_path, map_location=device)['state_dict']
    state_dict = {k[6:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    score_dict = {}

    with torch.no_grad():
        for data in tqdm(valid_dataloader, desc="Scoring models"):
            name = data["name"][0]

            # Skip if invalid input
            if "score" in data or len(data.keys()) == 1:
                continue

            # try:
            data = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()}
            output = model(data)
            score = float(output.cpu().numpy().squeeze())
            # except Exception as e:
            #     with open(join(config.output, "err.txt"), "a") as f:
            #         f.write(f"{name}\n")
            #     continue

            target = name.split("/")[-2]
            model_name = name.split("/")[-1]
            score_dict.setdefault(target, []).append([model_name, score])

    # Save results
    for target, scores in score_dict.items():
        df = pd.DataFrame(scores, columns=["name", "score"])
        df.to_csv(join(tm_dir, f"{target}.csv"), index=False)

    print(f"Predicted TM scores saved to: {tm_dir}")

if __name__ == "__main__":
    main()

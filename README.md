# MViewEMA: Efficient Global Accuracy Estimation for Protein Complex Structural Models Using Multi-View Representation Learning

**MViewEMA** is a global accuracy estimation method for protein complex models.  
It provides a TM-score-based confidence score to assess the accuracy of structural models, particularly for complexes and multimeric assemblies, enabling efficient estimation and model selection in large-scale structural sampling scenarios.

---
## 🧪 Method Overview

This method evaluates the overall quality of input protein models by generating a global confidence score based on TM-score, leveraging multi-view representation learning for robust and efficient estimation.

---

## 🚀 Getting Started

### 🔧 Software Requirements

To run this project, you need the following dependencies installed **(or use the provided Singularity container)**:

- Python ≥ 3.8  
- PyTorch 1.11.0  
- PyTorch Geometric 2.0.4  
- PyRosetta ≥ 2021.38+release.4d5a969  

Alternatively, you can run everything inside a [Singularity container](http://zhanglab-bioinf.com/DeepUMQA-X/static/env.sif) to avoid dependency issues (container size: 7.31 GB).

---


## 🏃 Running the Pipeline

### 📌 Command-Line Usage

```
usage: run_inf.py [--infer INFER] --test_fpath TEST_FPATH --output OUTPUT --ckpt_path CKPT_PATH
                  [--num_workers NUM_WORKERS] [--only_process_feat] [--process_feat] [--max_length MAX_LENGTH]

Inference pipeline for structure estimation

arguments:
  --infer INFER              Whether to run in inference mode (default: True)
  --test_fpath TEST_FPATH    Path to input target list (required)
  --output OUTPUT            Path to output directory (required)
  --ckpt_path CKPT_PATH      Path to the model checkpoint (required)
  --num_workers NUM_WORKERS  Number of workers for data loading (default: 1)
  --only_process_feat, -f    Only process features without running inference (default: False)
  --process_feat             Whether to process features (default: True)
  --max_length MAX_LENGTH    Maximum sequence length allowed (default: 999)
```
💡 Example: Run on a folder of PDB files
```
python run_inf.py --test_fpath ${input_pdbs_list} --ckpt_path ${tm_ckpt} --output ${out_dir}
```
🐳 Option : Using Singularity
```
# Launch the Singularity container and bind your project directory
singularity shell --nv \
  --bind /path/to/project:/mnt/project \
  /path/to/env.sif

# Inside the container, activate the conda environment
source /miniconda/bin/activate
conda activate pytorch

# Run the inference script
python /mnt/project/run_inf.py \
  --test_fpath /mnt/project/${input_pdbs_list} \
  --ckpt_path /mnt/project/${tm_ckpt} \
  --output /mnt/project/${out_dir}
```
📂 Output
```[out_dir].csv```
Each row corresponds to a predicted protein model and its estimated TM-score-based confidence.

### 🛠️ Troubleshooting

OOM (Out of Memory) Errors:
If you encounter memory errors during inference, it is likely due to the size of the input complex. Try the following:

Use a machine with more GPU memory.

Disable GPU and run on CPU (slower but more stable for large models).

Still having issues?
Contact us at 📧 guijunlab06@163.com

### 📚 Resources
[CASP15 EMA Data](https://predictioncenter.org/download_area/CASP15/):
Includes predicted models, experimental structures, and EMA results.

[CASP16 EMA Data](https://predictioncenter.org/download_area/CASP16/)
Includes predicted models, experimental structures, and EMA results.

[PDB-2024 Targets](https://www.rcsb.org/): Estimation for docking models.

# RAPPPID

***R**egularised **A**utomative **P**rediction of **P**rotein-**P**rotein **I**nteractions using **D**eep Learning*

---

RAPPPID is a deep learning model for predicting protein interactions. You can 
read more about it in the OUP Bionformatics [paper](https://doi.org/10.1093/bioinformatics/btac429).

## How to Use RAPPPID

### Training New Models
It's possible to train a RAPPPID model using the `train.py` utility. For precise instructions, see [docs/train.md](docs/train.md).

### Data
See [docs/data.md](docs/data.md) for information about downloading data from the manuscript, pre-trained weights, or preparing your own datasets.

### Infering
See [docs/infer.md](docs/infer.md) for advice on how to use RAPPPID for infering protein interaction probabilities.

## Environment/Requirements

### Setup Environment

For all NVIDIA GPUs with CUDA support, use the modern environment:

```bash
conda env create -f environment_modern.yml
conda activate rapppid_modern
```

**GPU Requirements:**
- NVIDIA GPU with CUDA support
- Compatible with RTX 20/30/40/50 series and datacenter GPUs
- Tested on RTX 2080, V100, A100, and RTX 5070 Ti

## Quick Start Training

### 1. Set up Environment
```bash
# Set up the environment
conda env create -f environment_modern.yml
conda activate rapppid_modern

# Navigate to training directory
cd rapppid/
```

### 2. Start Training
**Zero-configuration training (uses intelligent defaults):**
```bash
python train.py
```

**Custom parameters:**
```bash
python train.py --c_type 2 --batch_size 64 --num_epochs 50
```

**All available parameters:**
```bash
python train.py --help
```

### 3. What Happens Automatically
**Intelligent Path Detection:**
- ✅ **Data**: Auto-detects `../data/rapppid data/comparatives/string_c{1,2,3}/`
- ✅ **Model**: Auto-finds `smp250.model` and `smp250.vocab`  
- ✅ **Logs**: Auto-creates and uses `logs/` directory
- ✅ **Parameters**: Uses research paper defaults for all hyperparameters

### 4. Monitor Training
- Logs and checkpoints are saved to `logs/` directory
- View tensorboard: `tensorboard --logdir logs/tb_logs`
- Model metrics are in `logs/args/[model_name].json`

### 5. Available Datasets
- **String C1**: Dataset 1 (default) - use `--c_type 1`
- **String C2**: Dataset 2 - use `--c_type 2`  
- **String C3**: Dataset 3 - use `--c_type 3`

**No manual configuration needed** - everything is auto-detected from your current directory structure!


## License

RAPPPID

***R**egularised **A**utomative **P**rediction of **P**rotein-**P**rotein **I**nteractions using **D**eep Learning*

Copyright (C) 2021  Joseph Szymborski

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

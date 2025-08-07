# RAPPPID

***R**egularised **A**utomative **P**rediction of **P**rotein-**P**rotein **I**nteractions using **D**eep Learning*

---

RAPPPID is a deep learning model for predicting protein interactions. You can 
read more about it in the OUP Bionformatics [paper](https://doi.org/10.1093/bioinformatics/btac429).

## 모델 사용 방법 (Korean Instructions)

### 🚀 추론 실행하기 (사전 훈련된 모델 사용)

**1. 환경 설정**
```bash
conda env create -f environment_modern.yml
conda activate rapppid_modern
cd rapppid/
```

**2. 사전 훈련된 모델 위치**
- 모델 체크포인트: `../data/pretrained_weights/1690837077.519848_red-dreamy/1690837077.519848_red-dreamy.ckpt`
- SentencePiece 모델: `../data/pretrained_weights/1690837077.519848_red-dreamy/smp.model`

**3. 추론 코드 예시**
```python
from infer import *
import sentencepiece as sp

# 모델 로드
chkpt_path = '../data/pretrained_weights/1690837077.519848_red-dreamy/1690837077.519848_red-dreamy.ckpt'
model = load_chkpt(chkpt_path)

# SentencePiece 모델 로드
model_file = '../data/pretrained_weights/1690837077.519848_red-dreamy/smp.model'
spp = sp.SentencePieceProcessor(model_file=model_file)

# 테스트할 단백질 시퀀스 (아미노산 문자열)
seqs = [
    'LVYTDCTESGQNLCLCEGSNVCGQGNKCILGSDGEKNQCVTGEGTPKPQSHNDGDFEEIPEEYLQ',
    'QVQLKQSGPGLVQPSQSLSITCTVSGFSLTNYGVHWVRQSPGKGLEWLGVIWSGGNTDYNTPFTSRLSINKDNSKSQVFFKMNSLQSNDTAIYYCARALTYYDYEFAYWGQGTLVTVSAASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPKSPKSCDKTHTCPPCPAPELLGGP'
]

# 시퀀스 처리 및 예측
toks = process_seqs(spp, seqs, 1500)
embeddings = model(toks)
interaction_prob = predict(model, embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))

print(f"단백질 상호작용 확률: {interaction_prob.item():.4f}")
```

**4. 결과 해석**
- **출력값**: 0.0 ~ 1.0 사이의 확률값
- **0.5 이상**: 단백질 간 상호작용 가능성 높음
- **0.5 미만**: 단백질 간 상호작용 가능성 낮음

**5. 테스트 데이터 위치**
- 기존 테스트 데이터: `../data/rapppid/comparatives/string_c1/test_pairs.pkl.gz`
- 시퀀스 데이터: `../data/rapppid/comparatives/string_c1/seqs.pkl.gz`

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

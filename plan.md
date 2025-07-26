# RAPPPID Training Plan

## Overview

RAPPPID (Regularised Automative Prediction of Protein-Protein Interactions using Deep Learning) is a deep learning model that predicts protein-protein interactions using bidirectional LSTM networks with SentencePiece tokenization. This document provides a comprehensive step-by-step guide to train the model from scratch.

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with CUDA support (tested on RTX 2080, V100, A100)
- Minimum 8GB GPU memory (16GB+ recommended)
- Multi-core CPU (32 cores @ 2.2GHz used in original paper)

### Software Requirements
- Python 3.8+
- CUDA-compatible PyTorch
- Git (for cloning repository)

## Training Pipeline Overview

```
Data Preparation → Vocabulary Generation → Model Training → Evaluation
      ↓                    ↓                    ↓             ↓
  Format Data      SentencePiece Model    LSTM Training   Metrics & Charts
```

## Step-by-Step Training Guide

### 1. Environment Setup

Create and activate the conda environment:

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate rapppid
```

### 2. Data Preparation

Use the pre-existing datasets already downloaded in the `data/rappid data` folder. We'll use the `string_c1` dataset for training:

**Data Structure:**
```
data/rappid data/
├── comparatives/
│   ├── string_c1/          # We'll use this dataset
│   ├── string_c2/
│   └── string_c3/
└── repeatability/
    └── string_c1_seed-*/
```

The `string_c1` dataset contains:
- `seqs.pkl.gz` - Protein sequences
- `train_pairs.pkl.gz` - Training pairs
- `val_pairs.pkl.gz` - Validation pairs  
- `test_pairs.pkl.gz` - Test pairs

### 3. SentencePiece Vocabulary Generation

Generate tokenization vocabulary to prevent data leakage:

```bash
cd rapppid
python train_seg.py \
    --seed 8675309 \
    --vocab_size 250 \
    "data/rappid data/comparatives/string_c1/seqs.pkl.gz" \
    "data/rappid data/comparatives/string_c1/train_pairs.pkl.gz"
```

**Parameters:**
- `--seed`: Random seed for reproducibility (8675309, 5353456, 1234 used in paper)
- `--vocab_size`: Vocabulary size (250 recommended)

**Output Files:**
- `smp250.model` - SentencePiece model
- `smp250.vocab` - Vocabulary file

### 4. Logging Directory Setup

Create directory structure for training outputs:

```bash
mkdir -p logs/{args,chkpts,tb_logs,onnx,charts}
```

**Directory Purpose:**
- `args/` - Hyperparameters and metrics (JSON format)
- `chkpts/` - PyTorch Lightning model checkpoints
- `tb_logs/` - TensorBoard logs
- `onnx/` - ONNX model exports (optional)
- `charts/` - ROC and Precision-Recall curves

### 5. Model Training

Run the training command using the `string_c1` dataset:

```bash
python train.py \
    80 \                                    # batch_size
    1500 \                                  # trunc_len
    64 \                                    # embedding_size
    100 \                                   # num_epochs
    0.3 \                                   # lstm_dropout_rate
    0.2 \                                   # classhead_dropout_rate
    2 \                                     # rnn_num_layers
    2 \                                     # classhead_num_layers
    0.01 \                                  # lr
    0.0001 \                                # weight_decay
    "last" \                                # bi_reduce
    "mult" \                                # class_head_name
    False \                                 # variational_dropout
    False \                                 # lr_scaling
    ./smp250.model \                        # model_file
    --log_path ./logs \                     # log_path
    --vocab_size 250 \                      # vocab_size
    --embedding_droprate 0.3 \              # embedding_droprate
    --optimizer_type ranger21 \             # optimizer_type
    --swa True \                            # swa
    --seed 8675309 \                        # seed
    --train_path "data/rappid data/comparatives/string_c1/train_pairs.pkl.gz" \
    --val_path "data/rappid data/comparatives/string_c1/val_pairs.pkl.gz" \
    --test_path "data/rappid data/comparatives/string_c1/test_pairs.pkl.gz" \
    --seqs_path "data/rappid data/comparatives/string_c1/seqs.pkl.gz"
```

#### Parameter Explanations

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 80 | Number of protein pairs per training batch |
| `trunc_len` | 1500 | Maximum protein sequence length |
| `embedding_size` | 64 | Dimension of token embeddings |
| `num_epochs` | 100 | Maximum training epochs |
| `lstm_dropout_rate` | 0.3 | Dropout rate for LSTM layers |
| `classhead_dropout_rate` | 0.2 | Dropout rate for classifier |
| `rnn_num_layers` | 2 | Number of LSTM layers |
| `lr` | 0.01 | Learning rate |
| `class_head_name` | "mult" | Classifier type (mult/concat/mean/manhattan) |
| `optimizer_type` | "ranger21" | Optimizer (ranger21/adam) |
| `swa` | True | Stochastic Weight Averaging |

### 6. Training Monitoring

Monitor training progress using TensorBoard:

```bash
tensorboard --logdir logs/tb_logs
# Open http://localhost:6006 in browser
```

#### Key Metrics to Track
- **Training Loss**: Should decrease steadily
- **Validation Loss**: Should decrease without overfitting
- **AUROC**: Area Under ROC Curve (target: ~0.80)
- **AUPR**: Area Under Precision-Recall (target: ~0.80)
- **Accuracy**: Classification accuracy (target: ~0.71)

### 7. Training Results

Training will automatically:
- Save the best model based on validation loss
- Test the model on your test set
- Generate performance charts (ROC and precision-recall curves)
- Save all results in JSON format

#### Expected Performance Benchmarks
- **AUROC**: ~0.801
- **AUPR**: ~0.805  
- **Accuracy**: ~0.710

#### Result Files Location
- **Model**: `logs/chkpts/{timestamp}_{name}.ckpt`
- **Metrics**: `logs/args/{timestamp}_{name}.json`
- **Charts**: `logs/charts/{timestamp}_{name}_roc.pdf`

### 8. Troubleshooting

#### Common Issues and Solutions

**Out of Memory Error:**
- Reduce `batch_size` (try 40, 20)
- Reduce `trunc_len` (try 1000, 800)
- Use smaller `embedding_size` (try 32)

**Training Instability:**
- Lower learning rate (try 0.005, 0.001)
- Increase dropout rates
- Check for NaN values in loss

**Slow Training:**
- Verify CUDA is available: `torch.cuda.is_available()`
- Check GPU utilization: `nvidia-smi`
- Increase number of workers if CPU-bound

**Poor Performance:**
- Verify data quality and labels
- Check for class imbalance
- Try different `class_head_name` options
- Increase model capacity (more layers/embedding size)

### 9. Model Inference

After training, use the model for predictions:

```bash
python infer.py \
    --model_path logs/chkpts/your_model.ckpt \
    --input_pairs protein_pairs.pkl \
    --output_predictions predictions.json
```

### 10. Advanced Training Options

#### Multi-seed Training (Recommended)
Train multiple models with different seeds for ensemble:
```bash
for seed in 8675309 5353456 1234; do
    python train.py [parameters] --seed $seed
done
```

#### Custom Dataset Training
For new datasets, consider:
- Hyperparameter tuning specific to your data
- Cross-validation for robust evaluation
- Data augmentation techniques
- Domain-specific evaluation metrics

## Training Timeline

**Typical Training Duration:**
- Small dataset (100K pairs): 2-4 hours
- Medium dataset (1M pairs): 8-12 hours  
- Large dataset (10M+ pairs): 1-2 days

**Factors Affecting Speed:**
- Dataset size
- Sequence lengths
- Batch size
- GPU memory and compute capability
- Number of epochs until convergence

## Quality Assurance Checklist

Before considering training complete:

- [ ] Training converged (validation loss plateaued)
- [ ] No overfitting (validation loss not increasing)
- [ ] Test metrics meet expectations
- [ ] Model checkpoints saved successfully
- [ ] Training logs complete and readable
- [ ] Performance charts generated
- [ ] Results documented and reproducible

## Next Steps

After successful training:
1. Validate model on independent test sets
2. Compare performance with baseline methods
3. Analyze failure cases and model limitations
4. Deploy model for production inference
5. Share results and trained models with community

## References

- [RAPPPID Paper](https://doi.org/10.1093/bioinformatics/btac429)
- [Original Repository](https://github.com/jszym/rapppid)
- [Dataset Downloads](https://archive.org/details/rapppid_dataset)

## Support

For issues and questions:
- Check documentation in `docs/` directory
- Review GitHub issues
- Consult original paper for methodology details

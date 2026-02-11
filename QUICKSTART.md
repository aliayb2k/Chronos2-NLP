# Quick Start Guide: Adaptive Attention Extension

## 🚀 You're Ready to Train!

I've set up everything for you. Here's what's been created:

### New Files
- `prepare_ett_data.py` - Loads and prepares ETTm1 dataset
- `train_on_ett.py` - Trains adaptive attention on real data
- `evaluate_statistical.py` - Evaluates with statistical testing

### What You Have
- ✅ ETTm1 dataset ready (69,680 timesteps, 7 variables)
- ✅ Train/Val/Test splits (60/20/20)
- ✅ 868 training windows, 144 test samples
- ✅ All dependencies installed

---

## 📝 How to Run

### Option 1: Quick Test (5 minutes on CPU)
Just test that everything works:
```bash
# This will train for just a few steps to verify setup
python -c "from train_on_ett import train_on_ett; train_on_ett(max_steps=100)"
```

### Option 2: Full Training (CPU: ~24-48 hours, GPU: ~2-4 hours)
Full training with early stopping:
```bash
python train_on_ett.py
```

This will:
- Train for up to 10,000 steps
- Validate every 100 steps
- Auto-save best model to `./checkpoints/best_model.pt`
- Stop early if validation doesn't improve for 20 checks

### Option 3: Evaluate Only (10 minutes)
If you just want to see baseline vs untrained adaptive:
```bash
python evaluate_statistical.py
```

---

## 📊 What Happens During Training

```
1. Loads ETTm1 data
2. Prepares 868 training batches
3. Freezes Chronos-2 backbone (119M params)
4. Trains only adaptive bias (1.18M params)
5. Auto-saves best model based on validation loss
6. Stops early if no improvement
```

**Training output**:
- `./checkpoints/best_model.pt` - Best model
- `./results/training_log.json` - Loss curves

---

## 📈 How to Evaluate

After training completes:

```bash
python evaluate_statistical.py
```

This will:
1. Load baseline Chronos-2
2. Load your trained adaptive model
3. Evaluate both on 50 test samples
4. Perform paired t-test
5. Save results to `./results/evaluation_results.json`

**Output shows**:
- Baseline MAE ± std
- Adaptive MAE ± std  
- Statistical significance (p-value)
- Cohen's d (effect size)
- 95% confidence interval

---

## ⏱️ Time Estimates

| Task | CPU | GPU (T4) |
|------|-----|----------|
| Data loading | 1 min | 1 min |
| Training (10K steps) | 24-48 hrs | 2-4 hrs |
| Evaluation (50 samples) | 10 min | 2 min |

---

## 💡 Tips

**If training is too slow on CPU**:
1. Use Google Colab (free T4 GPU)
2. Or reduce steps: `train_on_ett(max_steps=1000)`

**To monitor progress**:
```bash
# Watch training log in real-time
tail -f results/training_log.json
```

**To resume from checkpoint**:
```python
# In Python:
from train_on_ett import train_on_ett
import torch

# Load existing checkpoint and continue
pipeline, log = train_on_ett(max_steps=15000)  # Will continue if checkpoint exists
```

---

## 🎯 Expected Results

Based on realistic expectations:

**Success Scenarios**:
1. **Best case**: Adaptive < Baseline with p < 0.05 ✨
2. **Good case**: Adaptive ≈ Baseline (within 5%)
3. **Learning case**: Higher MAE but mechanism works

**All are valid research outcomes!**

---

## 📁 Project Structure

```
DNLPproject/
├── prepare_ett_data.py      # Data loading
├── train_on_ett.py           # Training script
├── evaluate_statistical.py   # Evaluation
├── adaptive_attention.py     # Your extension
├── results/                  # Training logs
│   ├── training_log.json
│   └── evaluation_results.json
└── checkpoints/              # Model saves
    ├── best_model.pt
    └── final_model.pt
```

---

## 🐛 Troubleshooting

**Out of memory?**
- Reduce batch size: `train_on_ett(batch_size=16)`

**Training too slow?**
- Reduce steps: `train_on_ett(max_steps=1000)`
- Use Google Colab with GPU

**Want to test quickly?**
```python
# Mini training run (1-2 minutes)
from train_on_ett import train_on_ett
train_on_ett(max_steps=50, val_every=25)
```

---

## ✅ Next Steps

1. **Run quick test** to verify setup works
2. **Start full training** (can run overnight)
3. **Evaluate results** with statistical testing
4. **Document findings** in walkthrough

You're all set! 🚀

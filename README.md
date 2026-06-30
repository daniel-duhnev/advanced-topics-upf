# Skin Lesion Classification with Transfer Learning

Technical work for the Advanced Topics on Intelligent Systems module (UPF, 2025-26).

Reproduces deep learning skin lesion classification on the HAM10000 dataset. Compares two transfer learning architectures (ResNet-50 and EfficientNet-B0) and uses Grad-CAM for model explainability.

**Author:** Daniel Duhnev
**Report:** [docs/report.pdf](docs/report.pdf)

---

## Dataset

**HAM10000** - 10,015 dermoscopic images across 7 diagnostic categories.

Download from Kaggle: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

You need a Kaggle account and API key to download programmatically. See setup instructions below.

| Class | Abbreviation | Count | % |
|-------|-------------|-------|---|
| Melanocytic Nevi | nv | 6,705 | 67.0% |
| Melanoma | mel | 1,113 | 11.1% |
| Benign Keratosis | bkl | 1,099 | 11.0% |
| Basal Cell Carcinoma | bcc | 514 | 5.1% |
| Actinic Keratoses | akiec | 327 | 3.3% |
| Vascular Lesions | vasc | 142 | 1.4% |
| Dermatofibroma | df | 115 | 1.1% |

---

## Results

| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| ResNet-50 | 79.7% | 0.67 | 0.80 |
| EfficientNet-B0 | 71.2% | 0.54 | 0.73 |

---

## Project Structure

```
src/                Python modules (config, dataset, models, training, evaluation, grad-cam)
scripts/            Standalone scripts for local training and grad-cam generation
notebooks/          Jupyter notebooks (Colab-ready, run in order: 01 -> 02 -> 03)
docs/               Report (markdown + PDF) and presentation transcript
results/            Generated figures (gitignored - produced during training)
models/             Saved model checkpoints (gitignored)
data/               HAM10000 dataset (gitignored - download separately)
```

---

## How to Run

### Option A: Local (Mac or Linux with GPU)

**Requirements:** Conda (Miniconda or Anaconda), a Kaggle API key.

**1. Set up the environment:**

```bash
conda env create -f environment.yml
conda activate skin-lesion
```

**2. Get a Kaggle API key:**

Go to https://www.kaggle.com/settings and click "Create New Token". This downloads a `kaggle.json` file. Place it at `~/.kaggle/kaggle.json`:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

**3. Download the dataset:**

```bash
pip install kaggle
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/
unzip data/skin-cancer-mnist-ham10000.zip -d data/HAM10000/
```

After unzipping, make sure images from both `HAM10000_images_part_1/` and `HAM10000_images_part_2/` are accessible. The code handles both folder structures automatically.

**4. Run training:**

```bash
python scripts/run_training.py
```

This trains both models sequentially, saves checkpoints to `models/`, and saves all figures and metrics to `results/`. Takes about 1-2 hours on an Apple Silicon Mac (MPS) or NVIDIA GPU.

**5. Generate Grad-CAM visualisations:**

```bash
python scripts/run_gradcam.py
```

### Option B: Google Colab

1. Open `notebooks/02_train_and_evaluate.ipynb` in Google Colab
2. Change the runtime to T4 GPU (Runtime - Change runtime type - T4)
3. Run the first cell to install dependencies and download the dataset (you will need to upload your `kaggle.json` when prompted)
4. Run all cells - training takes about 1-2 hours

The notebooks are designed to work on Colab out of the box. They detect the environment (Colab vs local) and set paths accordingly.

---

## Key Design Decisions

- **Split by lesion_id, not image_id.** Some lesions have multiple images in the dataset. Splitting by image would leak the same lesion into train and test sets, inflating accuracy.
- **Weighted cross-entropy loss.** The nv class is 67% of the data. Without weighting, the model just predicts nv for everything and gets 67% accuracy for free.
- **Freeze early layers.** We only fine-tune the last convolutional block and the classifier head. The earlier layers already know how to extract general image features from ImageNet pretraining.
- **ImageNet normalisation.** Both models expect inputs normalised with ImageNet mean and std (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

---

## Dependencies

See `requirements.txt` (pip) or `environment.yml` (conda). Main packages:

- PyTorch 2.0+
- torchvision
- scikit-learn
- matplotlib, seaborn
- pandas, numpy
- Pillow, tqdm

---

## References

1. Tschandl, P., Rosendahl, C. & Kittler, H. (2018). The HAM10000 dataset. Scientific Data, 5, 180161.
2. Tschandl, P. et al. (2019). Comparison of the accuracy of human readers versus machine-learning algorithms for pigmented skin lesion classification. The Lancet Oncology, 20(7), 938-947.
3. He, K. et al. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.
4. Tan, M. & Le, Q.V. (2019). EfficientNet: Rethinking Model Scaling for CNNs. ICML 2019.
5. Selvaraju, R.R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. ICCV 2017.

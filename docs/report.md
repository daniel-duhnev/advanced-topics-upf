# Reproducing Deep Learning Skin Lesion Classification on HAM10000: Comparing Transfer Learning Architectures

**Author:** Daniel Duhnev
**Course:** Advanced Topics on Intelligent Systems, UPF 2025-26
**Repository:** https://github.com/daniel-duhnev/advanced-topics-upf

---

## 1. Introduction

Skin cancer is the most common type of cancer worldwide. When detected early, the five-year survival rate for melanoma exceeds 95%, but this drops below 30% for late-stage diagnoses. Dermoscopy - the examination of skin lesions with a magnifying device - is the standard screening method, but its accuracy depends heavily on the clinician's experience and training.

In recent years, deep learning models trained on dermoscopic images have shown accuracy comparable to board-certified dermatologists [1]. This suggests that automated systems could assist clinicians in triaging suspicious lesions for biopsy, particularly in settings with limited access to specialist dermatologists.

In this work, I reproduce the core transfer learning approach from the literature: training pretrained convolutional neural networks on the HAM10000 dermoscopic image dataset to classify skin lesions into seven diagnostic categories. Two architectures are compared - ResNet-50 [3] and EfficientNet-B0 [4] - and Grad-CAM [5] is used to visualise which image regions drive the model's predictions. The goal is to understand how well standard transfer learning performs on a challenging medical imaging task with severe class imbalance.

## 2. Dataset

The HAM10000 dataset [2] is used, containing 10,015 dermoscopic images of pigmented skin lesions collected from the Medical University of Vienna and the Skin Cancer Practice of Cliff Rosendahl. Each image is labelled with one of seven diagnoses:

- Melanocytic nevi (nv) - 6,705 images (66.9%)
- Melanoma (mel) - 1,113 images (11.1%)
- Benign keratosis (bkl) - 1,099 images (11.0%)
- Basal cell carcinoma (bcc) - 514 images (5.1%)
- Actinic keratoses (akiec) - 327 images (3.3%)
- Vascular lesions (vasc) - 142 images (1.4%)
- Dermatofibroma (df) - 115 images (1.1%)

The class distribution is severely imbalanced (see Figure 1 in Appendix). Melanocytic nevi alone account for two thirds of all samples, while dermatofibroma and vascular lesions together make up less than 3%. This imbalance is a central challenge - a naive model can reach 67% accuracy simply by predicting nv for every input.

The data was split into 70% training, 15% validation, and 15% test sets. Crucially, the split was performed by lesion_id rather than image_id, because some lesions have multiple images taken at different angles. Splitting by image_id would place different views of the same lesion into both training and test sets, causing data leakage and inflated performance estimates.

## 3. Methods

**Transfer learning.** Both models are pretrained on ImageNet (1.2 million natural images, 1000 classes). The early convolutional layers - which learn general features like edges and textures - are frozen, and only the later layers plus a new classifier head are fine-tuned for this 7-class problem. This is standard practice when training data is limited, as it prevents overfitting while still allowing the model to adapt to the target domain.

**Architectures.** ResNet-50 uses residual (skip) connections to train very deep networks without degradation. Layers 1-3 are frozen and layer 4 plus the classifier is trained, giving approximately 15 million trainable parameters out of 25 million total. EfficientNet-B0 uses compound scaling to balance network depth, width, and resolution. All feature blocks except the last two are frozen and a new classifier is trained, giving approximately 1.1 million trainable parameters out of 5.3 million total.

**Handling class imbalance.** Weighted cross-entropy loss is used with inverse-frequency weights. Classes with fewer samples receive higher loss weights, forcing the model to pay attention to rare classes rather than defaulting to predicting nv. For example, dermatofibroma (115 samples) receives a weight roughly 58 times higher than nv (6,705 samples).

**Data augmentation.** During training, random horizontal and vertical flips, rotation up to 20 degrees, and slight colour jitter (brightness, contrast, saturation, hue) are applied. These augmentations simulate natural variation in how lesions might be photographed and help the model generalise rather than memorise specific images. Augmentation is only applied to the training set.

**Training details.** The Adam optimiser is used with a learning rate of 1e-4 and weight decay of 1e-4. Images are resized to 224x224 pixels and normalised with ImageNet statistics. Batch size is 32. Early stopping is applied with patience of 5 epochs - if validation loss does not improve for 5 consecutive epochs, training stops to prevent overfitting.

**Explainability.** Grad-CAM (Gradient-weighted Class Activation Mapping) [5] is applied to the final convolutional layer of each model. This produces a heatmap showing which spatial regions of the input image contributed most to the model's prediction. This helps verify whether the model learns clinically meaningful features or relies on spurious correlations like pen markings or image borders.

## 4. Results

Table 1 summarises the test set performance of both models.

| Model | Accuracy | F1 (macro) | F1 (weighted) |
|-------|----------|------------|---------------|
| ResNet-50 | 79.7% | 0.674 | 0.799 |
| EfficientNet-B0 | 71.2% | 0.544 | 0.734 |

**Table 1.** Test set metrics. Macro F1 treats all classes equally; weighted F1 accounts for class support.

ResNet-50 outperforms EfficientNet-B0 across all metrics. The gap is most visible in macro F1 (0.674 vs 0.544), which is sensitive to performance on minority classes. This suggests that EfficientNet-B0's smaller capacity (1.1M trainable parameters vs 15M) limits its ability to adapt from natural images to the medical domain with the freezing strategy applied.

Looking at per-class performance for ResNet-50 (see confusion matrix in Figure 2): melanocytic nevi achieves 88% recall, benign keratosis 71%, and basal cell carcinoma 63%. The hardest class is melanoma at 52% recall - most misclassifications go to nv, which makes clinical sense since both are melanocytic lesions and can look visually similar. Vascular lesions and dermatofibroma achieve reasonable recall (81% and 100% respectively) but have very few test samples (21 and 7), so these numbers should be interpreted cautiously.

The training curves (Figure 3) show a clear overfitting pattern for both models: training loss continues to decrease while validation loss plateaus after epoch 6-8. Early stopping triggers at epoch 16 for ResNet-50 and epoch 21 for EfficientNet-B0, correctly preventing further overfitting.

## 5. Grad-CAM Analysis

Figure 4 shows Grad-CAM attention maps for both models on one representative image per class. In most cases, both models focus their attention on the lesion area itself rather than on surrounding skin or image artifacts. This is a positive sign - it indicates the models have learned to identify lesion-specific visual features rather than exploiting dataset biases.

ResNet-50 produces slightly more focused and concentrated heatmaps compared to EfficientNet-B0, whose attention tends to be more diffuse. This correlates with its higher classification accuracy - a model that looks at more specific lesion features is likely to make better distinctions between similar classes.

For melanoma specifically, the model attends to areas with colour variation and irregular borders, which are known dermoscopic indicators of malignancy. For vascular lesions, attention concentrates on the characteristic reddish-purple areas. These patterns suggest the model has picked up on clinically relevant features.

## 6. Discussion and Limitations

Results are consistent with published baselines on HAM10000. Without ensembling or complex preprocessing, single models typically achieve 75-87% accuracy on this dataset [2]. The ResNet-50 result at 79.7% falls within this expected range.

The performance gap between ResNet-50 and EfficientNet-B0 is likely due to the freezing strategy. EfficientNet-B0 is designed to be parameter-efficient, but with only 1.1M trainable parameters, it may lack the capacity to fully adapt to dermoscopic images which differ significantly from ImageNet's natural images. Unfreezing more layers or using a higher learning rate could improve its performance.

Several limitations apply to this work. First, classification is performed on single images without patient metadata (age, sex, lesion localisation) which could improve accuracy. Second, there is no temporal information - in practice, monitoring change over time is a key diagnostic tool. Third, HAM10000 is relatively small by deep learning standards and originates from only two institutions, which limits generalisability. Finally, there is no external validation set from a different population or imaging setup.

## 7. Conclusion

Transfer learning-based skin lesion classification was reproduced on the HAM10000 dataset, comparing ResNet-50 and EfficientNet-B0. ResNet-50 achieved 79.7% accuracy and 0.674 macro F1, outperforming EfficientNet-B0 (71.2% accuracy, 0.544 macro F1) with the same fine-tuning configuration. Grad-CAM visualisations confirm that both models attend to clinically relevant lesion features in most cases. This work demonstrates that pretrained CNNs can achieve reasonable diagnostic accuracy on dermoscopic images with modest computational resources, supporting the feasibility of AI-assisted skin cancer screening tools.

## References

[1] Esteva, A. et al. (2017). Dermatologist-level classification of skin cancer with deep neural networks. *Nature*, 542, 115-118.

[2] Tschandl, P., Rosendahl, C. & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*, 5, 180161.

[3] He, K. et al. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.

[4] Tan, M. & Le, Q.V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML 2019*.

[5] Selvaraju, R.R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV 2017*.

---

## Appendix: Figures

**Figure 1.** HAM10000 class distribution. Melanocytic nevi (nv) accounts for 67% of all images.

![Class Distribution](../results/class_distribution.png){ width=85% }

**Figure 2.** Normalised confusion matrices on the test set. ResNet-50 (left), EfficientNet-B0 (right).

![ResNet-50](../results/confusion_matrix_resnet50.png){ width=48% } ![EfficientNet-B0](../results/confusion_matrix_efficientnet.png){ width=48% }

**Figure 3.** Training curves. ResNet-50 (left), EfficientNet-B0 (right). Early stopping prevents overfitting.

![ResNet-50](../results/training_curves_resnet50.png){ width=48% } ![EfficientNet-B0](../results/training_curves_efficientnet.png){ width=48% }

**Figure 4.** Grad-CAM attention maps (ResNet-50). Each row: original, heatmap, overlay. The model focuses on the lesion in most cases.

![Grad-CAM](../results/gradcam_resnet50.png){ width=50% }

---
title: "Reproducing Deep Learning Skin Lesion Classification on HAM10000"
subtitle: "Comparing Transfer Learning Architectures"
author: "Daniel Duhnev"
institute: "Advanced Topics on Intelligent Systems - UPF 2025-26"
date: ""
theme: "Madrid"
colortheme: "default"
fontsize: 11pt
aspectratio: 169
header-includes:
  - \usepackage{graphicx}
  - \usepackage{booktabs}
  - \setbeamertemplate{navigation symbols}{}
  - \input{slides_header.tex}
---

# Motivation

- Skin cancer is the most common cancer type globally
- Early detection: 5-year melanoma survival >95% (early) vs <30% (late)
- Dermoscopy accuracy depends on clinician experience
- Deep learning has matched dermatologist-level accuracy (Esteva et al., 2017)

\vspace{0.5cm}

**This project:** reproduce the transfer learning approach on HAM10000

- Compare two architectures: ResNet-50 vs EfficientNet-B0
- Use Grad-CAM to visualise what the models focus on

---

# Dataset: HAM10000

\begin{columns}
\begin{column}{0.45\textwidth}
\begin{itemize}
\item 10,015 dermoscopic images
\item 7 diagnostic categories
\item Severe class imbalance (67\% nv)
\item Split by lesion\_id (not image\_id) to prevent data leakage
\end{itemize}
\vspace{0.3cm}
\footnotesize{Tschandl et al. (2018), Scientific Data}
\end{column}
\begin{column}{0.55\textwidth}
\includegraphics[width=\textwidth]{../results/class_distribution.png}
\end{column}
\end{columns}

---

# Methods

**Transfer learning:**

- Both models pretrained on ImageNet (1.2M natural images)
- Freeze early layers, fine-tune last block + classifier head
- ResNet-50: ~15M trainable params / EfficientNet-B0: ~1.1M trainable params

\vspace{0.3cm}

**Handling class imbalance:**

- Weighted cross-entropy loss (inverse frequency weighting)
- Data augmentation: random flips, rotation, colour jitter

\vspace{0.3cm}

**Training:**

- Adam optimiser, LR = 1e-4, batch size 32, images resized to 224x224
- Early stopping (patience = 5 epochs on validation loss)

---

# Results

\begin{table}
\centering
\begin{tabular}{lccc}
\toprule
Model & Accuracy & Macro F1 & Weighted F1 \\
\midrule
ResNet-50 & 79.7\% & 0.674 & 0.799 \\
EfficientNet-B0 & 71.2\% & 0.544 & 0.734 \\
\bottomrule
\end{tabular}
\end{table}

\vspace{0.3cm}

- ResNet-50 clearly outperforms EfficientNet-B0 in this setup
- Hardest class: melanoma (52% recall) - confused with melanocytic nevi
- Easiest class: melanocytic nevi (88% recall) - most training data
- EfficientNet gap likely due to too few trainable parameters

---

# Results - Confusion Matrix (ResNet-50)

\begin{center}
\includegraphics[height=0.85\textheight]{../results/confusion_matrix_resnet50.png}
\end{center}

---

# Grad-CAM Analysis

\begin{columns}
\begin{column}{0.5\textwidth}
\includegraphics[width=\textwidth]{../results/gradcam_resnet50_slide.png}
\end{column}
\begin{column}{0.5\textwidth}
\begin{itemize}
\item Models focus on the lesion itself (not artifacts or borders)
\item ResNet-50 produces more focused heatmaps than EfficientNet-B0
\item Melanoma: attention on colour variation and irregular borders
\item Suggests clinically relevant features learned
\end{itemize}
\vspace{0.3cm}
\footnotesize{Columns: original image, heatmap, overlay. Grad-CAM run on both models (ResNet-50 shown).}
\end{column}
\end{columns}

---

# Conclusion

**Summary:**

- Reproduced transfer learning classification on HAM10000
- ResNet-50 (79.7\%) outperformed EfficientNet-B0 (71.2\%)
- Grad-CAM confirms models attend to clinically relevant features

\vspace{0.3cm}

**Limitations:**

- Single-image classification (no temporal info, no patient metadata)
- Small dataset from only two institutions
- EfficientNet may improve with more unfrozen layers

\vspace{0.5cm}

\begin{center}
\small Code: \texttt{github.com/daniel-duhnev/advanced-topics-upf}
\end{center}

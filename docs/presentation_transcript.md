# Presentation Transcript (~6 minutes)

Read this over the slides. Each section matches a slide. Speak at a natural pace.

---

## Slide 1: Title (20 seconds)

Hi, my name is Daniel Duhnev. This presentation is for the Advanced Topics on Intelligent Systems module. My technical work reproduces deep learning skin lesion classification on the HAM10000 dataset, comparing two transfer learning architectures.

---

## Slide 2: Motivation (1 minute)

Skin cancer is the most common type of cancer globally. The key thing is that early detection makes a huge difference - the five-year survival rate for melanoma is over 95% if caught early, but drops below 30% for late-stage cases.

The standard screening method is dermoscopy, where a clinician examines skin lesions with a magnifying device. The problem is that accuracy depends heavily on the clinician's experience.

In 2017, Esteva and colleagues showed that a deep learning model could match dermatologist-level accuracy on this task. I found this interesting, so for my project I reproduce the core approach: train two pretrained CNNs on dermoscopic images and compare them. I also use Grad-CAM to visualise what the models focus on.

---

## Slide 3: Dataset (1 minute)

I used the HAM10000 dataset, published by Tschandl and colleagues in 2018. It contains 10,015 dermoscopic images across seven diagnostic categories.

The most important thing here is the class imbalance. Melanocytic nevi - common moles - make up 67% of the dataset. Meanwhile some classes like dermatofibroma have fewer than 150 samples. A model that just predicts "mole" for everything would get 67% accuracy without learning anything. So we need to handle this explicitly.

I split the data by lesion ID, not image ID, because some lesions have multiple photos. If I split by image, the same lesion could appear in both training and test sets, which would be data leakage.

---

## Slide 4: Methods (1 minute)

Both models start from ImageNet pretrained weights - they already know general visual features from 1.2 million natural images. I freeze the early layers and fine-tune the last block plus a new classifier head.

ResNet-50 has about 15 million trainable parameters. EfficientNet-B0 is much smaller with 1.1 million trainable parameters.

To handle class imbalance I use weighted cross-entropy loss - rare classes get higher weights. I also apply data augmentation: random flips, rotations, and colour jitter.

Training uses Adam with learning rate 1e-4, batch size 32, images resized to 224 by 224. Early stopping with patience of 5 prevents overfitting.

---

## Slide 5: Results (1 minute)

On the test set, ResNet-50 achieved 79.7% accuracy with a macro F1 of 0.674. EfficientNet-B0 got 71.2% accuracy with macro F1 of 0.544. So ResNet-50 clearly outperforms in this setup.

The hardest class is melanoma at 52% recall - most misclassifications go to melanocytic nevi, which makes sense because both are melanocytic lesions and look similar. The easiest class is nevi at 88% recall since it has the most training data.

The performance gap is likely because EfficientNet has too few trainable parameters to adapt well from natural images to dermoscopic images.

---

## Slide 6: Confusion Matrix (30 seconds)

Here is the confusion matrix for ResNet-50. You can see the diagonal is strong for most classes. The main off-diagonal pattern is melanoma being confused with nevi - the 0.33 in that cell. Dermatofibroma achieves perfect recall in the test set but this is likely due to the very small sample size rather than the model being exceptionally good at that class.

---

## Slide 7: Grad-CAM (45 seconds)

To check what the models actually look at, I used Grad-CAM - which produces heatmaps showing which image regions drove the prediction.

The results are encouraging. In most cases the model focuses on the lesion itself rather than on artifacts or image borders. For melanoma, attention is on areas with colour variation and irregular borders, which are known clinical indicators of malignancy. This suggests the model has learned clinically relevant features rather than just memorising shortcuts.

---

## Slide 8: Conclusion (30 seconds)

To summarise: I reproduced transfer learning classification on HAM10000 and compared two architectures. ResNet-50 at 79.7% outperformed EfficientNet-B0 at 71.2%. Grad-CAM confirms the models attend to clinically relevant features.

Main limitations: single-image classification without patient history, small dataset from two institutions, and EfficientNet could likely improve with more unfrozen layers.

The code is available on GitHub at the link shown. Thank you.

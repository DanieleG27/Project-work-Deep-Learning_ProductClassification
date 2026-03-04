# Grocery Product Classification: Custom ResVGG vs. ResNet-18

This project addresses the challenge of **fine-grained image classification** in a real-world retail environment using the **Grocery Store Dataset**. The goal is to benchmark a custom-designed CNN architecture against a fine-tuned state-of-the-art model.

##  Project Overview
The system was developed in two main phases:
1.  **Task 1 - Custom ResVGG:** Designing and training an original 5.1M parameter CNN from scratch without pre-trained weights.
2.  **Task 2 - Transfer Learning:** Fine-tuning a pre-trained **ResNet-18** (ImageNet-1K) to maximize accuracy across the 81 product classes.

##  Custom Architecture (ResVGG)
For Task 1, I developed a hybrid architecture combining the strengths of VGG and ResNet:
* **VGG-style Initial Blocks:** 3x3 convolutions and Max Pooling for low-level spatial feature extraction.
* **Residual Learning Integration:** Shortcut connections (Skip Connections) in deeper blocks (256 and 512 filters) to mitigate the **vanishing gradient** problem.
* **Regularization Strategy:** Systematic use of **Batch Normalization**, **L2 Weight Regularization**, and **Global Average Pooling (GAP)** to keep the parameter count manageable and prevent overfitting.



##  Data-Centric Innovations
The model's success relied heavily on advanced data management strategies:
* **Iconic Image Injection:** I augmented the training set with "Iconic" images (clean reference views) using a **20x oversampling** weight. This established a strong visual baseline before the model attempted to generalize to noisy "Natural" images.
* **High-Performance tf.data Pipeline:** Implementation of **Caching** and **Prefetching** using `tf.data.AUTOTUNE` to ensure the GPU remains fully saturated during training, eliminating I/O bottlenecks.
* **Label Smoothing:** Applied to prevent the model from becoming overconfident, significantly improving generalization on visually similar categories (e.g., different apple varieties).

##  Results & Performance Evolution
The project followed an evolutionary roadmap monitored through **Random Search** for hyperparameter optimization.

| Version | Architecture | Key Characteristics | Val Accuracy |
| :--- | :--- | :--- | :--- |
| **V0** | Baseline | 2 Simple Conv Layers | <25% |
| **V2** | VGG-like | Multiple Convs + L2 | ~50% |
| **V4** | **ResVGG** | Residual Blocks + Iconic Weighting | **>60%** |
| **Final** | **ResNet-18** | Fine-tuning (Transfer Learning) | **~88%** |



##  Tech Stack
* **Framework:** TensorFlow / Keras
* **Language:** Python
* **Libraries:** NumPy, Pandas, Matplotlib, OpenCV
* **Key Techniques:** Transfer Learning, Data Augmentation, Residual Learning, Hyperparameter Tuning.

##  Conclusion
This project demonstrates that while **Transfer Learning** provides superior performance with less computational effort, a well-designed **custom CNN** utilizing residual blocks and an intelligent data strategy can achieve high robustness even on complex, unbalanced datasets with 81 classes.

---
*Developed as part of the Product Classification project for the course Computer Vision and Deep Learning - University of Bologna.*

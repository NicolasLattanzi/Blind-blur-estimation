# Blind blur estimation for image restoration

Computer vision 2024/25 project. Sapienza University of Rome.  

Abstract: Image restoration is a fundamental task in computer vision, with image blur being one of the
most common types of degradation. Blind blur estimation, which aims to identify the type and parameters
of a blur without access to the original image or the blur kernel, is particularly challenging due to the
diversity and ambiguity of real-world degradations. This project focuses on the problem of blind image
blur estimation by designing a two-stage learning-based framework. In the first stage, a neural model
classifies the type of blur affecting an image region. In the second stage, a regression module estimates the
corresponding blur parameters to support further restoration. The study also explores the effectiveness of
different architectures for feature extraction, comparing traditional deep neural networks with more recent
lightweight transformer-based encoders.

Dataset: BSDS500, DIV2K  

The goal of this project is to develop and compare two models for blind image blur estimation. The
first replicates the original approach based on a deep neural network (DNN) followed by a general regression
neural network (GRNN) for parameter prediction. The second replaces the DNN with a lightweight transformer
encoder (e.g., MobileViT, TinyViT) trained on Fourier-domain representations of degraded image patches. The
task includes training, evaluation, and analysis of classification accuracy, regression precision, and computational
performance.

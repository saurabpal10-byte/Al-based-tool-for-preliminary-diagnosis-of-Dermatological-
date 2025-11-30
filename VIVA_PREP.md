# Viva Preparation Guide: DermAI

This document covers the key concepts, technical details, and potential questions you might face during your viva for the DermAI project.

## 1. Project Overview
*   **Project Name:** DermAI (Dermatology AI)
*   **Objective:** To develop a web-based application that uses Deep Learning to detect and classify various skin diseases from images.
*   **Problem Solved:** Accessibility to early diagnosis of skin conditions, reducing the workload on dermatologists, and providing instant preliminary analysis.

## 2. Technology Stack
*   **Frontend (User Interface):**
    *   **HTML5:** Structure of the pages.
    *   **CSS3:** Styling (Flexbox, Grid, Variables, Animations).
    *   **JavaScript (Vanilla):** Logic for image upload, API calls (Fetch), and DOM manipulation.
*   **Backend (Server):**
    *   **Python:** Core programming language.
    *   **Flask:** A lightweight web framework to handle API requests and serve pages.
    *   **SQLite:** A lightweight, file-based relational database for storing user credentials.
*   **Machine Learning / AI:**
    *   **PyTorch:** The deep learning framework used to build and run the model.
    *   **Torchvision:** Used for image transformations and loading pre-trained models.
    *   **Pillow (PIL):** Python Imaging Library for image processing.

## 3. The Core Algorithm: ResNet (Residual Networks)
*   **What model did you use?** We used **ResNet-18** (or ResNet-34/50 depending on your specific training).
*   **Why ResNet?**
    *   Standard CNNs (Convolutional Neural Networks) get harder to train as they get deeper due to the **Vanishing Gradient Problem**.
    *   ResNet introduces **Skip Connections** (or Residual Blocks) that allow gradients to flow through the network more easily.
    *   This allows us to train deeper networks that can learn more complex features without performance degradation.
*   **Transfer Learning:**
    *   We likely used a model **pre-trained on ImageNet** (a massive dataset of everyday objects).
    *   We then **fine-tuned** it on our specific Dermatology dataset. This saves training time and improves accuracy.

## 4. Key Features Explained

### A. Grad-CAM (Gradient-weighted Class Activation Mapping)
*   **What is it?** It's an "Explainable AI" (XAI) technique.
*   **How it works:** It looks at the gradients of the target class flowing into the final convolutional layer to produce a coarse localization map.
*   **Purpose:** It generates a **Heatmap** that highlights the important regions in the image used for predicting the concept. It tells us *where* the model is looking (e.g., focusing on the lesion and not the background skin).

### B. Accuracy Score
*   **What is it?** It represents the **Confidence** (probability) of the model's prediction.
*   **Calculation:** The raw output of the model (logits) is passed through a **Softmax** function, which converts them into probabilities summing to 1 (or 100%).

### C. Authentication
*   **How is it secure?** Passwords are **hashed** using `werkzeug.security.generate_password_hash` before being stored in the SQLite database. We never store plain-text passwords.
*   **Session Management:** Flask's `session` object is used to keep users logged in securely.

## 5. Common Viva Questions

### Q1: How does your model process an image?
**Answer:**
1.  **Preprocessing:** The image is resized to 224x224 pixels and converted to a Tensor.
2.  **Forward Pass:** The Tensor is passed through the ResNet model layers (Convolutions -> ReLU -> Pooling).
3.  **Feature Extraction:** The model extracts low-level features (edges, colors) and high-level features (textures, shapes).
4.  **Classification:** The final Fully Connected (FC) layer maps these features to the 9 disease classes.
5.  **Output:** Softmax is applied to get the probability for each class.

### Q2: What is the difference between validation and testing?
**Answer:**
*   **Training Set:** Used to teach the model.
*   **Validation Set:** Used to tune hyperparameters (like learning rate) during training.
*   **Test Set:** Used to evaluate the final performance on unseen data.

### Q3: Why did you choose Flask over Django?
**Answer:** Flask is a **micro-framework**. It is lightweight, flexible, and easy to set up for simple API-based applications like this. Django is "batteries-included" and would be overkill for this specific scope.

### Q4: What are the limitations of your project?
**Answer:**
*   **Data Bias:** The model might perform poorly on skin tones that were underrepresented in the training data.
*   **Not a Doctor:** It provides a probability, not a definitive medical diagnosis.
*   **Lighting Conditions:** Poor lighting or blurry images can significantly affect accuracy.

### Q5: How would you improve this in the future?
**Answer:**
*   **Data Augmentation:** Use more techniques to increase dataset variety.
*   **Mobile App:** Build a native mobile app for easier access.
*   **Real-time Inference:** Optimize the model to run directly in the browser (using ONNX.js or TensorFlow.js) for privacy and speed.

## 6. Code Structure
*   `site_server.py`: The main entry point. Handles the web server, database connections, and model inference.
*   `web/`: Contains all frontend assets.
    *   `index.html`: Main dashboard.
    *   `style.css`: All styling.
    *   `app.js`: Connects the frontend to the backend API.
*   `users.db`: The SQLite database file.

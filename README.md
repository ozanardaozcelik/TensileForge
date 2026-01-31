# ⚒️ TensileForge: AI-Driven Metallurgy Optimizer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python)
![Framework](https://img.shields.io/badge/Backend-Flask-lightgrey?style=flat&logo=flask)
![ML](https://img.shields.io/badge/ML-XGBoost-green?style=flat)
![DL](https://img.shields.io/badge/DL-PyTorch-red?style=flat&logo=pytorch)
![Domain](https://img.shields.io/badge/Domain-Material%20Science-orange?style=flat)
![Status](https://img.shields.io/badge/Status-Experimental-yellow?style=flat)

**TensileForge** is an advanced machine learning platform designed to bridge the gap between traditional **metallurgy** and **data science**. 

The primary goal of this project is to predict the **Tensile Strength (MPa)** of steel and steel alloys based on critical processing parameters such as temperature, chemical composition, and mechanical properties. By leveraging AI, engineers can optimize manufacturing conditions without relying solely on costly and destructive physical testing.

---

## 📖 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution Overview](#-solution-overview)
- [System Architecture](#-system-architecture)
  - [The XGBoost Engine](#1-the-xgboost-engine)
  - [The Neural Network (Experimental)](#2-the-neural-network-experimental)
- [Web Interface & UX](#-web-interface--ux)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Future Roadmap](#-future-roadmap)

---

## 🚩 Problem Statement

In material science, determining the optimal tensile strength of an alloy typically requires:
1.  Manufacturing a sample.
2.  Conducting destructive tensile testing using universal testing machines.
3.  Analyzing the fracture point.

This process is **time-consuming**, **expensive**, and results in **material waste**. Engineers need a way to simulate and predict these properties instantly based on input parameters before the physical manufacturing process begins.

---

## 💡 Solution Overview

**TensileForge** provides a web-based simulation environment where users can input alloy parameters and receive instant strength predictions. 

The system compares two distinct AI approaches:
1.  **Gradient Boosting (XGBoost):** Optimized for tabular data, providing interpretability and high accuracy.
2.  **Deep Learning (PyTorch):** A neural network approach designed to capture complex, non-linear relationships in the data.

---

## 🏗️ System Architecture

The application is built on a robust pipeline that handles data preprocessing, scaling, and inference.

### 1. The XGBoost Engine
* **Status:** ✅ Stable & Production Ready
* **Implementation:** Uses the `XGBRegressor` with optimized hyperparameters.
* **Why XGBoost?** It excels at structured/tabular datasets common in engineering logs. It handles feature interactions effectively and is robust against outliers.
* **Data Handling:** Utilizes `DMatrix` for efficient memory usage during inference.

### 2. The Neural Network (Experimental)
* **Status:** 🚧 Under Development / Calibration
* **Framework:** PyTorch (`torch.nn`)
* **Architecture:** * **Input Layer:** Matches the dimensionality of the dataset features.
    * **Hidden Layers:** Fully connected layers with ReLU activation functions to introduce non-linearity.
    * **Output Layer:** Single neuron regression output (Tensile Strength).
* **Current Challenge:** The NN model is currently being fine-tuned to match the accuracy of the XGBoost model. It requires further normalization of inputs and hyperparameter optimization (Learning Rate, Batch Size).

### 3. Data Preprocessing
Before feeding data into the models, the system uses:
* **Label Encoders:** To convert categorical alloy types into numerical format.
* **Standard Scalers:** To normalize numerical inputs (e.g., Temperature, Pressure) ensuring that high-magnitude features don't dominate the model.

---

## 💻 Web Interface & UX

The frontend is designed with **Flask (Jinja2)** and focuses on usability for engineers.

* **Real-Time Inference:** Predictions are generated instantly upon form submission.
* **Dark Mode Support:** A toggleable theme (managed via JavaScript/CSS) allows users to switch between Light and Dark modes, reducing eye strain during long analysis sessions.
* **Responsive Design:** The interface adapts to different screen sizes.

---

## 📂 Project Structure

A breakdown of the repository's file organization:

```text
TensileForge/
├── app.py                 # The main Flask application server
├── main.py                # (Optional) Alternative entry point or testing script
├── requirements.txt       # List of python dependencies
│
├── dataset/
│   └── steel_data.csv     # The raw dataset used for training models
│
├── models/
│   ├── xgboost_model.json # Serialized XGBoost model
│   └── nn_model.pth       # Serialized PyTorch model weights
│
├── scalers/
│   ├── scaler_X.pkl       # Pre-fitted StandardScaler for features
│   └── label_encoder.pkl  # Pre-fitted LabelEncoder for categorical data
│
├── train/
│   ├── train_xgb.py       # Script to train and save the XGBoost model
│   └── train_nn.py        # Script to train and save the PyTorch model
│
├── static/
│   ├── css/style.css      # Styling for the web interface
│   └── js/script.js       # Logic for Dark Mode and UI interactions
│
└── templates/
    └── index.html         # Main dashboard interface

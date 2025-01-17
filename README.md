# Covid-19 Regression Analysis

This project demonstrates a regression analysis using a Covid-19 dataset to predict the number of deaths based on other variables such as confirmed cases, recoveries, and active cases.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Steps and Methodology](#steps-and-methodology)
4. [Model Performance](#model-performance)
5. [How to Run](#how-to-run)
6. [Results](#results)
7. [Future Improvements](#future-improvements)

---

## Project Overview
The goal of this project is to build and evaluate a regression model to estimate the number of deaths due to Covid-19 based on specific features in the dataset. The project implements a deep learning regression model using TensorFlow and provides visualizations to analyze model performance.

---

## Dataset
- **Source:** The dataset was sourced from Kaggle (Covid-19 Clean Complete Dataset).
- **Features Used:**
  - Confirmed cases
  - Recovered cases
  - Active cases
  - Deaths (target variable)

The dataset was preprocessed to handle missing values and scaled to ensure the model's performance.

---

## Steps and Methodology
1. **Data Preprocessing:**
   - Handled missing values by filling them with zeros.
   - Scaled data using MinMaxScaler to normalize feature values.

2. **Model Building:**
   - Used a Sequential neural network with multiple dense layers and dropout for regularization.
   - Configured the model with the Adam optimizer and mean squared error (MSE) as the loss function.

3. **Training:**
   - Trained the model for 300 epochs with a batch size of 128.
   - Used ReduceLROnPlateau callback to adjust the learning rate dynamically.

4. **Evaluation:**
   - Evaluated model performance using R² Score, Mean Absolute Error (MAE), and Mean Squared Error (MSE).
   - Visualized training loss and predicted vs. actual values.

---

## Model Performance
- **R² Score:** 0.83
- **MAE:** Approximately 413 deaths
- **MSE:** 2,691,921

The model shows strong predictive power with minimal overfitting due to dropout and learning rate adjustments.

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/emreyoruk64/REGRESYON.git
   ```
2. Navigate to the project directory:
   ```bash
   cd REGRESYON
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Covid19Regresyon.ipynb
   ```

---

## Results
- **Training vs. Prediction Visualization:**
  The scatter plot between actual and predicted values shows a strong correlation, with most points closely aligned along the diagonal.
- **Training Loss:**
  The training loss decreased significantly, indicating effective learning during training.

---

## Future Improvements
1. Explore additional features or engineered variables to enhance prediction accuracy.
2. Compare the neural network model with traditional regression techniques like Random Forest or XGBoost.
3. Perform hyperparameter optimization to refine the model further.
4. Add more detailed data exploration and visualization for deeper insights.




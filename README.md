# 00_KB-2_Secondary-Mushroom

## Project Overview

This repository contains code and models related to the analysis and classification of mushroom data. The project leverages machine learning techniques to predict mushroom characteristics, likely edibility or other relevant properties. This project uses `Logistic Regression`.

## Key Features & Benefits

*   **Mushroom Classification:** Implements machine learning models for classifying mushrooms based on their features.
*   **Pre-trained Model:** Includes a pre-trained Logistic Regression model for immediate use.
*   **Data Preprocessing:** Provides preprocessing steps for cleaning and preparing the mushroom dataset.
*   **Model Persistence:** Saves trained models and related artifacts for reproducibility and deployment.

## Prerequisites & Dependencies

Before running the code in this repository, you'll need the following:

*   **Python 3.x:**  Ensure you have Python 3 installed.
*   **Jupyter Notebook:** The primary code is in a Jupyter Notebook.
*   **Required Python Packages:**
    *   `pandas`
    *   `scikit-learn`
    *   `pickle`
    *   `joblib`
    *   (Other libraries used in the notebook. Review the notebook for full list).

You can install these packages using `pip`:

```bash
pip install pandas scikit-learn joblib
```

## Installation & Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/intxdv/00_KB-2_Secondary-Mushroom.git
    cd 00_KB-2_Secondary-Mushroom
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt # if a requirements.txt file is created.
    # OR install manually as described in Prerequisites section
    ```

3.  **Open the Jupyter Notebook:**
    ```bash
    jupyter notebook 00_KB-2_Secondary Mushroom.ipynb
    ```

## Usage Examples

Open the `00_KB-2_Secondary Mushroom.ipynb` notebook and follow the instructions within to load data, preprocess it, and use the pre-trained model for predictions.

**Loading the Model:**

The notebook provides example code to load the saved model and related objects. Here's an example (check the notebook for the most up-to-date code):

```python
import joblib

model = joblib.load("models/best_model_Logistic_Regression_20251124_074406.pkl")
preprocessor = joblib.load("models/preprocessor_20251124_074406.pkl")
# ... load other artifacts like label_encoder, feature_info if needed.
```

**Making Predictions:**

```python
# Assuming you have new mushroom data in a DataFrame called 'new_data'
processed_data = preprocessor.transform(new_data) # Preprocess the new data
predictions = model.predict(processed_data) # Make predictions
print(predictions)
```

## Configuration Options

There are no specific configuration files in this repository outside of the model and preprocessor files. The primary configurations exist within the `00_KB-2_Secondary Mushroom.ipynb` notebook. You can modify the data loading, preprocessing steps, and model parameters within the notebook.
The most important configruation options can be found in `models/`. Namely:
* `best_model_Logistic_Regression_20251124_074406.pkl`: The Logistic Regression Model.
* `preprocessor_20251124_074406.pkl`: The preprocessing steps applied to data.
* `label_encoder_20251124_074406.pkl`: The mapping used for labels.
* `feature_info_20251124_074406.pkl`: Information about the features used.
* `metadata_20251124_074406.pkl`: Other metadata about the training of the model.

## Contributing Guidelines

Contributions are welcome! If you'd like to contribute to this project, please follow these guidelines:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with descriptive commit messages.
4.  Submit a pull request.

## License Information

License not specified. All rights reserved.

## Acknowledgments

This project utilizes the [Scikit-learn](https://scikit-learn.org/stable/) library for machine learning tasks.

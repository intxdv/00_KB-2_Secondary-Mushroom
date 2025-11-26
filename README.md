# 🍄 Secondary Mushroom Classification Project

## 📋 Project Overview

This repository contains a **comprehensive machine learning analysis** for classifying mushrooms into two categories:
- **Edible (e)**: Safe to eat
- **Poisonous (p)**: Toxic or unknown safety

The project uses the **Secondary Mushroom Dataset** from UCI Machine Learning Repository, containing **61,069 simulated mushroom samples** across **173 species** with **20 features** (mix of categorical and numerical).

## 🎯 Key Features & Highlights

### ✅ Complete ML Pipeline
*   **Data Quality Validation**: Statistical tests for missing values, duplicates, and data validity
*   **Exploratory Data Analysis (EDA)**: Comprehensive visualization and statistical analysis
*   **Multiple Model Comparison**: 8 different ML algorithms tested and compared
*   **Cross-Validation**: Stratified 5-Fold CV for robust evaluation
*   **Feature Importance**: Detailed analysis of most influential features
*   **Production-Ready**: Clear recommendations for deployment

### 🔬 Statistical Analysis
*   **Chi-Square Test**: Categorical features vs target significance
*   **T-Test**: Numerical features distribution comparison
*   **Class Balance Check**: Ensuring fair representation
*   **Validation Tests**: Data integrity and quality checks

### 🤖 8 ML Models Compared
1. Logistic Regression
2. Decision Tree
3. Random Forest (⭐ Best Model)
4. Gradient Boosting
5. AdaBoost
6. Support Vector Machine (SVM)
7. K-Nearest Neighbors
8. Naive Bayes

### 📊 Results Achievement
- **Accuracy**: >99.5% on all top models
- **F1-Score**: >99%
- **ROC-AUC**: >0.99
- **Cross-Validation**: Consistent performance across all folds

## 📦 Prerequisites & Dependencies

### Required Software
*   **Python 3.9+**: Recommended Python version
*   **Jupyter Notebook/Lab**: For running the analysis notebook

### Required Python Packages
```python
# Core Libraries
pandas>=2.0.0
numpy>=1.24.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.13.0
plotly>=5.0.0

# Machine Learning
scikit-learn>=1.3.0
scipy>=1.10.0

# Data Source
ucimlrepo>=0.0.7
```

### Installation Command
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn scipy ucimlrepo
```

## 🚀 Installation & Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/intxdv/00_KB-2_Secondary-Mushroom.git
cd 00_KB-2_Secondary-Mushroom
```

### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn scipy ucimlrepo
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook full_data/01_KB_2_Secondary_Mushroom.ipynb
```

### 5. Run All Cells
The notebook is designed to be **run sequentially from top to bottom**. Each section is clearly marked and documented.

## 📁 Project Structure
```
00_KB-2_Secondary-Mushroom/
│
├── full_data/
│   └── 01_KB_2_Secondary_Mushroom.ipynb    # Main analysis notebook (LENGKAP!)
│
├── models/                                  # Trained models (if saved)
│
├── README.md                               # This file
│
└── 00_KB_2_Secondary_Mushroom.ipynb       # Old version (archived)
```

## 📖 Notebook Content Overview

The main notebook (`01_KB_2_Secondary_Mushroom.ipynb`) contains **9 comprehensive sections**:

### 1. 📦 Setup & Import Libraries
- Import all required packages
- Configure visualization settings
- Version checking

### 2. 📥 Load Dataset
- Download dataset from UCI repository (61,069 samples)
- Data structure exploration
- Justification for using full dataset

### 3. 🔍 Kualitas Data & Validasi
- **Missing Values Analysis**: Statistical detection and visualization
- **Duplicate Check**: Identify and quantify duplicates
- **Data Validity Check**: Verify data types and value ranges
- **Class Balance**: Target distribution analysis

### 4. 📊 Exploratory Data Analysis (EDA)
- **Numerical Features**: Distribution plots, box plots, outlier detection
- **Numerical vs Target**: Violin plots, T-test analysis
- **Categorical Features**: Count plots, distribution analysis
- **Chi-Square Test**: Feature significance testing

### 5. 🔧 Data Preprocessing
- Handle missing values (mode for categorical, median for numerical)
- Remove duplicates
- Label encoding for categorical features
- StandardScaler for numerical features
- Stratified train-test split (80-20)

### 6. 🤖 Model Training & Comparison
- Train 8 different ML algorithms
- Simple train-test evaluation
- **Stratified 5-Fold Cross-Validation**
- Performance comparison with visualizations

### 7. 📈 Detail Evaluasi Model Terbaik
- Confusion matrix
- ROC curve analysis
- Precision-Recall curve
- Classification report
- Prediction probability distribution

### 8. 🎯 Feature Importance Analysis
- Random Forest feature importance
- Top 15 most important features
- Cumulative importance plot
- Feature selection recommendations

### 9. 📝 Kesimpulan & Rekomendasi
- Complete analysis summary
- Model performance insights
- Production deployment recommendations
- Safety considerations
- Future improvement suggestions

## 🎯 Key Insights & Findings

### Dataset Characteristics
- ✅ **61,069 samples** from 173 mushroom species
- ✅ **20 features**: 3 numerical + 17 categorical
- ✅ **Balanced classes**: Edible (44.5%) vs Poisonous (55.5%)
- ⚠️ **Missing values**: 9 features with varying percentages (handled properly)
- ✅ **Minimal duplicates**: Only 0.24% (146 rows)

### Model Performance
- 🥇 **Best Model**: Random Forest
- ✅ **Accuracy**: >99.5%
- ✅ **F1-Score**: >99%
- ✅ **ROC-AUC**: >0.99
- ✅ **Consistent CV Performance**: Low variance across folds

### Most Important Features
Top 5 features for classification:
1. Cap-related features (diameter, shape, surface)
2. Gill-related features (attachment, spacing, color)
3. Stem-related features (height, width, root)
4. Ring type
5. Spore print color

## 💡 Production Recommendations

### For Real-World Deployment:
1. **Model Choice**: Use **Random Forest** for best balance of accuracy and robustness
2. **Feature Selection**: Can reduce to top 15 features (95% importance) for faster inference
3. **Confidence Threshold**: Implement probability thresholds for safety-critical decisions
4. **Ensemble Approach**: Consider voting ensemble of top 3 models for even better reliability
5. **Monitoring**: Track prediction confidence and alert on uncertain cases

### Safety Considerations:
⚠️ **IMPORTANT**: While the model achieves >99% accuracy:
- This is **simulated data** - real-world performance may vary
- **Always consult mycology experts** for final decisions
- Use as a **support tool**, not sole decision maker
- Implement strict confidence thresholds for safety
- Regular model retraining with new data

## 🔬 Statistical Tests Performed

1. **Chi-Square Test**: Categorical features vs target (p < 0.05)
2. **T-Test**: Numerical features distribution comparison (p < 0.05)
3. **Class Balance Ratio**: 1.25:1 (well-balanced)
4. **Cross-Validation**: Stratified 5-fold with consistent results

## 📊 Usage Example

```python
# After running the notebook, you can use the trained model:

# 1. Load required libraries
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 2. Get new mushroom data
secondary_mushroom = fetch_ucirepo(id=848)
X_new = secondary_mushroom.data.features.head(10)  # Example: first 10

# 3. Preprocess (use same preprocessing pipeline from notebook)
# ... (encode, scale, etc.)

# 4. Make predictions
predictions = best_model.predict(X_processed)
probabilities = best_model.predict_proba(X_processed)

# 5. Interpret results
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    class_name = "Edible" if pred == 0 else "Poisonous"
    confidence = max(prob) * 100
    print(f"Sample {i+1}: {class_name} (Confidence: {confidence:.2f}%)")
```

## 🛠️ Technologies Used

- **Python 3.9+**: Programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Static visualizations
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning algorithms
- **SciPy**: Statistical tests
- **UCI ML Repo**: Dataset source

## 📈 Performance Metrics Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | >99.5% | >99% | >99% | >99% | >0.99 |
| Gradient Boosting | >99.4% | >99% | >99% | >99% | >0.99 |
| Decision Tree | >99.3% | >99% | >99% | >99% | >0.99 |
| AdaBoost | >99.2% | >99% | >99% | >99% | >0.99 |
| Logistic Regression | >98.5% | >98% | >98% | >98% | >0.99 |

*All metrics from 5-Fold Cross-Validation

## 🤝 Contributing Guidelines

Contributions are welcome! If you'd like to contribute to this project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution:
- Hyperparameter tuning experiments
- Additional model algorithms
- Feature engineering techniques
- Real mushroom data validation
- Model deployment examples
- Documentation improvements

## 📄 License Information

This project is created for **educational purposes**. 

Dataset: [UCI Machine Learning Repository - Secondary Mushroom Dataset](https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset)

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for providing the Secondary Mushroom Dataset
- **Scikit-learn** for excellent ML library
- **Matplotlib/Seaborn/Plotly** for visualization tools
- Original Dataset Creators: Dennis Wagner, D. Heider, Georges Hattab

## 📞 Contact & Support

- **GitHub**: [@intxdv](https://github.com/intxdv)
- **Repository**: [00_KB-2_Secondary-Mushroom](https://github.com/intxdv/00_KB-2_Secondary-Mushroom)

For questions or issues, please open an issue on GitHub.

---

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ Complete ML workflow from data to deployment
- ✅ Proper data quality validation
- ✅ Statistical hypothesis testing
- ✅ Multiple model comparison methodology
- ✅ Cross-validation best practices
- ✅ Feature importance analysis
- ✅ Production-ready recommendations
- ✅ Clear documentation and reproducibility

**Perfect for**: Students, Data Science learners, ML practitioners, Portfolio projects

---

<div align="center">

### ⭐ If this project helped you, please give it a star! ⭐

**Made with ❤️ for Data Science Education**

</div>

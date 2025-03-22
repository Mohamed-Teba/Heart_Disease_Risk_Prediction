## Heart_Disease_Risk_Prediction

## Project Overview
This project aims to detect heart disease using both a machine learning model and a rule-based expert system. It includes data analysis, model training, and a Streamlit-based UI for user interaction.

## Project Structure
```
Heart_Disease_Detection/
│
├── data/                  # Raw and cleaned datasets
├── ml_model/              # ML model training and prediction scripts
├── rule_based_system/     # Expert system based on medical rules
├── notebooks/             # Jupyter notebooks for data analysis and training
├── reports/               # Accuracy comparison and visualizations
├── ui/                    # Streamlit app for user interaction
├── utils/                 # Data processing utilities
├── requirements.txt       # List of dependencies
└── README.md              # Project documentation
```

## Installation
1. Clone this repository.
2. Install dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
- Run the Streamlit UI:
  ```bash
  streamlit run ui/heart_disease_app.py
  ```
- Open the notebooks in `notebooks/` to analyze the data and train models.

## Dependencies
The project requires the following Python libraries:
- experta
- matplotlib
- pandas
- seaborn
- streamlit

Ensure you have Python 3 installed before running the project.

## Authors
This project was developed for heart disease detection using machine learning and expert systems.

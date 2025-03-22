import streamlit as st
import collections
if not hasattr(collections, 'Mapping'):
    import collections.abc
    collections.Mapping = collections.abc.Mapping
import pandas as pd
import os
from experta import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

class HealthRiskExpert(KnowledgeEngine):
    result = "No Risk Detected"
    predicted_slope = None
    predicted_target = None
    
    @Rule(Fact(cp=3) & Fact(thal=3))
    def high_risk(self):
        self.result = "High Risk detected!"
        self.predicted_target = 1
        self.predicted_slope = 2
    
    @Rule(Fact(oldpeak=P(lambda x: x > 2.5)))
    def high_oldpeak(self):
        self.result = "Warning: High oldpeak value detected!"
        self.predicted_target = 1
        self.predicted_slope = 1
    
    @Rule(Fact(exang=1) & Fact(thalach=P(lambda x: x < 100)))
    def exang_thalach_risk(self):
        self.result = "Moderate Risk: Exang and low thalach detected!"
        self.predicted_target = 1
        self.predicted_slope = 0
    
    @Rule(Fact(cp=2) & Fact(thal=2))
    def moderate_risk(self):
        self.result = "Moderate Risk detected!"
        self.predicted_target = 1
        self.predicted_slope = 1
    
    @Rule(Fact(oldpeak=P(lambda x: x < 0)))
    def low_oldpeak(self):
        self.result = "Low Risk: Negative oldpeak value detected!"
        self.predicted_target = 0
        self.predicted_slope = 2
    
    @Rule(Fact(thalach=P(lambda x: x > 180)))
    def high_thalach(self):
        self.result = "High Risk: Extremely high thalach detected!"
        self.predicted_target = 1
        self.predicted_slope = 2
    
    @Rule(Fact(exang=0) & Fact(oldpeak=P(lambda x: x < 1)))
    def no_exang_low_oldpeak(self):
        self.result = "Low Risk: No Exang and low oldpeak!"
        self.predicted_target = 0
        self.predicted_slope = 2
    
    @Rule(Fact(cp=1) & Fact(thalach=P(lambda x: x < 120)))
    def cp1_low_thalach(self):
        self.result = "Moderate Risk: CP type 1 and low thalach detected!"
        self.predicted_target = 1
        self.predicted_slope = 1
    
    @Rule(Fact(oldpeak=P(lambda x: 1 <= x <= 2)))
    def moderate_oldpeak(self):
        self.result = "Moderate Risk: Oldpeak in moderate range!"
        self.predicted_target = 1
        self.predicted_slope = 1
    
    @Rule(Fact(cp=0) & Fact(exang=0) & Fact(thalach=P(lambda x: x > 160)))
    def low_risk_no_cp_exang(self):
        self.result = "Low Risk: No CP, no Exang, and high thalach!"
        self.predicted_target = 0
        self.predicted_slope = 2

file_path = r"C:\Users\mrahm\Downloads\heart_disease_project_restructured\data\cleaned_data3.csv"

def load_data():
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()

def train_decision_tree(df):
    features = ['cp', 'thal', 'oldpeak', 'exang', 'thalach']
    X = df[features]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')  
    
    joblib.dump(model, r"C:\Users\mrahm\Downloads\heart_disease_project_restructured\ml_model\heart_disease_model.pkl")
    
    return model, acc, prec, rec, f1


def plot_comparison(df, user_data, column):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.histplot(df[column], bins=20, kde=True, ax=ax, color='skyblue', label="Dataset")
    
    ax.axvline(user_data[column][0], color='red', linestyle='--', label="User Input")
    
    ax.set_title(f"Comparison of {column} (User vs Dataset)")
    ax.legend()
    st.pyplot(fig)

def main():
    st.title("🩺 Heart Disease Risk Prediction")
    
    st.sidebar.header("User Input")
    age = st.sidebar.slider("Age", 20, 100, 50)
    sex = st.sidebar.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    cp = st.sidebar.selectbox("cp (Chest Pain Type)", [0, 1, 2, 3])
    trestbps = st.sidebar.slider("trestbps (Resting Blood Pressure)", 80, 200, 120)
    chol = st.sidebar.slider("chol (Serum Cholesterol)", 100, 600, 200)
    fbs = st.sidebar.selectbox("fbs (Fasting Blood Sugar > 120 mg/dl)", [0, 1])
    restecg = st.sidebar.selectbox("restecg (Resting ECG)", [0, 1, 2])
    thalach = st.sidebar.slider("thalach (Max Heart Rate)", 60, 220, 150)
    exang = st.sidebar.selectbox("exang (Exercise Induced Angina)", [0, 1])
    oldpeak = st.sidebar.slider("oldpeak", -3.0, 3.0, 0.0)
    ca = st.sidebar.slider("ca (Number of Major Vessels Colored by Flourosopy)", 0, 4, 0)
    thal = st.sidebar.selectbox("thal (Thalassemia)", [2, 3])
    
    if thalach > 160:
        slope = 2  
    elif thalach > 120:
        slope = 1
    else:
        slope = 0  
    
    user_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })
    
    if st.sidebar.button("Predict Risk"):
        engine = HealthRiskExpert()
        engine.reset()
        engine.declare(
            Fact(age=age), Fact(sex=sex), Fact(cp=cp), Fact(trestbps=trestbps),
            Fact(chol=chol), Fact(fbs=fbs), Fact(restecg=restecg), Fact(thalach=thalach),
            Fact(exang=exang), Fact(oldpeak=oldpeak), Fact(slope=slope), Fact(ca=ca), Fact(thal=thal)
        )
        engine.run()
        st.success(f"🚨 {engine.result}")
        st.write(f"Predicted Target: {engine.predicted_target}")
        st.write(f"Predicted Slope: {engine.predicted_slope}")
    
    df = load_data()
    
    if not df.empty:
        column_to_plot = st.selectbox("Select Column to Visualize", df.columns)
        plot_comparison(df, user_data, column_to_plot)
        model, acc, prec, rec, f1 = train_decision_tree(df)
        st.write(f"Accuracy: {acc:.2f}")
        st.write(f"Precision: {prec:.2f}")
        st.write(f"Recall: {rec:.2f}")
        st.write(f"F1 Score: {f1:.2f}")
    else:
        st.warning("No data available yet. Enter some inputs to populate the dataset.")

if __name__ == "__main__":
    main()

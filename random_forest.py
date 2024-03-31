import streamlit as st
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# Load data
@st.cache
def load_data():
    df = pd.read_csv("admission_predict.csv")
    df = df.rename(columns={'GRE Score': 'GRE', 'TOEFL Score': 'TOEFL', 'LOR ': 'LOR', 'Chance of Admit ': 'Probability'})
    return df

df = load_data()

# Preprocess data
scaler = MinMaxScaler()
X = df[['GRE', 'TOEFL', 'LOR', 'CGPA', 'SOP', 'Research', 'University Rating']]
X_scaled = scaler.fit_transform(X)
y = df["Probability"]

# Train models
import joblib

# Train models
def train_models(X, y):
    models = {
        'Random Forest Regression': {
            'model': RandomForestRegressor(),
            'parameters': {
                'n_estimators': [5,10,15,20],
                'criterion': ['mse', 'friedman_mse', 'mae', 'poisson']
            }
        }
    }
    
    trained_models = {}
    best_model_name = None
    best_score = float('-inf')
    
    for model_name, model_params in models.items():
        gs = GridSearchCV(model_params['model'], model_params['parameters'], cv=5, return_train_score=False)
        gs.fit(X, y)
        
        if gs.best_score_ > best_score:
            best_score = gs.best_score_
            best_model_name = model_name
            
        trained_models[model_name] = {'model': gs.best_estimator_, 'score': gs.best_score_}
        
        # Save the trained model if it's Random Forest Regression
        if model_name == 'Random Forest Regression':
            joblib.dump(gs.best_estimator_, "random_forest_regression_model.pkl")
        
    return trained_models, best_model_name

# Train models and get best model name
trained_models, best_model_name = train_models(X_scaled, y)


# UI
st.title("Admission Prediction")

# Sidebar
st.sidebar.title("Predict Admission Probability")
gre = st.sidebar.slider("GRE Score", min_value=0, max_value=340, step=1, value=300)
toefl = st.sidebar.slider("TOEFL Score", min_value=0, max_value=120, step=1, value=100)
lor = st.sidebar.slider("LOR", min_value=1.0, max_value=5.0, step=0.1, value=3.0)
cgpa = st.sidebar.slider("CGPA", min_value=0.0, max_value=10.0, step=0.1, value=7.5)
sop = st.sidebar.slider("SOP", min_value=1.0, max_value=5.0, step=0.1, value=3.0)
research = st.sidebar.selectbox("Research Experience", ['Yes', 'No'])
university_rating = st.sidebar.slider("University Rating", min_value=1, max_value=5, step=1, value=3)

model_options = list(trained_models.keys())
selected_model = st.sidebar.selectbox("Select Model", model_options)

# Convert Research Experience to binary
research_binary = 1 if research == 'Yes' else 0

# Scale input features
input_features = [[gre, toefl, lor, cgpa, sop, research_binary, university_rating]]
input_features_scaled = scaler.transform(input_features)

# Prediction
predicted_probability = trained_models[selected_model]['model'].predict(input_features_scaled)[0]

# Display Prediction
st.subheader("Predicted Probability of Admission")
st.write(f"Using {selected_model}: {predicted_probability}")

# Display Best Model and its Accuracy
st.subheader("Best Model and Accuracy")
st.write(f"The best model is {best_model_name} with an accuracy of {trained_models[best_model_name]['score']:.2f}")

# Display Data
st.subheader("Admission Prediction Data")
st.write(df)

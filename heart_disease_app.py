# heart_disease_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .safe {
        background-color: #d4edda;
        border: 2px solid #c3e6cb;
        color: #155724;
    }
    .risk {
        background-color: #f8d7da;
        border: 2px solid #f5c6cb;
        color: #721c24;
    }
    .feature-importance {
        background-color: #e2e3e5;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class HeartDiseasePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.accuracy = None
        self.feature_names = [
            'age', 'sex', 'chest pain type', 'resting blood pressure', 
            'serum cholestoral', 'fasting blood sugar', 
            'resting electrocardiographic results', 'max heart rate', 
            'exercise induced angina', 'oldpeak', 'ST segment', 
            'major vessels', 'thal'
        ]
        
    def load_and_preprocess_data(self):
        """Load and preprocess the heart disease dataset"""
        try:
            # Load dataset
            df = pd.read_csv('dataset_heart.csv')
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Remove duplicates
            df = df.drop_duplicates()
            
            return df
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return None
    
    def train_model(self, df):
        """Train the logistic regression model"""
        try:
            # Separate features and target
            X = df.drop('heart disease', axis=1)
            y = df['heart disease']
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = LogisticRegression(random_state=42, max_iter=1000)
            self.model.fit(X_train_scaled, y_train)
            
            # Calculate accuracy
            y_pred = self.model.predict(X_test_scaled)
            self.accuracy = accuracy_score(y_test, y_pred)
            
            return True
        except Exception as e:
            st.error(f"Error training model: {e}")
            return False
    
    def predict(self, input_features):
        """Make prediction on new data"""
        try:
            if self.model is None or self.scaler is None:
                return None, None
            
            # Scale input features
            input_scaled = self.scaler.transform([input_features])
            
            # Make prediction
            prediction = self.model.predict(input_scaled)[0]
            probability = self.model.predict_proba(input_scaled)[0][1]
            
            return prediction, probability
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            return None, None
    
    def get_feature_importance(self):
        """Get feature importance from the model"""
        if self.model is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': abs(self.model.coef_[0])
        }).sort_values('importance', ascending=False)
        
        return importance_df

def main():
    # Initialize predictor
    predictor = HeartDiseasePredictor()
    
    # Title
    st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction App</h1>', 
               unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose App Mode", 
                                   ["🏠 Home", "📊 Data Analysis", "🔍 Predict", "ℹ️ About"])
    
    # Load data and train model
    if 'model_trained' not in st.session_state:
        with st.spinner("Loading data and training model..."):
            df = predictor.load_and_preprocess_data()
            if df is not None:
                success = predictor.train_model(df)
                if success:
                    st.session_state.model_trained = True
                    st.session_state.df = df
                    st.session_state.predictor = predictor
                else:
                    st.error("Failed to train model. Please check your dataset.")
                    return
            else:
                st.error("Failed to load dataset. Please ensure 'dataset_heart.csv' is in the same directory.")
                return
    
    df = st.session_state.df
    predictor = st.session_state.predictor
    
    # Home Page
    if app_mode == "🏠 Home":
        st.subheader("Welcome to the Heart Disease Prediction System")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("""
            This application uses machine learning to predict the likelihood of heart disease 
            based on various medical parameters and lifestyle factors.
            
            ### How it works:
            1. **Data Analysis**: Explore the heart disease dataset with interactive visualizations
            2. **Prediction**: Enter patient information to get a heart disease risk assessment
            3. **Results**: Get instant predictions with probability scores and recommendations
            
            ### Model Performance:
            """)
            
            if predictor.accuracy:
                st.metric("Model Accuracy", f"{predictor.accuracy:.2%}")
            
            st.write("""
            ### Medical Disclaimer:
            This tool is for educational purposes only. Always consult healthcare professionals 
            for medical advice and diagnosis.
            """)
        
        with col2:
            st.image("https://cdn-icons-png.flaticon.com/512/2547/2547037.png", 
                    width=200, caption="Heart Health Monitoring")
    
    # Data Analysis Page
    elif app_mode == "📊 Data Analysis":
        st.header("📊 Data Analysis & Visualization")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Dataset Overview", "Distributions", "Correlations", "Feature Importance"])
        
        with tab1:
            st.subheader("Dataset Overview")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Dataset Shape:**", df.shape)
                st.write("**First 5 rows:**")
                st.dataframe(df.head())
            
            with col2:
                st.write("**Target Variable Distribution:**")
                fig, ax = plt.subplots(figsize=(8, 6))
                df['heart disease'].value_counts().plot(kind='pie', autopct='%1.1f%%', 
                                                     colors=['lightblue', 'lightcoral'], ax=ax)
                ax.set_ylabel('')
                st.pyplot(fig)
        
        with tab2:
            st.subheader("Feature Distributions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                selected_num = st.selectbox("Select Numerical Feature", 
                                          ['age', 'resting blood pressure', 'serum cholestoral', 
                                           'max heart rate', 'oldpeak'])
                
                fig, ax = plt.subplots(figsize=(10, 6))
                df[selected_num].hist(bins=30, color='skyblue', alpha=0.7, ax=ax)
                ax.set_title(f'Distribution of {selected_num}')
                ax.set_xlabel(selected_num)
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
            
            with col2:
                selected_cat = st.selectbox("Select Categorical Feature", 
                                          ['sex', 'chest pain type', 'fasting blood sugar', 
                                           'exercise induced angina'])
                
                fig, ax = plt.subplots(figsize=(10, 6))
                df[selected_cat].value_counts().plot(kind='bar', color='lightgreen', ax=ax)
                ax.set_title(f'Distribution of {selected_cat}')
                ax.set_xlabel(selected_cat)
                ax.set_ylabel('Count')
                plt.xticks(rotation=45)
                st.pyplot(fig)
        
        with tab3:
            st.subheader("Feature Correlations")
            
            fig, ax = plt.subplots(figsize=(12, 8))
            correlation_matrix = df.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=0.5, ax=ax)
            ax.set_title('Correlation Heatmap of Features')
            st.pyplot(fig)
        
        with tab4:
            st.subheader("Feature Importance")
            
            importance_df = predictor.get_feature_importance()
            if importance_df is not None:
                fig, ax = plt.subplots(figsize=(10, 8))
                y_pos = np.arange(len(importance_df))
                ax.barh(y_pos, importance_df['importance'], color='lightseagreen')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(importance_df['feature'])
                ax.invert_yaxis()
                ax.set_xlabel('Feature Importance (Absolute Coefficient Value)')
                ax.set_title('Logistic Regression Feature Importance')
                st.pyplot(fig)
                
                st.write("**Top 5 Most Important Features:**")
                st.dataframe(importance_df.head())
    
    # Prediction Page
    elif app_mode == "🔍 Predict":
        st.header("🔍 Heart Disease Prediction")
        
        st.write("Enter the patient's information below to assess heart disease risk:")
        
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Personal Information")
                age = st.slider("Age (years)", 20, 100, 50)
                sex = st.radio("Sex", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]
                
                st.subheader("Symptoms")
                chest_pain_type = st.selectbox(
                    "Chest Pain Type",
                    options=[1, 2, 3, 4],
                    format_func=lambda x: {
                        1: "Typical Angina",
                        2: "Atypical Angina", 
                        3: "Non-anginal Pain",
                        4: "Asymptomatic"
                    }[x]
                )
            
            with col2:
                st.subheader("Medical Measurements")
                resting_bp = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
                cholesterol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
                fasting_blood_sugar = st.radio(
                    "Fasting Blood Sugar > 120 mg/dl", 
                    options=[("No", 0), ("Yes", 1)], 
                    format_func=lambda x: x[0]
                )[1]
                max_heart_rate = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
            
            with col3:
                st.subheader("Test Results")
                resting_ecg = st.selectbox(
                    "Resting Electrocardiographic Results",
                    options=[0, 1, 2],
                    format_func=lambda x: {
                        0: "Normal",
                        1: "ST-T Wave Abnormality", 
                        2: "Left Ventricular Hypertrophy"
                    }[x]
                )
                
                exercise_angina = st.radio(
                    "Exercise Induced Angina", 
                    options=[("No", 0), ("Yes", 1)], 
                    format_func=lambda x: x[0]
                )[1]
                
                oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, 0.1)
                
                st_segment = st.selectbox(
                    "ST Segment",
                    options=[1, 2, 3],
                    format_func=lambda x: {
                        1: "Up Sloping",
                        2: "Flat", 
                        3: "Down Sloping"
                    }[x]
                )
            
            col4, col5 = st.columns(2)
            
            with col4:
                major_vessels = st.slider("Major Vessels Colored by Fluoroscopy", 0, 3, 0)
            
            with col5:
                thal = st.selectbox(
                    "Thalassemia",
                    options=[3, 6, 7],
                    format_func=lambda x: {
                        3: "Normal",
                        6: "Fixed Defect", 
                        7: "Reversible Defect"
                    }[x]
                )
            
            submitted = st.form_submit_button("Predict Heart Disease Risk")
        
        if submitted:
            # Prepare input features
            input_features = [
                age, sex, chest_pain_type, resting_bp, cholesterol,
                fasting_blood_sugar, resting_ecg, max_heart_rate,
                exercise_angina, oldpeak, st_segment, major_vessels, thal
            ]
            
            # Make prediction
            prediction, probability = predictor.predict(input_features)
            
            if prediction is not None:
                # Display results
                st.markdown("---")
                st.subheader("Prediction Results")
                
                if prediction >= 0.5:
                    st.markdown(
                        f'<div class="prediction-box risk">'
                        f'<h2>🔴 HIGH RISK OF HEART DISEASE</h2>'
                        f'<h3>Probability: {probability:.1%}</h3>'
                        f'</div>', 
                        unsafe_allow_html=True
                    )
                    
                    st.warning("""
                    **Recommendation:** 
                    - Consult a cardiologist immediately
                    - Consider lifestyle changes (diet, exercise)
                    - Monitor blood pressure regularly
                    - Schedule follow-up tests
                    """)
                else:
                    st.markdown(
                        f'<div class="prediction-box safe">'
                        f'<h2>🟢 LOW RISK OF HEART DISEASE</h2>'
                        f'<h3>Probability: {probability:.1%}</h3>'
                        f'</div>', 
                        unsafe_allow_html=True
                    )
                    
                    st.success("""
                    **Recommendation:** 
                    - Continue regular health check-ups
                    - Maintain healthy lifestyle
                    - Monitor risk factors periodically
                    """)
                
                # Show confidence level
                confidence = "High" if (probability > 0.7 or probability < 0.3) else "Medium"
                st.info(f"**Confidence Level:** {confidence}")
                
                # Show key influencing factors
                st.subheader("Key Factors Influencing This Prediction")
                importance_df = predictor.get_feature_importance()
                if importance_df is not None:
                    top_3 = importance_df.head(3)
                    
                    for _, row in top_3.iterrows():
                        feature_idx = predictor.feature_names.index(row['feature'])
                        user_value = input_features[feature_idx]
                        st.write(f"**{row['feature']}:** {user_value}")
    
    # About Page
    else:
        st.header("ℹ️ About This App")
        
        st.write("""
        ### Heart Disease Prediction Application
        
        This web application uses machine learning to predict the likelihood of heart disease 
        based on various medical parameters.
        
        #### Features:
        - **Data Analysis**: Interactive visualizations of the heart disease dataset
        - **Real-time Prediction**: Instant heart disease risk assessment
        - **Feature Importance**: Understand which factors most influence predictions
        - **Medical Recommendations**: Actionable insights based on predictions
        
        #### Technical Details:
        - **Algorithm**: Logistic Regression
        - **Dataset**: Heart Disease Diagnosis Dataset from Kaggle
        - **Preprocessing**: Standard scaling, duplicate removal
        - **Validation**: 80-20 train-test split
        
        #### Model Features Used:
        1. Age
        2. Sex
        3. Chest Pain Type
        4. Resting Blood Pressure
        5. Serum Cholesterol
        6. Fasting Blood Sugar
        7. Resting ECG Results
        8. Maximum Heart Rate
        9. Exercise Induced Angina
        10. ST Depression (oldpeak)
        11. ST Segment
        12. Major Vessels
        13. Thalassemia
        
        #### Disclaimer:
        This application is for educational and informational purposes only. 
        It is not a substitute for professional medical advice, diagnosis, or treatment. 
        Always seek the advice of your physician or other qualified health provider 
        with any questions you may have regarding a medical condition.
        """)

if __name__ == "__main__":
    main()

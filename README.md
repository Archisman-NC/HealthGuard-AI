# HealthGuard-AI: Intelligent Patient Risk Assessment System 🏥 (Milestone 1)

## Problem Understanding & Healthcare Use-Case
Diabetes is a chronic disease that affects millions of people globally and can lead to severe health complications if not properly managed. Early diagnosis and proactive intervention are critical. The objective of the **Intelligent Patient Risk Assessment System** is to build a machine learning model capable of predicting the onset of diabetes based on diagnostic measurements. By providing a quick, accessible risk assessment tool, this system empowers patients and enables healthcare professionals to identify high-risk individuals early.

## Dataset
The dataset used is the [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). It contains clinical data designed diagnostically to predict whether a patient has diabetes based on several physiological measurements.

## Input-Output Specification
### Model Inputs (Features)
The system accepts structured clinical diagnostic data including:
- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration (2 hours in an oral glucose tolerance test)
- **Blood Pressure**: Diastolic blood pressure (mm Hg)
- **Skin Thickness**: Triceps skinfold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **Diabetes Pedigree Function**: A synthesized metric scoring diabetes likelihood based on family history
- **Age**: Patient's age in years

### System Outputs
- **Risk Score**: The predicted predictive probability (0-100%) indicating the likelihood of diabetes.
- **Risk Category**: Actionable health categorization into Low, Moderate, or High risk.
- **Top Contributing Factors**: The system natively extracts and explains the top 3 specific diagnostic features raising or reducing the patient's individual calculated risk in real time.


## Model Performance Analysis
The selected ML model is **Logistic Regression**. It was deemed ideal due to its interpretability compared to opaque neural networks and clear decision boundaries on binary classification sets. 
- **Preprocessing:** Incorrect or missing zero-values in biological features (e.g., Blood Pressure, Glucose) were imputed with sequence medians, substantially improving data integrity. 
- **Performance Evaluation:** The model was tested on a 20% holdout test dataset. Accuracy and strictly evaluated ROC-AUC scores dynamically generated during the compilation phase show significant reliability in successfully classifying positive instances of diabetes risk versus negative ones.

## Running the Application Locally
1. Ensure Python 3.8+ is installed locally.
2. Install the application's required libraries:
   ```bash
   pip install pandas numpy scikit-learn streamlit joblib
   ```
3. Execute the training script to fetch the `.csv` data, train the model, and create the necessary predictive binaries:
   ```bash
   python model_training.py
   ```
4. Run the interactive Streamlit user-interface:
   ```bash
   streamlit run app.py
   ```
5. Open your browser to the local URL provided by Streamlit to query health predictions.

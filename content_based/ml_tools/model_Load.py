import joblib
def load_models():
    dt_model = joblib.load('content_based/ml_models/dt_model.pkl')
    rf_model = joblib.load('content_based/ml_models/rf_model.pkl')
    xgb_model = joblib.load('content_based/ml_models/xgb_model.pkl')
    return {
        'Decision Tree': dt_model,
        'Random Forest': rf_model,
       'XGBoost Classifier':xgb_model,
    }
def predict(model, vector):
    prediction = model.predict(vector)
    return prediction

import streamlit as st
import pandas as pd
import joblib


# Load the preprocessor and model
preprocessor = joblib.load('D:/Projects/Concrete_Prediction/Concrete_Compressive_Strength_Prediction/model/compressive_strength_preprocessor.joblib')
model = joblib.load('D:/Projects/Concrete_Prediction/Concrete_Compressive_Strength_Prediction/model/compressive_strength_model.joblib')

def predict_compressive_strength(cement, blast_furnace_slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age):
    data_dict = {
        'cement': float(cement),
        'blast_furnace_slag': float(blast_furnace_slag),
        'fly_ash': float(fly_ash),
        'water': float(water),
        'superplasticizer': float(superplasticizer),
        'coarse_aggregate': float(coarse_aggregate),
        'fine_aggregate ': float(fine_aggregate),
        'age': int(age)
    }

    data = pd.DataFrame([data_dict])
    transformed_data = preprocessor.transform(data)
    prediction = model.predict(transformed_data)

    return prediction[0]

def main():
    st.title('Concrete Compressive Strength Prediction')

    # Input fields for user to enter values
    cement = st.text_input('Cement', '300.0')
    blast_furnace_slag = st.text_input('Blast Furnace Slag', '150.0')
    fly_ash = st.text_input('Fly Ash', '100.0')
    water = st.text_input('Water', '200.0')
    superplasticizer = st.text_input('Superplasticizer', '15.0')
    coarse_aggregate = st.text_input('Coarse Aggregate', '1000.0')
    fine_aggregate = st.text_input('Fine Aggregate', '750.0')
    age = st.text_input('Age (days)', '28')

    # Button to make predictions
    if st.button('Predict Compressive Strength'):
        prediction = predict_compressive_strength(cement, blast_furnace_slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age)
        st.success(f'Predicted Compressive Strength: {prediction:.2f} MPa')

if __name__ == '__main__':
    main()

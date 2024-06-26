import joblib
import pandas as pd

def get_input_for_model1():
    n = float(input("Enter value for n: "))
    p = float(input("Enter value for p: "))
    k = float(input("Enter value for k: "))
    temperature = float(input("Enter temperature: "))
    humidity = float(input("Enter humidity: "))
    ph = float(input("Enter pH: "))
    rainfall = float(input("Enter rainfall: "))

    return {'n': n, 'p': p, 'k': k, 'temperature': temperature, 'humidity': humidity, 'ph': ph, 'rainfall': rainfall}

def get_input_for_model2(predicted_crop):
    area = input("Enter area: ")
    item = predicted_crop
    year = int(input("Enter year: "))
    avg_rainfall = float(input("Enter average rainfall (mm per year): "))
    pesticides = float(input("Enter pesticides (tonnes): "))
    avg_temperature = float(input("Enter average temperature: "))

    # Specify the column names based on the actual requirements of your second model
    columns = ['Area', 'Item', 'Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']

    return pd.DataFrame([[area, item, year, avg_rainfall, pesticides, avg_temperature]], columns=columns)

def main():
    # Load the models
    model1 = joblib.load('model_joblib.pkl')
    model2 = joblib.load('sree.joblib')

    # Get input for Model 1
    input_data_model1 = get_input_for_model1()

    # Predict using Model 1
    predicted_crop = model1.predict([[input_data_model1['n'], input_data_model1['p'], input_data_model1['k'],
                                      input_data_model1['temperature'], input_data_model1['humidity'],
                                      input_data_model1['ph'], input_data_model1['rainfall']]])[0]
    predicted_crop = predicted_crop.title()
    

    print(f"Predicted Crop: {predicted_crop}")

    # Get input for Model 2
    input_data_model2 = get_input_for_model2(predicted_crop)

    # Predict using Model 2
    hg_ha_yield = model2.predict(input_data_model2)[0]

    print(f"Predicted hg/ha_yield: {hg_ha_yield}")

if __name__ == "__main__":
    main()

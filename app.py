from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)




def get_input_for_model1():
    n = float(request.form['n'])
    p = float(request.form['p'])
    k = float(request.form['k'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    return {'n': n, 'p': p, 'k': k, 'temperature': temperature, 'humidity': humidity, 'ph': ph, 'rainfall': rainfall}

def get_input_for_model2():
    
    area = request.form.get('area', 'default_value')
    item = request.form.get('pre', 'default_value')
    year = int(request.form.get('year', 2023))  # Use a default value or handle the absence of 'year'
    avg_rainfall = float(request.form.get('avg_rainfall', 0.0))  # Use default values for other fields as needed
    pesticides = float(request.form.get('pesticides', 0.0))
    avg_temperature = float(request.form.get('avg_temperature', 0.0))

    columns = ['Area', 'Item', 'Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']

    return pd.DataFrame([[area, item, year, avg_rainfall, pesticides, avg_temperature]], columns=columns)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    model1 = joblib.load('model_joblib.pkl')
    model2 = joblib.load('sree.joblib')

    input_data_model1 = get_input_for_model1()

    predicted_crop = model1.predict([[input_data_model1['n'], input_data_model1['p'], input_data_model1['k'],
                                      input_data_model1['temperature'], input_data_model1['humidity'],
                                      input_data_model1['ph'], input_data_model1['rainfall']]])[0]
    predicted_crop = predicted_crop.title()
    

    return render_template('result.html', predicted_crop=predicted_crop)

   # input_data_model2 = get_input_for_model2(predicted_crop)

   # hg_ha_yield = model2.predict(input_data_model2)[0]
    #hg_ha_yield = predict_yield(input_data_model2.iloc[0].to_dict())

   

@app.route('/predict_yield', methods=['POST'])
def predict_yield():
    model2 = joblib.load('sree.joblib')
    input_data_model2 = get_input_for_model2()

    hg_ha_yield = model2.predict(input_data_model2)[0]
    # Assuming model2 is already loaded
    

    # Assuming model2 predicts yield
    return render_template('result2.html', hg_ha_yield=hg_ha_yield)

  

    






if __name__ == "__main__":
    app.run(debug=True)

    
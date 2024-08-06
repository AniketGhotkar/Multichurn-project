from flask import Flask, render_template, request, redirect, url_for, jsonify
import requests
import pickle
import numpy as np
import sklearn
import matplotlib
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

# Load bank customer churn prediction model
bank_model = pickle.load(open('Churn_Prediction.pkl', 'rb'))

# Load employee churn prediction model
employee_model = pickle.load(open('model.pkl', 'rb'))
employee_scaler = pickle.load(open('scaler.pkl', 'rb'))


standard_to = StandardScaler()



@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        churn_type = request.form.get('churn_type')
        if churn_type == 'bank':
            return redirect(url_for('bank_churn'))
        elif churn_type == 'employee':
            return redirect(url_for('employee_churn'))
    else:
        return render_template('index.html')


@app.route('/bank_churn', methods=['GET', 'POST'])
def bank_churn():
    if request.method == 'POST':
        # Extract features for bank churn prediction
        CreditScore = int(request.form['CreditScore'])
        Age = int(request.form['Age'])
        Tenure = int(request.form['Tenure'])
        Balance = float(request.form['Balance'])
        NumOfProducts = int(request.form['NumOfProducts'])
        HasCrCard = int(request.form['HasCrCard'])
        IsActiveMember = int(request.form['IsActiveMember'])
        EstimatedSalary = float(request.form['EstimatedSalary'])
        Geography_Germany = request.form['Geography_Germany']
         # Preprocess features
        if(Geography_Germany == 'Germany'):
            Geography_Germany = 1
            Geography_Spain= 0
            Geography_France = 0
                
        elif(Geography_Germany == 'Spain'):
            Geography_Germany = 0
            Geography_Spain= 1
            Geography_France = 0
        
        else:
            Geography_Germany = 0
            Geography_Spain= 0
            Geography_France = 1
        Gender_Male = request.form['Gender_Male']
        if(Gender_Male == 'Male'):
            Gender_Male = 1
            Gender_Female = 0
        else:
            Gender_Male = 0
            Gender_Female = 1
        # Make prediction using bank model
        prediction = bank_model.predict([[CreditScore,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Geography_Germany,Geography_Spain,Gender_Male]])
        # prediction_probability = bank_model.predict_proba([[CreditScore,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Geography_Germany,Geography_Spain,Gender_Male]])

        # # Extracting the probability of the positive class (class 1)
        # probability_of_leaving = prediction_probability[0][1]

        # # Convert probability to percentage
        # probability_percentage = round(probability_of_leaving * 100, 2) 
        # noprob_percentage = 100 - probability_percentage

        # print("{probability_percentage} is leave")
        # print(noprob_percentage)
        # Render prediction result
        # 4 th is predicted as churn for example
        if prediction==1:
             return render_template('bank_churn.html',prediction_text="The Customer will leave the bank")
        else:
             return render_template('bank_churn.html',prediction_text="The Customer will not leave the bank")
    else:
        return render_template('bank_churn.html')


@app.route('/employee_churn', methods=['GET', 'POST'])
def employee_churn():
    if request.method == 'POST':
        # Extracting features from the request
        features = [
            float(request.form['satisfaction_level']),
            float(request.form['last_evaluation']),
            int(request.form['number_project']),
            int(request.form['average_monthly_hours']),
            int(request.form['time_spend_company']),
            int(request.form['work_accident']),
            int(request.form['promotion_last_5years']),
            request.form['salary']
        ]
        
        # Encoding salary category
        salary_mapping = {'low': 0, 'medium': 1, 'high': 2}
        features[-1] = salary_mapping[features[-1]]
        
        features_array = np.array(features).reshape(1, -1)

        # Scale the features using the loaded scaler
        scaled_features = employee_scaler.transform(features_array)

        # Making prediction
        prediction = employee_model.predict(scaled_features)
        
        # Returning prediction result
        if prediction == 1:
            return render_template('employee_churn.html', prediction_text="The employee will leave.")
        else:
            return render_template('employee_churn.html', prediction_text="The employee will stay.")
    else:
        return render_template('employee_churn.html')



if __name__=="__main__":
    app.run(debug=True)




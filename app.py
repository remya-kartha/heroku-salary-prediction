from types import MethodDescriptorType
from flask import Flask,render_template,request
import joblib

# start
app = Flask(__name__)

#load the model
model = joblib.load('hiring_model.pkl')


@app.route('/predict',methods=['POST'])
def predict():
    exp= request.form.get('experience')
    score= request.form.get('test_score')
    interview_score= request.form.get('interview_score')   

    prediction = model.predict([[int(exp),int(score),int(interview_score)]])
    output = round(prediction[0],2)

    return render_template('base.html',prediction_text=f"Employee Salary will be $ {output}")   

@app.route("/")
def index():
    return render_template("base.html")

#run the pgm
app.run(debug = True)
from logging import DEBUG
from flask import Flask,render_template,request
import joblib

# start app
app = Flask(__name__)

@app.route("/")
def main():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    # import the model
    model = joblib.load('hiring_model.pkl')
    experience = request.form.get('experience')
    test_score = request.form.get('test_score')
    interview_score = request.form.get('interview_score')
    prediction = model.predict([[int(experience),int(test_score),int(interview_score)]])
    return render_template('index.html',prediction_text=f"The predicted Salary is ${prediction}")

if __name__ == '__main__':
    print(__name__)
    # run the app
    app.run(debug=True)



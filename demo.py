from flask import Flask, render_template ,request
import pandas as pd
import pickle
import logging
logging.basicConfig(filename='logger.log',level=logging.INFO,format='%(asctime)s %(message)s')

model=pickle.load(open('model/model.pkl','rb'))
preprocessor=pickle.load(open('model/preprocessor.pkl','rb'))

logging.info("Model is loaded succesfully")
app=Flask(__name__)

@app.route("/")
def home_page():
    return render_template('home.html')

@app.route("/predict_churn",methods=['GET','POST'])
def Customer_churn():
    if (request.method=="POST"):
        logging.info("values entered is initiated")
        #return render_template('index.html')
        Geography=request.form['Geography']
        Gender=request.form['Gender']
        CreditScore=float(request.form['CreditScore'])
        Age=float(request.form['Age'])
        Tenure=float(request.form['Tenure'])
        Balance=float(request.form['Balance'])
        NumOfProducts=float(request.form['NumOfProducts'])
        HasCrCard=float(request.form['HasCrCard'])
        IsActiveMember=float(request.form['IsActiveMember'])
        EstimatedSalary=float(request.form['EstimatedSalary'])

        logging.info("Data is stored in variables")

        d1={"Geography":Geography,'Gender':Gender,'CreditScore':CreditScore,'Age':Age,'Tenure':Tenure,
            'Balance':Balance,'NumOfProducts':NumOfProducts,'HasCrCard':HasCrCard,
            'IsActiveMember':IsActiveMember,'EstimatedSalary':EstimatedSalary}
        df1=pd.DataFrame(d1,index=[0])

        logging.info("Dataframe has been generated for prediction")

        scaled_data=preprocessor.transform(df1)
        logging.info("Data is scaled for predicition")

        result=model.predict(scaled_data)
        logging.info(f"REsult is generated and output is {result[0]}")
        output=''
        if result[0] == 1:
            output="**Exited**"
        else:
            output="**Not Exited**"

        return render_template('home.html',results=output)
    else:
        return render_template('index.html')

if __name__=='__main__':
    app.run(host="0.0.0.0",port=8080)
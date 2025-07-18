from flask import Flask ,request,render_template,jsonify
from flask_cors import cross_origin
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.exception import CustomException
import sys
import os

application=Flask(__name__)

app=application

@app.route('/')
@cross_origin()
def home_page():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
@cross_origin()
def predict_datapoint():
    if request.method == 'GET':
       return render_template('index.html')
    
    else:
        try:
            data=CustomData(
            carat = float(request.form.get('carat')),
            depth = float(request.form.get('depth')),
            table = float(request.form.get('table')),
            x = float(request.form.get('x')),
            y = float(request.form.get('y')),
            z = float(request.form.get('z')),
            cut = request.form.get('cut'),
            color= request.form.get('color'),
            clarity = request.form.get('clarity')               
            )

            pred_df=data.get_data_as_dataframe()

            print(pred_df)

            model=PredictPipeline()
            pred=model.predict(pred_df)

            result=round(pred[0],2)
            print("Prediction result : ",result)
            return render_template('index.html',results=result,pred_df=pred_df)
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
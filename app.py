from flask import Flask, make_response,render_template, request, redirect, url_for,jsonify
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
from pylab import rcParams
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
from io import BytesIO


from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def index():	
	return render_template('index.html')

@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/dealer')
def dealer():
    return render_template('dealer.html')

@app.route('/consumer')
def consumer():
    return render_template('consumer.html')   


@app.route('/predict', methods=['GET','POST'])
def predict():

	date = request.args.get('date')
	
	date = pd.to_datetime(date,format='%Y-%m')
	
	future2 = model.fit()
	forecast2 = future2.forecast(steps=25) 
	forecast2 = pd.DataFrame(forecast2)

	pred = future2.get_prediction(start='2016-01-01',end=date)

	asd = pred.predicted_mean.plot(label='Future forecast',alpha=.7, figsize=(16,9))

	sert = pd.DataFrame(pred.predicted_mean)

	x = sert.index
	y = sert[0]

	plt.plot(x,y)
	plt.ylim(25000,125000)
	plt.legend()
	plt.ylabel('Number of LPG bookings provided',fontsize=14)
	plt.xlabel('Date',fontsize=14)
	plt.savefig('img.jpg')

	img = BytesIO()

	plt.savefig(img, format='jpg')
	img.seek(0)
	figdata_jpg = base64.b64encode(img.getvalue()).decode('ascii')
	result = figdata_jpg
	plt.close()

	sert.shape
	
	a = sert[0].iloc[-1]

	return render_template('predict.html', result=result, prediction_text='The number of LPG bookings predicted for the given date {} is {} bookings'.format(date,round(a)))

if __name__ == '__main__':
    app.run(debug=True)
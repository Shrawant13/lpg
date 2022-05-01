import os
import pandas as pd
import numpy as np


os.chdir('/home/shrawant/Downloads')

dealer=pd.read_csv('dealer (1).csv')

dealer=dealer[dealer['dealer'] == 1]

dealer=dealer[['date','lpg provided']]

dealer['date2'] = pd.to_datetime(dealer['date'],format='%d-%m-%y').dt.strftime('%m-%Y')

dealer=dealer.rename(columns={'lpg provided':'lpg_provided'})

dealer['total_lpg_provided'] = dealer.date2.map(dealer.groupby(['date2']).lpg_provided.sum())

dealer=dealer.drop_duplicates(subset='date2',keep='last')

dealer=dealer[['date2','total_lpg_provided']]

dealer['date3'] = pd.to_datetime(dealer['date2'])

dealer=dealer.drop('date2',axis=1)

dealer = dealer.set_index('date3')

from pylab import rcParams
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
#decomposing for more better diagnostic purpose
rcParams['figure.figsize'] = 15, 10
decomposition = sm.tsa.seasonal_decompose(dealer, model='additive')
fig = decomposition.plot()
plt.show()

import itertools

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


import warnings
warnings.filterwarnings('ignore')

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(dealer,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

import statsmodels.api as sm

mod = sm.tsa.statespace.SARIMAX(dealer,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])
results.forecast(steps=5)

from pylab import rcParams
import statsmodels.api as sm
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
pred = results.get_prediction(start='2013-01-01',end='2020-05-01', dynamic=False)
pred_ci = pred.conf_int()
ax = dealer['2013-01-01':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(16, 9))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('date'),
ax.set_ylabel('grams consumed'),
plt.ylim(35000,120000)
plt.legend()
plt.show()

import pickle
pickle.dump(mod, open('/home/shrawant/Desktop/timeserieslpg/model.pkl','wb'), protocol=3)



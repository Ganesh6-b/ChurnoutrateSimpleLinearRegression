# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 11:30:02 2019

@author: Ganesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

churndata = pd.read_csv("F:\\R\\files\\emp_data.csv")

churndata.columns

churndata = churndata.rename(columns = {"Salary_hike" : "Shike", "Churn_out_rate" : "Churnrate"})
churndata.columns

plt.plot(churndata.Shike)
# this salary hike is normally distributed in a positive rate
plt.plot(churndata.Churnrate)
# churnout rate is negatively normal distributed
plt.plot(churndata.Shike, churndata.Churnrate)
# both are negatively normal distributed with high correlation between them
churndata.Shike.corr(churndata.Churnrate)
#negatively high correlated

#building a model

import statsmodels.formula.api as smf

model1 = smf.ols("Shike~Churnrate", data = churndata).fit()
model1.summary() # 0.83

pred = model1.predict(churndata)
pred.corr(churndata.Shike) #91 % is correctly predicted

#another model

model2 = smf.ols("Shike~np.log(Churnrate)", data = churndata).fit()
model2.summary() #0.87

pred2 = model2.predict(churndata)
pred2.corr(churndata.Shike) #93% is correctly predicted

#another model
Churnratesqr = churndata.Churnrate * churndata.Churnrate
model3 = smf.ols("Shike~Churnrate  + Churnratesqr", data = churndata).fit()
model3.summary() #0.97
pred3 = model3.predict(churndata)
pred3.corr(churndata.Shike) #93% is correctly predicted

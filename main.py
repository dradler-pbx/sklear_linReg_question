import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data = pd.read_csv('data.csv')

x_data = data[['u', 'v', 'w']]
Y_data = data['Y']

poly_feat = PolynomialFeatures(degree=2)
poly = poly_feat.fit_transform(x_data)

reg = LinearRegression().fit(poly, Y_data)

print(reg.score(poly, Y_data))

coeffs = reg.coef_
print(coeffs)

x_predict = poly_feat.fit_transform([[298.15, 268.15, 2600]])

print(reg.predict(x_predict))
print(np.sum([coeffs[i]*x_predict[0][i] for i in range(10)]))

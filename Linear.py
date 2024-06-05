import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

url = 'C:/Users/gary/python/Numerik-Regresi/student_performance.csv'
data = pd.read_csv(url)
print(data.head())


X = data['Hours Studied'].values.reshape(-1, 1)
y = data['Performance Index'].values

linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)

rms_error_linear = np.sqrt(mean_squared_error(y, y_pred_linear))

plt.scatter(X, y, color='blue', label='Data Asli')
plt.plot(X, y_pred_linear, color='red', label='Regresi Linear')
plt.xlabel('Hours Studied')
plt.ylabel('Performance Index')
plt.title('Regresi Linear')
plt.legend()
plt.show()

print('RMS Error untuk Model Linear:', rms_error_linear)

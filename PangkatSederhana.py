import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

url = 'C:/Users/gary/python/Numerik-Regresi/student_performance.csv'
data = pd.read_csv(url)
print(data.head())

X = data['Hours Studied'].values.reshape(-1, 1)
y = data['Performance Index'].values

def power_law(X, a, b):
    return a * np.power(X, b)

popt, pcov = curve_fit(power_law, X.flatten(), y)

y_pred_power = power_law(X.flatten(), *popt)

rms_error_power = np.sqrt(mean_squared_error(y, y_pred_power))

plt.scatter(X, y, color='blue', label='Data Asli')
plt.plot(X, y_pred_power, color='red', label='Regresi Pangkat Sederhana')
plt.xlabel('Hours Studied')
plt.ylabel('Performance Index')
plt.title('Regresi Pangkat Sederhana')
plt.legend()
plt.show()

print('RMS Error untuk Model Pangkat Sederhana:', rms_error_power)

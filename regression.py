import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data
x = np.array([[15], [23], [18], [23], [24], [22], [22], [19], [19], [16], [24], [11], [24], [16], [23]])
y = np.array([49, 63, 68, 60, 58, 61, 60, 63, 60, 52, 62, 30, 59, 49, 68])

# Linear regression model
model = LinearRegression()
model.fit(x, y)

# Predict for a new value
new_value = np.array([[20]])
predicted_value = model.predict(new_value)
print(model.coef_)
print(model.intercept_)
print("Predicted value for 20:", predicted_value)

# Plotting the data and regression line
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, model.predict(x), color='red', label='Regression Line')
plt.scatter(new_value, predicted_value, color='green', label='Predicted Value (20)')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.legend()
plt.grid(True)
plt.show()

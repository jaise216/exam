import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
dataset = pd.read_csv("Salary_Data.csv")
a = dataset.iloc[:, :-1]
b = dataset.iloc[:, 1]
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.3, random_state=0)
reg = LinearRegression()
reg.fit(a_train, b_train)
pred = reg.predict(a_test)
plt.scatter(a_train, b_train, c="yellow")
plt.plot(a_train, reg.predict(a_train), c="green")
plt.title("Salary v/s Years of experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
plt.scatter(a_test, b_test, c="blue")
plt.plot(a_train, reg.predict(a_train), c="red")
plt.title("Salary v/s Years of experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
mse = mean_squared_error(b_test, pred)
mae = mean_absolute_error(b_test, pred)
print(mse)
print(mae)

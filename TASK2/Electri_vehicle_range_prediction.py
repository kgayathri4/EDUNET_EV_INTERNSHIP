import pandas as pd
import numpy as np
df=pd.read_csv("electric_vehicles_spec_2025.csv.csv")
print(df)
df.info()
#removing all null values
df.dropna(inplace=True)
df.info()
print(df.columns)
data=df[['top_speed_kmh','battery_capacity_kWh','torque_nm','range_km']]
#heatmap to find correlation with columns
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(),annot=True)
plt.title("Feature correlation Heatmap",fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.subplots_adjust(bottom=0.2)
plt.show()
#divide the dataset into x and y(x is input and y is output)
x=df[['top_speed_kmh','battery_capacity_kWh','torque_nm']]
y=df['range_km']
print(x)
#split dataset into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("predicted values: ",y_pred)
comparision=pd.DataFrame({'Actual':y_test,'predicted':y_pred})
print(comparision.head(10))
#mse
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,y_pred))
#r2 score
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))
plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred,color='blue',label='Predicted vs Actual')
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='red',lw=2,label="Perfect Prediction Line")
plt.xlabel("Actual EV Range(km)")
plt.ylabel("Predicted EV Range(km)")
plt.title("Actual vs Predicted Electric Vehicle Range")
plt.legend()
plt.grid(True)
plt.show()
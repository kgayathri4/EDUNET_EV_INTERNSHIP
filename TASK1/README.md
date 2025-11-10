🔋 Electric Vehicle Price Prediction using Machine Learning & Generative AI

🧠 Overview

This project aims to predict the price of electric vehicles (EVs) based on their technical specifications such as battery capacity, efficiency, range, and performance metrics.

The model helps manufacturers, analysts, and buyers estimate EV prices accurately using machine learning techniques.

This project was developed as part of an internship focusing on the application of data science and generative AI concepts to real-world sustainability problems.


---

📦 Dataset

Source: Kaggle — Electric Vehicle Specifications and Prices
Author: Fatih Ilhan

The dataset contains details for over 360 EVs, including:

🔋 Battery capacity (kWh)

⚡ Efficiency (Wh/km)

⏱ Fast charge speed (km/h)

💶 Price (EUR)

🛣 Range (km)

🚗 Top speed (km/h)

⚙ Acceleration (0–100 km/h)


After cleaning missing and invalid entries, 309 records were used for model training.


---

🧹 Data Preprocessing

Steps performed in the notebook / script:

1. Load the dataset using pandas

df = pd.read_csv('EV_cars.csv')


2. Inspect and clean missing values

Dropped rows where price or essential features were missing.



3. Renamed columns for consistency:

'Price.DE.' → 'Price'

'acceleration..0.100.' → 'Acceleration_0_100'



4. Selected useful features

['Battery', 'Efficiency', 'Fast_charge', 'Range', 'Top_speed', 'Acceleration_0_100']


5. Split data into training (80%) and testing (20%) sets.




---

🤖 Machine Learning Model

Algorithm Used: Random Forest Regressor

Why Random Forest?
It handles nonlinear relationships, is robust to outliers, and works well with tabular data.


Training

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

Evaluation Metrics

Metric	Value	Meaning

Mean Absolute Error (MAE)	~ €8,389	On average, the model’s predictions differ by this much
R² Score	0.90	Model explains ~90% of price variance


✅ Interpretation: The model has strong predictive performance and generalizes well.


---

📈 Example Output

Example Input:
Battery = 46.3 kWh
Efficiency = 250 Wh/km
Fast Charge = 290 km/h
Range = 185 km
Top Speed = 130 km/h
Acceleration = 12.1 s (0–100 km/h)

Predicted Price (€): 54,733.82


---

🧰 Tech Stack

Programming Language: Python 3.10+

Libraries Used:

pandas, numpy

scikit-learn

joblib

streamlit (for web app)


Platform: VS Code / Kaggle Notebook / GitHub



---

🚀 How to Run Locally

1️⃣ Clone the Repository

git clone https://github.com/<your-username>/EV_Price_Prediction.git
cd EV_Price_Prediction

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run Model Script

python Electric_vehicle_price_prediction.py

4️⃣ Run Streamlit Web App

streamlit run app.py


---

🖥 Streamlit Web App

A user-friendly app to test predictions interactively:

File: app.py

Users can input:

Range (km)

Battery size (kWh)

Efficiency

Fast Charge speed

Acceleration time


and get: → Predicted EV Price (€) instantly.


---

🧪 Files in Repository

File	Description

EV_cars.csv	Dataset
Electric_vehicle_price_prediction.py	Main ML model training code
app.py	Streamlit web app
ev_price_pipeline.pkl	Trained ML model
requirements.txt	Required Python libraries
README.md	Documentation



---

📊 Results Summary

Metric	Value

Rows used	309
Features used	6
R² Score	0.90
MAE	€8,389
Algorithm	Random Forest Regressor



---

🧩 Future Improvements

1. Hyperparameter tuning for Random Forest


2. Add deep learning models for comparison (e.g., XGBoost, LightGBM)


3. Include currency normalization for global EV markets


4. Add visualizations (price vs battery, range, etc.)


5. Integrate Generative AI (e.g., generate new EV specs to predict possible future models)




---

🏁 Conclusion

This project demonstrates how machine learning can accurately estimate EV prices using performance and efficiency data.
It contributes to understanding trends in the electric vehicle market and showcases the power of data-driven sustainability research.
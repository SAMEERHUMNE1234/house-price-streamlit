import pandas as pd
import numpy as np
import joblib
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import streamlit as st
import joblib
import numpy as np

# Load data
df = pd.read_csv("Housing.csv")

# -----------------------------
# 2️⃣ Encode Categorical Columns
# -----------------------------
binary_cols = [
    'mainroad', 'guestroom', 'basement',
    'hotwaterheating', 'airconditioning',
    'prefarea'
]

for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# Furnishing status encoding
df['furnishingstatus'] = df['furnishingstatus'].map({
    'furnished': 2,
    'semi-furnished': 1,
    'unfurnished': 0
})

# -----------------------------
# 3️⃣ Feature & Target
# -----------------------------
X = df.drop('price', axis=1)
y = df['price']

# -----------------------------
# 4️⃣ Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5️⃣ Train Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# 6️⃣ Evaluation
# -----------------------------
print("R2 Score:", r2_score(y_test, model.predict(X_test)))

# -----------------------------
# 7️⃣ Save Model
# -----------------------------
joblib.dump(model, "house_model.pkl")
print("Model saved successfully")


print("Streamlit version:", st.__version__)
print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
print("Scikit-learn version:", sklearn.__version__)
print("Joblib version:", joblib.__version__)

#stramlit app

model = joblib.load("house_model.pkl")

st.title("🏠 House Price Prediction App")

area = st.number_input("Area (sq ft)", 500, 20000, step=50)
bedrooms = st.slider("Bedrooms", 1, 10)
bathrooms = st.slider("Bathrooms", 1, 10)
stories = st.slider("Stories", 1, 5)
parking = st.slider("Parking", 0, 5)

mainroad = st.radio("Main Road", ["yes", "no"])
guestroom = st.radio("Guest Room", ["yes", "no"])
basement = st.radio("Basement", ["yes", "no"])
hotwater = st.radio("Hot Water Heating", ["yes", "no"])
ac = st.radio("Air Conditioning", ["yes", "no"])
prefarea = st.radio("Preferred Area", ["yes", "no"])
furnishing = st.radio("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])

# Convert categorical
binary = lambda x: 1 if x == "yes" else 0

furnish_map = {
    "furnished": 2,
    "semi-furnished": 1,
    "unfurnished": 0
}

if st.button("Predict Price"):
    features = np.array([[
        area, bedrooms, bathrooms, stories,
        binary(mainroad), binary(guestroom),
        binary(basement), binary(hotwater),
        binary(ac), parking,
        binary(prefarea), furnish_map[furnishing]
    ]])

    price = model.predict(features)[0]
    st.success(f"💰 Estimated House Price: ₹ {price:,.0f}")

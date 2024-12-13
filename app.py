import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

scaler = joblib.load("Scaler.pkl")

st.title("Restaurant Rating Prediction App")


st.caption("This app helps you to predict a restaurants review class")

st.divider()

averagecost = st.number_input("Please enter the estimated average cost for two", min_value=50, max_value=999999, value=1000,step=100 )

tablebooking = st.selectbox("Restaurant has table booking ?", ["Yes", "No"])

onlinedelivery = st.selectbox("Restaurant has Onine Booking ?", ["Yes", "No"])

pricerange = st.selectbox("What is the price range (1 Cheapest, 4 Most Expensive)", [1,2,3,4])

predictbutton = st.button("Predict the review!")
st.divider()

model = joblib.load("mlmodel.pkl")

bookingstatus = 1 if tablebooking == "Yes" else 0

deliverystatus = 1 if onlinedelivery == "Yes" else 0

#Has table booking 0 is no 1 is yes
#Has online delievery 0 is no 1 is yes

values = [[averagecost, bookingstatus, deliverystatus, pricerange]]
my_x_values = np.array(values)

X = scaler.transform(my_x_values)

if predictbutton:
    st.balloons()

    prediction = model.predict(X)
    st.write(prediction)

# above 2 below 2.5  = poor
# above 2.5 below 3.5 = average
# above 3.5 below 4.0 = good
# above 4.0 below 4.5 = very good
# above 4.5 = excellent

    if prediction < 2.5:
        st.write("Poor")
    elif prediction < 3.5:
        st.write("Average")
    elif prediction < 4.0:
        st.write("Good")
    elif prediction < 4.5:
        st.write("Very good")
    else:
        st.write("Excellent")    
        
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        datal = pickle.load(file)
    return datal
 
def predict(inp):
    datal = load_model()
    model = datal["model"]
    scaler = datal["scaler"]
    headings = datal["headers"]
    
    fixinp = np.asarray(inp).reshape(1,-1)
    fixinp = pd.DataFrame(fixinp,columns=headings)
    scaledinp =scaler.transform(fixinp)
    prediction = model.predict(scaledinp)

    return prediction[0]

def load_page():
    st.write("### Predict Diabetes")
    preg = st.number_input("Number of Pregnancies", min_value=0, step=1 )
    gl = st.number_input("Glucose Level", min_value=0, step=1 )
    bp = st.number_input("Blood Pressure", min_value=0, step=1 )
    thi = st.number_input("Skin Thickness", min_value=0, step=1 )
    ins = st.number_input("Insulin Level", min_value=0, step=1 )
    bmi = st.number_input("BMI", min_value=0.0)
    dpf = st.number_input("Diabetes Pedigree function", min_value=0.000, step=0.001 )
    age = st.number_input("Age", min_value=0, step=1 )
    if st.button("Predict"):
        inp= (preg,gl,bp,thi,ins,bmi,dpf,age)
        if predict(inp) == 0:
            st.popover("Prediction says you donot have Diabetes")
            st.balloons()
            st.toast("### Prediction says you DONOT HAVE Diabetes")
        elif predict(inp) == 1:
            st.balloons()
            st.toast("### Prediction says you HAVE Diabetes")
    

load_page()
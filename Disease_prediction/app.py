import streamlit as st 
import numpy as np
import pandas as pd
import pickle 
import os
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Mulitple Disease Prediction",layout="wide", page_icon="👨‍🦰🤶")

working_dir = os.path.dirname(os.path.abspath(__file__))

with open(f'{working_dir}/saved_models/parkinsons_disease_model.pkl','rb') as file:
    parkinsons_disease_model = pickle.load(file)

with open(f'{working_dir}/saved_models/heart_disease_model.pkl','rb') as file:
    heart_disease_model = pickle.load(file)

with open(f'{working_dir}/saved_models/diabetes.pkl','rb') as file:
    diabetes_model = pickle.load(file)

# kidney_disease_model = pickle.load(open(f'{working_dir}/saved_models/kidney.pkl','rb'))

with st.sidebar:
    selected = option_menu("Mulitple Disease Prediction", 
                ['Parkinson Disease Prediction',
                 'Heart Disease Prediction',
                 'Diabetes Prediction'],
                 menu_icon='hospital-fill',
                 icons=['activity','heart', 'person'],
                 default_index=0)

if selected == 'Parkinson Disease Prediction':
    st.title("Parkinson Disease Prediction Using Machine Learning")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        Name = st.text_input("Name")
    with col2:
        MDVP_Fo_Hz  = st.text_input("MDVP Fo(Hz) ")
    with col3:
        MDVP_Fhi_Hz = st.text_input("MDVP Fhi(Hz)")
    with col4:  
        MDVP_Flo_Hz = st.text_input("MDVP Flo(Hz)")
    with col5:
        MDVP_Jitter = st.text_input("MDVP Jitter(%)")
    with col1:
        MDVP_Jitter_Abs = st.text_input("MDVP Jitter(Abs)")
    with col2:
        MDVP_RAP = st.text_input("MDVP RAP")
    with col3:
        MDVP_PPQ = st.text_input("MDVP PPQ")
    with col4:
        Jitter_DDP = st.text_input("Jitter DDP")
    with col5:
        MDVP_Shimmer = st.text_input("MDVP Shimmer")
    with col1:
        MDVP_Shimmer_dB = st.text_input("MDVP Shimmer(dB)")
    with col2:
        Shimmer_APQ3 = st.text_input("Shimmer_APQ3")
    with col3:
        Shimmer_APQ5 = st.text_input("Shimmer_APQ5")
    with col4:
        MDVP_APQ = st.text_input("MDVP APQ")
    with col5:
        Shimmer_DDA = st.text_input("Shimmer DDA")
    with col1:
        NHR = st.text_input("NHR")
    with col2:
        HNR = st.text_input("HNR")
    with col3:
        RPDE = st.text_input("RPDE")
    with col4:
        DFA = st.text_input("DFA")
    with col5:
        spread1 = st.text_input("spread1")
    with col1:
        spread2 = st.text_input("spread2")
    with col2:
        D2 = st.text_input("D2")
    with col3:
        PPE = st.text_input("PPE")

    Parkinsons_result = ""
    if st.button("Parkinsons Test Result"):
        user_input=[MDVP_Fo_Hz,MDVP_Fhi_Hz,MDVP_Flo_Hz,MDVP_Jitter,MDVP_Jitter_Abs,MDVP_RAP,MDVP_PPQ,
                    Jitter_DDP,MDVP_Shimmer,MDVP_Shimmer_dB,Shimmer_APQ3,Shimmer_APQ5,MDVP_APQ,
                    Shimmer_DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE] 

        prediction = parkinsons_disease_model.predict([user_input])
        if prediction[0]==1:
            Parkinsons_result_result = "The person has Parkinsons"
        else:
            Parkinsons_result = "The person has no Parkinsons"
    st.success(Parkinsons_result)

if selected == 'Heart Disease Prediction':
    st.title("Heart Disease Prediction Using Machine Learning")
    col1, col2, col3  = st.columns(3)

    with col1:  
        age = st.text_input("Age in Years")
    with col2:
        sex = st.text_input("Sex (1 for male)")
    with col3:
        cp = st.text_input("Chest Pain Types")
    with col1:
        trestbps = st.text_input("Resting Blood Pressure (in mm hg)")
    with col2:
        chol = st.text_input("Serum Cholestroal in mg/dl")
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (type 1 if true)')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina (type 1 if yes)')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy (0-3)')

    with col1:
        thal = st.text_input('thal: 1 = normal; 2 = fixed defect; 3 = reversable defect')
    heart_disease_result = ""
    if st.button("Heart Disease Test Result"):  
        user_input = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
        user_input = [float(x) for x in user_input]
        prediction = heart_disease_model.predict([user_input])
        if prediction[0]==1:
            heart_disease_result = "This person is having heart disease"
        else:
            heart_disease_result = "This person does not have any heart disease"
    st.success(heart_disease_result)

if selected == 'Diabetes Prediction':
    
    st.title("Diabetes Prediction using ML")

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose')

    with col3:
        Blood_Pressure = st.text_input('Blood Pressure')

    with col1:
        Skin_Thickness = st.text_input('Skin Thickness')

    with col2:
        Insulin = st.text_input('Insulin')

    with col3:
        BMI = st.text_input('BMI')

    with col1:
        DPF = st.text_input('Diabetes Pedigree Function')

    with col2:
        Age = st.text_input('Age')

    # code for Prediction
    diabetes_prediction = ''

    # creating a button for Prediction    
    if st.button("Diabetes Test Result"):

        user_input = [Pregnancies,Glucose,Blood_Pressure,Skin_Thickness,Insulin,BMI,DPF,Age]

        user_input = [float(x) for x in user_input]

        prediction = diabetes_model.predict([user_input])

        if prediction[0] == 1:
            diabetes_prediction = "The person has Diabetes"
        else:
            diabetes_prediction = "The person does not have Diabetes"
    st.success(diabetes_prediction)
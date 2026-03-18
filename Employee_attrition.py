import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------
# Load Models & Feature Columns
# -----------------------------

attrition_model = pickle.load(open("attrition_model.pkl", "rb"))
performance_model = pickle.load(open("performance_model.pkl", "rb"))
feature_columns_attrition = pickle.load(open("feature_columns_attrition.pkl", "rb"))
feature_columns_perf = pickle.load(open("feature_columns_perf.pkl", "rb"))


st.title("Employee Analytics Dashboard 🏢")

st.write("Predict Employee Attrition and Performance Rating 🕵️‍♀️")

# -----------------------------------
# USER INPUT SECTION
# -----------------------------------

st.header("Enter Employee Details👩‍💻")

age = st.number_input("Age", 18, 60, 30)
monthly_income = st.number_input("Monthly Income", 1000, 50000, 5000)
job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
years_at_company = st.number_input("Years At Company", 0, 40, 5)

overtime = st.selectbox("OverTime", ["Yes", "No"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])

# -----------------------------------
# CREATE INPUT DATAFRAME (46 columns)
# -----------------------------------

input_df_attr = pd.DataFrame(columns=feature_columns_attrition)
input_df_attr.loc[0] = 0

# Fill numeric features
input_df_attr["Age"] = age
input_df_attr["MonthlyIncome"] = monthly_income
input_df_attr["JobSatisfaction"] = job_satisfaction
input_df_attr["YearsAtCompany"] = years_at_company

# Fill OverTime
if "OverTime_Yes" in input_df_attr.columns:
    input_df_attr["OverTime_Yes"] = 1 if overtime == "Yes" else 0

# Fill Marital Status
if marital_status == "Single" and "MaritalStatus_Single" in input_df_attr.columns:
    input_df_attr["MaritalStatus_Single"] = 1
elif marital_status == "Married" and "MaritalStatus_Married" in input_df_attr.columns:
    input_df_attr["MaritalStatus_Married"] = 1
elif marital_status == "Divorced" and "MaritalStatus_Divorced" in input_df_attr.columns:
    input_df_attr["MaritalStatus_Divorced"] = 1

# Fill Department
if department == "Sales" and "Department_Sales" in input_df_attr.columns:
    input_df_attr["Department_Sales"] = 1
elif department == "Research & Development" and "Department_Research & Development" in input_df_attr.columns:
    input_df_attr["Department_Research & Development"] = 1
elif department == "Human Resources" and "Department_Human Resources" in input_df_attr.columns:
    input_df_attr["Department_Human Resources"] = 1


# -----------------------------------
# ATTRITION PREDICTION
# -----------------------------------

st.header("Attrition Prediction✍️")

if st.button("Predict Attrition"):

    prob_attrition = attrition_model.predict_proba(input_df_attr)[:,1][0]

    st.write(f"Probability of Leaving: {prob_attrition:.2f}")

    if prob_attrition > 0.5:
        st.error("High Risk of Attrition ⚠")
    else:
        st.success("Low Risk of Attrition ✅")


# -----------------------------------
# PERFORMANCE PREDICTION
# -----------------------------------
input_df_perf = pd.DataFrame(columns=feature_columns_perf)
input_df_perf.loc[0] = 0
st.header("Performance Prediction👩‍💼🏅")

if st.button("Predict Performance"):

    prob_perf = performance_model.predict_proba(input_df_perf)[:,1][0]

    st.write(f"Probability of High Performance (Rating 4): {prob_perf:.2f}")

    if prob_perf > 0.5:
        st.success("Likely High Performer ⭐")
    else:
        st.warning("Standard Performer")


# -----------------------------------
# SIMPLE VISUAL DASHBOARD
# -----------------------------------

st.header("Dataset Insights🚀")

import matplotlib.pyplot as plt



try:
    df = pd.read_csv("employee_data.csv")

    st.subheader("Attrition Distribution")
    fig,ax=plt.subplots()
    df['Attrition'].value_counts().plot(kind='bar',color=['blue','salmon'],ax=ax)
    ax.set_xlabel("Attrition")
    ax.set_ylabel("Count")
    ax.set_title("Attrition Distribution")

    st.pyplot(fig)

    st.subheader("Performance Rating Distribution")
    fig2, ax2 = plt.subplots()

    df["PerformanceRating"].value_counts().plot(kind="bar",color=["lightgreen", "orange"],ax=ax2)

    ax2.set_xlabel("Performance Rating")
    ax2.set_ylabel("Count")
    ax2.set_title("Performance Distribution")

    st.pyplot(fig2)

except:
    st.info("Dataset file not found for visualization.")

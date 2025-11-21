import streamlit as st

st.title("Input Page")

# user_input = st.text_input("Enter something:")
# if user_input:
#     st.write("You entered:", user_input)
sex_value = st.radio("Select your Sex:", ["Female", "Male"], horizontal=True,
                     help="The model was only trained with binary sex data, therefore we cannot offer non-binary sex options at this time.")
# age
age_value = st.number_input("Enter your age:", min_value=1, 
                            max_value=120, value=30)

# smoking
sex_value = st.radio("Were you ever a smoker?", ["No", "Yes"], 
                     horizontal=True)

# diabetes
diabetes_value = st.radio("Do you have diabetes?", ["No","Yes"], 
                          horizontal=True)

# hight
hight_value = st.number_input("Enter your height in cm:", 
                              min_value=50, max_value=250, value=170)

# weight
weight_value = st.number_input("Enter your weight in kg:", 
                               min_value=20, max_value=300, value=70)

# BMI
bmi_value = weight_value / ((hight_value / 100) ** 2)
st.write("Your BMI is:", str(round(bmi_value, 2)))
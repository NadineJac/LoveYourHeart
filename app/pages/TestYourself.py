import streamlit as st
# Page config
st.set_page_config(page_title="Test Yourself", page_icon="❤️", layout="wide")

st.title(" ❤️ Get your cardiovascular risk profile")

st.write("Complete your profile to get personalized heart health advice from our AI assistant.")
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:

    # Initialize session state if not exists
    if "profile_submitted" not in st.session_state:
        st.session_state["profile_submitted"] = False

    # Sex
    sex_value = st.radio(
        "Select your Sex:", 
        ["Female", "Male"], 
        horizontal=True,
        help="The model was only trained with binary sex data, therefore we cannot offer non-binary sex options at this time.",
        index=0 if st.session_state.get("sex_value") != "Yes" else 1
    )

    # Age
    age_value = st.number_input(
        "Enter your age:", 
        min_value=1, 
        max_value=120, 
        value=st.session_state.get("age_value", 30)
    )

    # Smoking
    smoker_value = st.radio(
        "Were you ever a smoker?", 
        ["No", "Yes"], 
        horizontal=True,
        index=0 if st.session_state.get("smoker_value") != "Yes" else 1
    )

    # Diabetes
    diabetes_value = st.radio(
        "Do you have diabetes?", 
        ["No", "Yes"], 
        horizontal=True,
        index=0 if st.session_state.get("diabetes_value") != "Yes" else 1
    )

    # Height
    height_value = st.number_input(
        "Enter your height in cm:", 
        min_value=50, 
        max_value=250, 
        value=st.session_state.get("height_value", 170)
    )

    # Weight
    weight_value = st.number_input(
        "Enter your weight in kg:", 
        min_value=20, 
        max_value=300, 
        value=st.session_state.get("weight_value", 70)
    )

    # Calculate BMI
    bmi_value = weight_value / ((height_value / 100) ** 2)
    st.info(f"📊 Your BMI is: **{round(bmi_value, 2)}**")

    # Add some spacing
    st.write("")

    # Submit button to save profile
    if st.button("💾 Save Profile", type="primary", use_container_width=True):
        # Save all values to session state
        st.session_state["sex_value"] = sex_value
        st.session_state["age_value"] = age_value
        st.session_state["smoker_value"] = smoker_value
        st.session_state["diabetes_value"] = diabetes_value
        st.session_state["height_value"] = height_value
        st.session_state["weight_value"] = weight_value
        st.session_state["bmi_value"] = bmi_value
        st.session_state["profile_submitted"] = True
        
        st.toast("Profile saved! Head to the AI Assistant page to get personalized advice.", icon="✅")
        #st.success("✅ Profile saved! Head to the AI Assistant page to get personalized advice.")

        with col_right:
            st.write('''
                     ❗No proper risk estimation implemented yet. 
                     Current estimation only based on sex, smoking, diabetes and BMI.
                     Proceed with caution and confer with the AI Assistant to 
                     get personalized adviced based on research literature.''')
            import pickle
            model = pickle.load(open('../models/trained_pipe_logReg.sav', 'rb'))

            import pandas as pd
            user = pd.DataFrame({
                'BMI':[bmi_value], 
                'Smoking':[smoker_value], 
                'Sex':[sex_value],
                'Diabetic': [diabetes_value]
            })

            prediction = model.predict(user)

            # st.markdown
            st.write("### Are you at risk of heart disease:", prediction[0])
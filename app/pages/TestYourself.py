import streamlit as st
from streamlit_scroll_to_top import scroll_to_here
import os

from streamlit_scroll_to_top import scroll_to_here

if 'scroll_to_top' not in st.session_state:
    st.session_state.scroll_to_top = False
    
if 'scroll_to_header' not in st.session_state:
    st.session_state.scroll_to_header = False

if st.session_state.scroll_to_top:
    scroll_to_here(0, key='top')  # Scroll to the top of the page
    st.session_state.scroll_to_top = False  # Reset the state after scrolling

def scroll():
    st.session_state.scroll_to_top = True
    
def scrollheader():
    st.session_state.scroll_to_header = True

# Page config
st.set_page_config(page_title="Test Yourself", page_icon="❤️", layout="wide")
st.title(" ❤️ Get your cardiovascular risk profile")
st.write("Complete your profile to get personalized heart health advice from our AI assistant.")
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:

    # Initialize session state if not exists
    if "profile_submitted" not in st.session_state:
        st.session_state["profile_submitted"] = False

    with st.expander("Demographics"):
        # Sex
        sex_value = st.segmented_control(
            "Select your Sex:", 
            ["Female", "Male"], 
            help="The model was only trained with binary sex data, therefore we cannot offer non-binary sex options at this time.",
            default=st.session_state.get("sex_value")
        )

        # Age
        age_value = st.number_input(
            "Enter your age:", 
            min_value=18, 
            max_value=120, 
            value=st.session_state.get("age_value")
        )
        if age_value is not None:
            if age_value >= 18 and age_value <= 24:
                age_cat = "18-24"
            elif age_value >= 25 and age_value <= 29:
                age_cat = "25-29"
            elif age_value >= 30 and age_value <= 34:
                age_cat = "30-34"
            elif age_value >= 35 and age_value <= 39:
                age_cat = "35-39"
            elif age_value >= 40 and age_value <= 44:
                age_cat = "40-44"
            elif age_value >= 45 and age_value <= 49:
                age_cat = "45-49"
            elif age_value >= 50 and age_value <= 54:
                age_cat = "50-54"
            elif age_value >= 55 and age_value <= 59:
                age_cat = "55-59"
            elif age_value >= 60 and age_value <= 64:
                age_cat = "60-64"
            elif age_value >= 65 and age_value <= 69:
                age_cat = "65-69"
            elif age_value >= 70 and age_value <= 74:
                age_cat = "70-74"
            elif age_value >= 75 and age_value <= 79:
                age_cat = "75-79"
            elif age_value >= 80:
                age_cat = "80 or older"
        # Race
        race_options = ["White", "Hispanic", "Black", "Other", "Asian", "American Indian/Alaskan Native"]

        race_cat = st.selectbox(
            "Which of these groups best represents your race?",
            race_options,
            index=race_options.index(st.session_state.race_cat) if "race_cat" in st.session_state and st.session_state.race_cat in race_options else None
        )
        
        # Height
        height_value = st.number_input(
            "Enter your height in cm:", 
            min_value=50, 
            max_value=250, 
            value=st.session_state.get("height_value")
        )

        # Weight
        weight_value = st.number_input(
            "Enter your weight in kg:", 
            min_value=20, 
            max_value=300, 
            value=st.session_state.get("weight_value")
        )

        # Calculate BMI
        if height_value != None and weight_value != None:
            bmi_value = weight_value / ((height_value / 100) ** 2)
            st.info(f"📊 Your BMI is: **{round(bmi_value, 2)}**")

    with st.expander("General Health and Lifestyle"): 
        health_cat = st.segmented_control(
            "How would you rate your general health?", 
            ["Excellent", "Very good", "Good", "Fair", "Poor"], 
            default=st.session_state.get("health_cat")
        )

        # Sleep	
        sleep_value = st.select_slider(
            "How many hours of sleep do you get?", 
            range(0, 25), 
            help="On average during 24 hours.", 
            value=st.session_state.get("sleep_value")
        )
    
        st.write("**On how many days in the last 30 days:**")
        # Excercise
        excercise_value = st.select_slider(
            "Did you excercise?", 
            range(0, 31), 
            value=st.session_state.get("excercise_value")
        )
        # Mental health
        mentalhealth_value = st.select_slider(
            "Was your mental health poor?", 
            range(0, 31), 
            value=st.session_state.get("mentalhealth_value"),
            help="This includes stress, depression, and problems with emotions."
        )
        st.write("")
        # Difficulty walking
        walk_value = st.segmented_control(
            "Do you have trouble walking or climbing stairs?", 
            ["No", "Yes"], 
            default=st.session_state.get("walk_value")
        )
        # Smoking
        smoker_value = st.segmented_control(
            "Were you ever a smoker?", 
            ["No", "Yes"], 
            default=st.session_state.get("smoker_value")
        )
        # Alcohol drinking
        alc_value = st.number_input(
            "How many alcoholic drinks do you drink per week?", 
            min_value=0, 
            max_value=99, 
            value=st.session_state.get("alc_value")
        )
        if alc_value is not None:
            if (alc_value > 14 and sex_value == "Male") or (alc_value > 7 and sex_value == "Female"):
                st.info("⚠️ High alcohol consumption for your sex.")
                alc_cat = "Yes"
            else:
                alc_cat = "No"

    with st.expander("Disease History"):
        # Diabetes
        diabetes_value = st.segmented_control(
            "Do you have diabetes?", 
            ["No", "No, borderline diabetes", "Yes (during pregnancy)", "Yes"], 
            #vertical=True,
            default=st.session_state.get("diabetes_value")
        )
        # astma
        astma_value = st.segmented_control(
            "Did you ever have astma?", 
            ["No", "Yes"], 
            default=st.session_state.get("astma_value")
        )
        # Kidney disease
        kidney_value = st.segmented_control(
            "Did you ever have a kidney disease?", 
            ["No", "Yes"], 
            help="Apart from other than stones, bladder infection, or incontinence.",
            default=st.session_state.get("kidney_value")
        )
        # Skin Cancer
        skin_value = st.segmented_control(
            "Did you ever have skin cancer?", 
            ["No", "Yes"], 
            default=st.session_state.get("skin_value")
        )
        # stroke
        stroke_value = st.segmented_control(
            "Did you ever have a stroke?", 
            ["No", "Yes"], 
            default=st.session_state.get("stroke_value")
        )



    # Add some spacing
    st.write("")

    # Submit button to save profile
    if st.button("💾 Save Profile", type="primary", use_container_width=True):
        # Save all values to session state FIRST
        st.session_state["sex_value"] = sex_value
        st.session_state["age_value"] = age_value
        st.session_state["race_cat"] = race_cat
        st.session_state["age_cat"] = age_cat
        st.session_state["health_cat"] = health_cat
        st.session_state["sleep_value"] = sleep_value
        st.session_state["smoker_value"] = smoker_value
        st.session_state["diabetes_value"] = diabetes_value
        st.session_state["height_value"] = height_value
        st.session_state["weight_value"] = weight_value
        st.session_state["bmi_value"] = bmi_value
        st.session_state["alc_value"] = alc_value
        st.session_state["alc_cat"] = alc_cat 
        st.session_state["stroke_value"] = stroke_value
        st.session_state["astma_value"] = astma_value
        st.session_state["kidney_value"] = kidney_value
        st.session_state["skin_value"] = skin_value
        st.session_state["excercise_value"] = excercise_value
        st.session_state["mentalhealth_value"] = mentalhealth_value
        st.session_state["walk_value"] = walk_value        

        
        # Load model and make prediction
        import pickle
        import pandas as pd
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "trained_pipe_logReg.sav")
        model = pickle.load(open(MODEL_PATH, 'rb'))

        user = pd.DataFrame({             
            'Sex': [sex_value],
            'Age': [age_value],
            'Race': [race_cat],
            'Health': [health_cat],
            'SleepDuration': [sleep_value],
            'AgeCategory': [age_cat],
            'Smoking': [smoker_value], 
            'Alcohol consumption': [alc_cat],
            'BMI': [bmi_value],
            'Diabetic': [diabetes_value],                
            'Stroke': [stroke_value],
            'Astma': [astma_value],
            'KidneyDisease': [kidney_value],
            'SkinCancer': [skin_value],
            'Exercise':[excercise_value],
            'MentalHealth': [mentalhealth_value],
            'WalkingDifficulty': [walk_value]
        })

        prediction = model.predict(user)
        st.session_state["risk_value"] = prediction[0]
        st.session_state["profile_submitted"] = True
        st.session_state["just_saved"] = True  # Flag to show we just saved
        
        st.session_state.scroll_to_top = True
        st.rerun()
        
        # Display results in col_right
    if st.session_state["profile_submitted"]:
        with col_right:
            st.write('''
                    ❗No proper risk estimation implemented yet. 
                    Current estimation only based on sex, smoking, diabetes and BMI.
                    Proceed with caution and confer with the AI Assistant to 
                    get personalized adviced based on research literature.''')
            st.write("### Are you at risk of heart disease:", st.session_state["risk_value"])
            st.success("✅ Profile saved! Head to the AI Assistant page to get personalized advice.")
import streamlit as st

st.title("Test Yourself")

st.write("Complete your profile to get personalized heart health advice from our AI assistant.")

# Initialize session state if not exists
if "profile_submitted" not in st.session_state:
    st.session_state["profile_submitted"] = False

# Sex
sex_value = st.radio(
    "Select your Sex:", 
    ["Female", "Male"], 
    horizontal=True,
    help="The model was only trained with binary sex data, therefore we cannot offer non-binary sex options at this time.",
    index=0 if not st.session_state.get("sex_value") else (0 if st.session_state.get("sex_value") == "Female" else 1)
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
    index=0 if not st.session_state.get("smoker_value") else (0 if st.session_state.get("smoker_value") == "No" else 1)
)

# Diabetes
diabetes_value = st.radio(
    "Do you have diabetes?", 
    ["No", "Yes"], 
    horizontal=True,
    index=0 if not st.session_state.get("diabetes_value") else (0 if st.session_state.get("diabetes_value") == "No" else 1)
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
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
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
        
        st.success("✅ Profile saved! Head to the AI Assistant page to get personalized advice.")
        st.balloons()

# Show current profile if it exists
if st.session_state.get("profile_submitted"):
    st.write("---")
    st.subheader("📋 Your Current Profile")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Sex:** {st.session_state.get('sex_value', 'Not set')}")
        st.write(f"**Age:** {st.session_state.get('age_value', 'Not set')}")
        st.write(f"**Smoker:** {st.session_state.get('smoker_value', 'Not set')}")
    
    with col2:
        st.write(f"**Diabetes:** {st.session_state.get('diabetes_value', 'Not set')}")
        st.write(f"**Height:** {st.session_state.get('height_value', 'Not set')} cm")
        st.write(f"**Weight:** {st.session_state.get('weight_value', 'Not set')} kg")
    
    st.write(f"**BMI:** {round(st.session_state.get('bmi_value', 0), 2)}")
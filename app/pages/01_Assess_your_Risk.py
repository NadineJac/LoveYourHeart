### Load libraries
import streamlit as st
from streamlit_scroll_to_top import scroll_to_here
import os
import plotly.graph_objects as go
import pickle
import pandas as pd
import shap
from streamlit_scroll_to_top import scroll_to_here
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from io import BytesIO
from utils import compute_shap_plot_voting
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import shap
import hashlib
# Call the LLM + RAG system
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole

#----------------------------------------------------------------------------
### Define functions
#caching
@st.cache_data(show_spinner=False)
def compute_shap_plot_voting_cached(_model, _user_df, _user2_df, _background_sample):
    """Cached version of SHAP plot computation"""
    # Convert dataframes to hashable format for caching
    return compute_shap_plot_voting(_model, _user_df, user2=_user2_df, background_data=_background_sample)
@st.cache_data(show_spinner=False)
def bootstrap_confidence_interval_cached(_model, _user_df):
    """Cached version of bootstrap confidence interval computation"""
    return bootstrap_confidence_interval_single_row(_model, _user_df)

## Scrolling
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

## Assign age categories
def assign_age_cat(age_value):
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
    return age_cat

## Bootstrap confidence intervals
def bootstrap_confidence_interval_single_row(model, user_df, n_bootstrap=300):
    """
    Generate confidence intervals for a single user's prediction 
    by adding small noise to numeric features.
    """
    numeric_cols = user_df.select_dtypes(include=['float64', 'int64']).columns

    preds = []

    for _ in range(n_bootstrap):
        noisy_sample = user_df.copy()

        # add small gaussian noise (1–3%)
        for col in numeric_cols:
            val = user_df[col].iloc[0]
            noise = np.random.normal(0, val * 0.50)  # 50% noise                
            noisy_sample[col] = max(val + noise, 0)

        pred = model.predict_proba(noisy_sample)[0, 1] * 100
        preds.append(pred)

    lower = np.percentile(preds, 2.5)
    upper = np.percentile(preds, 97.5)

    return lower, upper

# Create PDF Report
def create_pdf_report(user_df, risk_value, ci_low, ci_high, what_if_df=None, what_if_risk=None):
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=60, bottomMargin=40)
    
    # Styles
    styles = getSampleStyleSheet()
    heading_style = ParagraphStyle(
        'Heading',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor("#C21807"),      # Light pink text
        #backColor=colors.HexColor("#C21807"),      # Dark red background
        alignment=TA_CENTER,                        # Center align
        spaceAfter=20,
        leading=24,                                 # line height
        fontName='Helvetica-Bold'
        )
    sub_heading_style = ParagraphStyle(
        'SubHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor("#C21807"),  # Pinkish for subheadings
        spaceAfter=8
    )
    normal_style = styles['Normal']
    normal_style.fontSize = 11
    
    story = []

    # Title
    story.append(Paragraph("Love Your Heart - Heart Risk Report", heading_style))
    story.append(Spacer(1, 12))

    # Original user data
    story.append(Paragraph("Original Data", sub_heading_style))
    
    # Build a table for better presentation
    original_data = [["Parameter", "Value"]]
    for col, val in user_df.loc[0].items():
        original_data.append([col, str(val)])
    original_data.append(["Predicted Risk (%)", f"{risk_value:.2f}%"])
    original_data.append(["Confidence Interval (%)", f"{ci_low:.2f}% - {ci_high:.2f}%"])

    table = Table(original_data, hAlign='LEFT', colWidths=[180, 180])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#C21807")),  # light red header
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),

        # Highlight last row, second column (Predicted Risk)
        ('BACKGROUND', (1,-2), (1,-2), colors.HexColor("#FFCDD2")),   # light pink background
        ('TEXTCOLOR', (1,-2), (1,-2), colors.HexColor("#C21807")),     # dark red text
        ('FONTNAME', (1,-2), (1,-2), 'Helvetica-Bold')
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    # What-If scenario (only if available)
    if what_if_df is not None:
        story.append(Paragraph("What-If Scenario", sub_heading_style))
        
        what_if_data = [["Parameter", "Value"]]
        for col, val in what_if_df.loc[0].items():
            what_if_data.append([col, str(val)])
        if what_if_risk is not None:
            what_if_data.append(["Predicted Risk (%)", f"{what_if_risk:.2f}%"])
            what_if_data.append(["Confidence Interval (%)", f"{lower_ci2:.1f}% – {upper_ci2:.1f}%"])
        
        table2 = Table(what_if_data, hAlign='LEFT', colWidths=[180, 180])
        table2.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#C21807")),  # pink header
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),

            # Highlight last row, second column (Predicted Risk)
            ('BACKGROUND', (1,-2), (1,-2), colors.HexColor("#FFCDD2")),   # light pink background
            ('TEXTCOLOR', (1,-2), (1,-2), colors.HexColor("#C21807")),     # dark red text
            ('FONTNAME', (1,-2), (1,-2), 'Helvetica-Bold')
        ]))
        story.append(table2)
        story.append(Spacer(1, 20))

    # AI Initial Recommendation Section
    story.append(Paragraph("Initial Recommendation", sub_heading_style))
    story.append(Spacer(1, 10))

    ai_reco = initial_recommendation or "No recommendation available."
    ai_reco = ai_reco.replace("\n", "<br/>")
    story.append(Paragraph(ai_reco, normal_style))
    story.append(Spacer(1, 20))

    doc.build(story)
    buffer.seek(0)
    return buffer

# Changed fetures for 'What-If"-Scenario
def display_changes_compact(user, user2):
    """
    Return only the changes in a compact markdown-ready format
    """
    differences = []
    for col in user.columns:
        val1 = user[col].iloc[0]
        val2 = user2[col].iloc[0]
        if val1 != val2:
            differences.append(f"* **{col}**: {val1} → {val2}")
    return differences

# Get feature Importances
def compute_shap_plot(model, user, user2=None):
    """
    Compute SHAP plot with optional What-If comparison.
    
    Parameters:
    - model: Trained pipeline model
    - user: Original user DataFrame
    - user2: Modified user DataFrame (optional, for What-If scenario)
    """
    X_user_preprocessed = model.named_steps["prep"].transform(user)
    explainer = shap.TreeExplainer(model.named_steps["model"])
    shap_values = explainer(X_user_preprocessed)

    # Get encoded feature names
    encoded_feature_names = model.named_steps["prep"].get_feature_names_out()

    # Aggregate SHAP values by original feature (keeping the sign!)
    original_importances = {}

    for i, encoded_name in enumerate(encoded_feature_names):
        # Extract original feature name
        if '__' in encoded_name:
            original_feature = encoded_name.split('__')[1].rsplit('_', 1)[0]
        else:
            original_feature = encoded_name.split('_')[0]
        
        # Sum SHAP values (keep the sign to show direction)
        shap_value = shap_values.values[0, i]
        
        if original_feature in original_importances:
            original_importances[original_feature] += shap_value
        else:
            original_importances[original_feature] = shap_value

    # Convert to DataFrame
    importance_df = pd.DataFrame({
        'feature': list(original_importances.keys()),
        'shap_value': list(original_importances.values())
    })

    # If user2 is provided, compute SHAP values for modified data
    whatif_importances = None
    changed_features = []
    if user2 is not None:
        # Find which features changed
        for col in user.columns:
            if user[col].iloc[0] != user2[col].iloc[0]:
                changed_features.append(col)
        
        # Compute SHAP values for user2
        X_user2_preprocessed = model.named_steps["prep"].transform(user2)
        shap_values2 = explainer(X_user2_preprocessed)
        
        # Aggregate SHAP values for user2
        whatif_importances = {}
        for i, encoded_name in enumerate(encoded_feature_names):
            if '__' in encoded_name:
                original_feature = encoded_name.split('__')[1].rsplit('_', 1)[0]
            else:
                original_feature = encoded_name.split('_')[0]
            
            shap_value = shap_values2.values[0, i]
            
            if original_feature in whatif_importances:
                whatif_importances[original_feature] += shap_value
            else:
                whatif_importances[original_feature] = shap_value

    # Define modifiability levels
    HIGHLY_MODIFIABLE = {
        "Smoking",
        "AlcoholDrinking",
        "PhysicalActivity",
        "BMI",
        "SleepTime",
    }

    MODERATELY_MODIFIABLE = {
        "PhysicalHealth",
        "MentalHealth",
        "GenHealth",
        "DiffWalking",
    }

    # Classify features
    def get_modifiability(feature):
        if feature in HIGHLY_MODIFIABLE:
            return 'highly'
        elif feature in MODERATELY_MODIFIABLE:
            return 'moderately'
        else:
            return 'non'

    importance_df['modifiability'] = importance_df['feature'].apply(get_modifiability)

    # Sort by absolute value
    importance_df['abs_shap'] = importance_df['shap_value'].abs()
    importance_df = importance_df.sort_values('abs_shap', ascending=True)

    # Assign colors based on direction and modifiability
    def get_color(row):
        if row['shap_value'] < 0:  # Decreases risk (green)
            if row['modifiability'] == 'highly':
                return "#0c632c"  # Dark green
            elif row['modifiability'] == 'moderately':
                return "#569C70"  # Medium green
            else:
                return "#94d1ab"  # Light green
        else:  # Increases risk (rose)
            if row['modifiability'] == 'highly':
                return '#f43f5e'  # Dark rose
            elif row['modifiability'] == 'moderately':
                return "#f78495"  # Medium rose
            else:
                return "#fcbbc3"  # Light rose

    importance_df['color'] = importance_df.apply(get_color, axis=1)

    # Create the plot
    fig, ax = plt.subplots(figsize=(7, 4))

    bars = ax.barh(importance_df['feature'], importance_df['shap_value'], 
                color=importance_df['color'], alpha=0.9)

    # Add value labels on bars with modifiability markers
    for i, (bar, value, modifiability) in enumerate(zip(bars, importance_df['shap_value'], 
                                                        importance_df['modifiability'])):
        label_x = value + (0.002 if value > 0 else -0.002)
        alignment = 'left' if value > 0 else 'right'
                                            
        ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                f'{value:.2f}', 
                va='center', ha=alignment, fontsize=8)

    # Plot What-If scenario as blue dots if available
    if whatif_importances is not None and changed_features:
        # Add What-If values as blue dots for changed features
        for idx, row in importance_df.iterrows():
            feature = row['feature']
            if feature in changed_features and feature in whatif_importances:
                whatif_value = whatif_importances[feature]
                # Use the enumerated position in the sorted dataframe
                y_position = list(importance_df['feature']).index(feature)
                
                # Plot blue dot
                ax.plot(whatif_value, y_position, 'o', color='#0055A4', 
                       markersize=7, zorder=5)
                
                # Add label for What-If value
                label_x = whatif_value#whatif_value + (0.02 if whatif_value > 0 else -0.02)
                alignment = 'left' if whatif_value > 0 else 'right'
                ax.text(label_x, y_position, f'  {whatif_value:.2f}  ', 
                       va='center', ha=alignment, fontsize=8, 
                       color='#0055A4')

    # Add vertical line at zero
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

    # Labels and title
    ax.set_xlabel('SHAP Value (Impact on Risk)', fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)

    # Add legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [
        Patch(facecolor='#f43f5e', label='Risk ↑ (Highly Modifiable)', alpha=0.9),
        Patch(facecolor="#f78495", label='Risk ↑ (Moderately Modifiable)', alpha=0.9),
        Patch(facecolor="#fcbbc3", label='Risk ↑ (Non-modifiable)', alpha=0.9),
        Patch(facecolor="#0c632c", label='Risk ↓ (Highly Modifiable)', alpha=0.9),
        Patch(facecolor="#569C70", label='Risk ↓ (Moderately Modifiable)', alpha=0.9),
        Patch(facecolor="#94d1ab", label='Risk ↓ (Non-modifiable)', alpha=0.9),
    ]
    
    # Add What-If legend item if applicable
    if whatif_importances is not None and changed_features:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#0055A4', 
                   markersize=8, label='What-If Scenario', markeredgecolor='white')
        )
    
    ax.legend(handles=legend_elements, loc='best', fontsize=8, framealpha=0)

    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    return fig, importance_df

#--------------------------------------------------------------------------
### App
#--------------------------------------------------------------------------

# Page config
st.set_page_config(page_title="Test Yourself", page_icon="❤️", layout="wide")
st.title(" ❤️ Get your cardiovascular risk profile")

 # Initialize session state if not exists
if "profile_submitted" not in st.session_state:
    st.session_state["profile_submitted"] = False
    st.session_state["profile_submitted2"] = False
    st.session_state["risk_value2"] = False
    st.session_state["plot_generated"] = False

#####
col_left, col_right = st.columns([1, 2], gap="large")
with col_left:
    with st.expander("How it works"):
        st.markdown("""
    1. **Enter Your Health Information:** Provide details about your lifestyle and health factors such as age, BMI, smoking status, physical activity, and medical history. These are the inputs our machine learning model uses to estimate your heart disease risk.
    2. **Get Your Personalized Risk Score:** Once you submit your information, the app instantly calculates your risk level and displays it as an easy-to-read gauge (Low, Moderate, or High risk).
    3. **Understand What Matters Most:** Our model highlights which factors have the greatest impact on your risk score—these are called modifiable factors: lifestyle behaviors and health conditions you can actively change, such as smoking, physical activity, diet, and weight management.
    4. **Explore "What-If" Scenarios:** Curious how quitting smoking or increasing exercise could affect your risk? Use the interactive tool to adjust specific factors and see how your risk score changes in real-time.
    5. **Download Your Results:** Save a copy of your risk assessment and personalized recommendations for your records or to share with your healthcare provider.
                """)
        st.write("")

    tab1, tab2,  = st.tabs(["Your Data", "What if?"])
    css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1.1rem;    
    }
</style>
'''

    st.markdown(css, unsafe_allow_html=True)
#---------------------------------------------------------------------------
### User Data
    with tab1:      
        with st.form("user_input", enter_to_submit=False):
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
                    value=st.session_state.get("age_value"),
                    key = "input_age"
                )

                    
                # Race
                race_options = ["Asian", "American Indian/Alaskan Native", "Black", "Hispanic", "White", "Other"]

                race_cat = st.selectbox(
                    "Which of these groups best represents your race?",
                    race_options,
                    index=race_options.index(st.session_state.race_cat) if "race_cat" in st.session_state and st.session_state.race_cat in race_options else None
                )   

            with st.expander("General Health and Lifestyle"): 
                health_cat = st.segmented_control(
                    "How would you rate your general health?", 
                    ["Excellent", "Very good", "Good", "Fair", "Poor"], 
                    default=st.session_state.get("health_cat")
                )
                # Height
                height_value = st.number_input(
                    "Enter your height in cm:", 
                    min_value=50, 
                    max_value=250, 
                    value=st.session_state.get("height_value"),
                    key = "input_height"
                )

                # Weight
                weight_value = st.number_input(
                    "Enter your weight in kg:", 
                    min_value=20, 
                    max_value=300, 
                    value=st.session_state.get("weight_value"),
                    key="input_weight"
                )

                # Sleep	
                sleep_value = st.select_slider(
                    "How many hours of sleep do you get?", 
                    range(0, 25), 
                    help="On average during 24 hours.", 
                    value=st.session_state.get("sleep_value")
                )
            
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
                    value=int(st.session_state.get("alc_value",0)),
                    key = "input_alc"
                )
                
                # Excercise
                excercise_value = st.segmented_control(
                        "Did you exercise in the last 30 days?", 
                        ["No", "Yes"], 
                        default=st.session_state.get("excercise_value")
                    )

                st.write("**On how many days in the last 30 days:**")
                
                # Physical health
                physhealth_value = st.select_slider(
                    "Was your physical health not good?", 
                    range(0, 31), 
                    value=st.session_state.get("physhealth_value"),
                    help="This includes physical illness and injury."
                )

                # Mental health
                mentalhealth_value = st.select_slider(
                    "Was your mental health poor?", 
                    range(0, 31), 
                    value=st.session_state.get("mentalhealth_value"),
                    help="This includes stress, depression, and problems with emotions."
                )

            with st.expander("Disease History"):
                # Diabetes
                diabetes_value = st.segmented_control(
                    "Do you have diabetes?", 
                    ["No", "No, borderline diabetes", "Yes (during pregnancy)", "Yes"], 
                    #vertical=True,
                    default=st.session_state.get("diabetes_value")
                )
                # asthma
                asthma_value = st.segmented_control(
                    "Did you ever have asthma?", 
                    ["No", "Yes"], 
                    default=st.session_state.get("asthma_value")
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

                required_fields = {
                "sex": sex_value,
                "age": age_value,
                "race": race_cat,
                "general_health": health_cat,
                "sleep": sleep_value,
                "smoking": smoker_value,
                "diabetes": diabetes_value,
                "height": height_value,
                "weight": weight_value,
                "alcohol": alc_value,
                "stroke": stroke_value,
                "asthma": asthma_value,
                "kidney": kidney_value,
                "skin_cancer": skin_value,
                "exercise": excercise_value,
                "mental_health": mentalhealth_value,
                "difficulty_walking": walk_value,
                "physhealth_value": physhealth_value
            }

            st.write(":gray[Please fill out all fields to enable saving your profile.]", )
            all_filled = all(v not in (None, "", []) for v in required_fields.values())
            # Submit button to save profile
            if st.form_submit_button("💾 Save Profile & Get Risk Estimate",
                type="primary",
                use_container_width=True,
                #disabled=not all_filled # did not get to work with form
                ): 

                # calculate features
                # age category
                age_cat = assign_age_cat(age_value)
                
                # Calculate BMI
                bmi_value = weight_value / ((height_value / 100) ** 2)
                
                
                # alcoholDrinking
                if (alc_value > 14 and sex_value == "Male") or (alc_value > 7 and sex_value == "Female"):
                    alc_cat = "Yes"
                else:
                    alc_cat = "No"
                
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
                st.session_state["asthma_value"] = asthma_value
                st.session_state["kidney_value"] = kidney_value
                st.session_state["skin_value"] = skin_value
                st.session_state["excercise_value"] = excercise_value
                st.session_state["mentalhealth_value"] = mentalhealth_value
                st.session_state["walk_value"] = walk_value   
                st.session_state["physhealth_value"] = physhealth_value     

                
                with st.spinner("Estimating your heart attack risk..."):
                    # Load model and make prediction
                    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                    MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "trained_pipe_soft_voting.sav") # did not work with "trained_pipe_voting.sav"
                    model = pickle.load(open(MODEL_PATH, 'rb'))
                    BACKGROUND_PATH = os.path.join(PROJECT_ROOT, "models", "background_sample.sav")
                    background_sample = pickle.load(open(BACKGROUND_PATH, 'rb'))
                    voting_threshold = pickle.load(open(os.path.join(
                        PROJECT_ROOT, "models", "trained_pipe_voting_threshold.sav"), 'rb'))

                    user = pd.DataFrame({             
                        'Sex': [sex_value],
                        'Age': [age_value],
                        'Race': [race_cat],
                        'SleepTime': [sleep_value],
                        'AgeCategory': [age_cat],
                        'Smoking': [smoker_value], 
                        'AlcoholDrinking': [alc_cat],
                        'BMI': [round(bmi_value,2)],
                        'Diabetic': [diabetes_value],                
                        'Stroke': [stroke_value],
                        'Asthma': [asthma_value],
                        'KidneyDisease': [kidney_value],
                        'SkinCancer': [skin_value],
                        'PhysicalActivity':[excercise_value],
                        'MentalHealth': [mentalhealth_value],
                        'DiffWalking': [walk_value],
                        'GenHealth': [health_cat],
                        'PhysicalHealth': [physhealth_value] 
                    })

                    prediction = model.predict_proba(user)[:, 1] #>= 0.5
                    st.session_state["risk_value"] = round(prediction[0]*100,2)  # Store as percentage
                    st.session_state["profile_submitted"] = True
                    st.session_state["just_saved"] = True  # Flag to show we just saved
                    st.session_state["model"] = model  # Store model for later use
                    st.session_state["background_sample"]=background_sample
                    st.session_state["user_data"] = user  # Store user data for later use
                
                st.session_state.scroll_to_top = True
                st.rerun()
#-----------------------------------------------------------------------------
### What if? tab
    with tab2:
        if st.session_state["profile_submitted"] is False:
            st.write("Please add your data before running a 'What-If' simulation.")
        else:
            with st.form("user_input2", enter_to_submit=False):
        
                with st.expander("Demographics"):
                # Age
                    age_value = st.number_input(
                        "Enter your age:", 
                        min_value=18, 
                        max_value=120, 
                        value=st.session_state.get("age_value"),
                        key = "age_input_whatif"
                    )
                         
                with st.expander("General Health and Lifestyle"): 
                    health_cat = st.segmented_control(
                        "How would you rate your general health?", 
                        ["Excellent", "Very good", "Good", "Fair", "Poor"], 
                        default=st.session_state.get("health_cat"),
                        key = "health_input_whatif"
                    )

                    # Weight
                    weight_value = st.number_input(
                        "Enter your weight in kg:", 
                        min_value=20, 
                        max_value=300, 
                        value=st.session_state.get("weight_value"),
                        key = "weight_input_whatif"
                    )

                    # Sleep	
                    sleep_value = st.select_slider(
                        "How many hours of sleep do you get?", 
                        range(0, 25), 
                        help="On average during 24 hours.", 
                        value=st.session_state.get("sleep_value"),
                        key = "sleep_input_whatif"
                    )
                
                    # Difficulty walking
                    walk_value = st.segmented_control(
                        "Do you have trouble walking or climbing stairs?", 
                        ["No", "Yes"], 
                        default=st.session_state.get("walk_value"),
                        key = "walk_input_whatif"
                    )

                    # Smoking
                    smoker_value = st.segmented_control(
                        "Were you ever a smoker?", 
                        ["No", "Yes"], 
                        default=st.session_state.get("smoker_value"),
                        key = "smoker_input_whatif"
                    )

                    # Alcohol drinking
                    alc_value = st.number_input(
                        "How many alcoholic drinks do you drink per week?", 
                        min_value=0, 
                        max_value=99, 
                        value=int(st.session_state.get("alc_value",0)),
                        key = "alc_input_whatif"
                    )

                    # Excercise
                    excercise_value = st.segmented_control(
                        "Did you exercise in the last 30 days?", 
                        ["No", "Yes"], 
                        default=st.session_state.get("excercise_value"),
                        key = "exercise_input_whatif"
                    )

                    st.write("**On how many days in the last 30 days:**")
                     # Physical health
                    physhealth_value = st.select_slider(
                        "Was your physical health not good?", 
                        range(0, 31), 
                        value=st.session_state.get("physhealth_value"),
                        help="This includes physical illness and injury.",
                        key = "physhealth_input_whatif"
                    )

                    # Mental health
                    mentalhealth_value = st.select_slider(
                        "Was your mental health poor?", 
                        range(0, 31), 
                        value=st.session_state.get("mentalhealth_value"),
                        help="This includes stress, depression, and problems with emotions.",
                        key = "mentalhealth_input_whatif"
                    )

                with st.expander("Disease History"):
                    # Diabetes
                    diabetes_value = st.segmented_control(
                        "Do you have diabetes?", 
                        ["No", "No, borderline diabetes", "Yes (during pregnancy)", "Yes"], 
                        #vertical=True,
                        default=st.session_state.get("diabetes_value"),
                        key = "diabetes_input_whatif"
                    )
                    # asthma
                    asthma_value = st.segmented_control(
                        "Did you ever have asthma?", 
                        ["No", "Yes"], 
                        default=st.session_state.get("asthma_value"),
                        key = "asthma_input_whatif"
                    )
                    # Kidney disease
                    kidney_value = st.segmented_control(
                        "Did you ever have a kidney disease?", 
                        ["No", "Yes"], 
                        help="Apart from other than stones, bladder infection, or incontinence.",
                        default=st.session_state.get("kidney_value"),
                        key = "kidney_input_whatif"
                    )
                    # Skin Cancer
                    skin_value = st.segmented_control(
                        "Did you ever have skin cancer?", 
                        ["No", "Yes"], 
                        default=st.session_state.get("skin_value"),
                        key = "skin_input_whatif"
                    )
                    # stroke
                    stroke_value = st.segmented_control(
                        "Did you ever have a stroke?", 
                        ["No", "Yes"], 
                        default=st.session_state.get("stroke_value"),
                        key = "stroke_input_whatif"
                    )

                    # Add some spacing
                    st.write("")

                # Submit button to save profile
                if st.form_submit_button("💾 Save 'What if'-Scenario & Get Risk Estimate",
                    key = "prediction_input_whatif",
                    type="primary",
                    use_container_width=True): 

                    # age
                    age_cat = assign_age_cat(age_value)

                    # Calculate BMI
                    bmi_value = weight_value / ((height_value / 100) ** 2)
                    st.info(f"📊 Your BMI is: **{round(bmi_value, 2)}**")

                    # alc
                    if (alc_value > 14 and sex_value == "Male") or (alc_value > 7 and sex_value == "Female"):
                        alc_cat = "Yes"
                    else:
                        alc_cat = "No"
                    
                    with st.spinner("Estimating your heart attack risk..."):
                        user2 = pd.DataFrame({             
                            'Sex': [sex_value],
                            'Age': [age_value],
                            'Race': [race_cat],
                            'SleepTime': [sleep_value],
                            'AgeCategory': [age_cat],
                            'Smoking': [smoker_value], 
                            'AlcoholDrinking': [alc_cat],
                            'BMI': [round(bmi_value, 2)],
                            'Diabetic': [diabetes_value],                
                            'Stroke': [stroke_value],
                            'Asthma': [asthma_value],
                            'KidneyDisease': [kidney_value],
                            'SkinCancer': [skin_value],
                            'PhysicalActivity':[excercise_value],
                            'MentalHealth': [mentalhealth_value],
                            'DiffWalking': [walk_value],
                            'GenHealth': [health_cat],
                            'PhysicalHealth': [physhealth_value] 
                        })

                        #prediction = model.predict(user)
                        model = st.session_state["model"]
                        prediction = model.predict_proba(user2)[:, 1] #>= 0.5
                        st.session_state["risk_value2"] = round(prediction[0]*100,2)  # Store as percentage
                        st.session_state["profile_submitted2"] = True
                        st.session_state["just_saved2"] = True  # Flag to show we just saved
                        st.session_state["user_data2"] = user2  # Store user data for later use
                    
                    st.session_state.scroll_to_top = True
                    st.rerun()
    
    # Display results in col_right
    if st.session_state["profile_submitted2"]:
        with col_left:
            with tab2:
                st.success("✅ 'What-If' scenario saved!")
                user2 = st.session_state["user_data2"]
                st.info(f"📊 Your BMI would be: **{user2.loc[0,"BMI"]}**") # maybe get from session_state user to not confuse with whatif?
                if user2.loc[0,"AlcoholDrinking"] == "Yes":
                    st.info("⚠️ Your alcohol consumption would be high.")

    if st.session_state["profile_submitted"]:
        with col_left:
            with tab1:
                st.success("✅ Profile saved!")
                st.info(f"📊 Your BMI is: **{round(st.session_state["bmi_value"], 2)}**")
                if st.session_state["alc_cat"] == "Yes":
                    st.info("⚠️ High alcohol consumption.")
        with col_right:
            col1, col2 = st.columns([1,1])
            with col1:
                st.write("#### Your current heart attack risk factor:", str(st.session_state["risk_value"]),"%")
            
                # Confidence Interval(CI)
                cache_key = "ci_result"
                if cache_key not in st.session_state:
                    with st.spinner("Calculating a confidence interval..."):
                        model = st.session_state["model"]
                        X_user = st.session_state["user_data"]
                        lower_ci, upper_ci = bootstrap_confidence_interval_cached(model, X_user)
                        st.session_state[cache_key] = (lower_ci, upper_ci)
                else:
                    lower_ci, upper_ci = st.session_state[cache_key]

                # Display the confidence interval
                st.write(f":grey[🔎 **95% Confidence Interval:** {lower_ci:.1f}% – {upper_ci:.1f}%]")
                # END CI

                with st.expander("What does this mean?"):
                    st.write("""
This score reflects how similar your answers are to people in the survey who reported a heart attack.

**What this doesn’t mean:**
* It is not a medical diagnosis
* It doesn’t tell your personal real-world risk

**What this does mean:**
* People with similar answers in the dataset had this likelihood of reporting a heart attack
* It helps identify patterns that might be worth discussing with a doctor

                             
**95% Confidence Interval:**
We're 95% confident your actual score falls somewhere in this range. The wider the range, the less certain we are about the exact number.

**Risk category:**                             
Below you can find your risk category.  We divide the probability score into different risk categories using thresholds  
chosen to *reduce false negatives* in the high-risk group.

A *false negative* means the model predicts *“low risk”* even though the
person’s information is similar to people in the dataset who *reported a heart attack*.

Reducing false negatives helps ensure that users with a high-risk pattern
are not mistakenly classified as low risk.
                             """)

            with col2:
# What-if info box---------------------------------------------------------
                if st.session_state["risk_value2"] is not False:
                    X_user = st.session_state["user_data"]
                    X_user2 = st.session_state["user_data2"]

                    cache_key = "ci_result_whatif"
                    if cache_key not in st.session_state:
                        with st.spinner("Running the simulation..."):
                            model = st.session_state["model"]                           
                            lower_ci2, upper_ci2 = bootstrap_confidence_interval_cached(model, X_user)
                            st.session_state[cache_key] = (lower_ci2, upper_ci2)
                    else:
                        lower_ci2, upper_ci2 = st.session_state[cache_key]
                    
                        # Collect changes
                        differences = display_changes_compact(X_user, X_user2)                       

                        # Assemble infobox content
                        info_text = ["#### 💭 What if?"]
                        info_text.append("##### Changed Risk Factors")

                        if differences:                        
                            info_text.extend(differences)
                            info_text.append(
                                f"##### What-If Heart Attack Risk: **{st.session_state['risk_value2']}%**"
                            )

                            info_text.append(
                                f"🔎 **95% Confidence Interval:** "
                                f"{lower_ci2:.1f}% – {upper_ci2:.1f}%"
                            )
                        else:
                            info_text.append("_No changes detected._")   
                    # Display everything in one infobox
                    st.info("\n\n".join(info_text))

# Gauge plot ----------------------------------------------------------
            
            # Get risk value and compute confidence interval
            with st.spinner("Plotting your risk..."):
                risk_percent = st.session_state["risk_value"]
                lower_ci, upper_ci = st.session_state["ci_result"]

                # Optional: Get What-If risk if available
                whatif_risk = st.session_state.get("risk_value2", None)

                # Create gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_percent,
                    
                    number={
                        "suffix": "%",
                        "font": {"size": 48, "family": "Arial, sans-serif", "color": "#C2185B"}
                    },

                    title={
                        "text": "<b>Heart Risk Level</b>",
                        "font": {"size": 26, "family": "Arial, sans-serif", "color": "#880E4F"}
                    },

                    gauge={
                        "axis": {
                            "range": [0, 100], 
                            "tickwidth": 1, 
                            "tickcolor": "#C2185B",
                            "showticklabels": True
                        },
                        "borderwidth": 0,

                        # Main needle color
                        "bar": {"color": "#C2185B", "thickness": 0.40},

                        # Gradient zones with CI and What-If as steps
                        "steps": [
                            {"range": [0, 20], "color": "#FCE4EC"},
                            {"range": [20, 40], "color": "#F8BBD0"},
                            {"range": [40, 60], "color": "#F06292"},
                            {"range": [60, 80], "color": "#E91E63"},
                            {"range": [80, 100], "color": "#C2185B"},
                            {"range": [lower_ci, upper_ci], "color": "#DFDBD8"},
                        ] + (
                            # Add CI and What-If steps if What-If risk is available
                            [
                                {"range": [lower_ci2, whatif_risk - 0.25,], "color": "#E8E5F3"},  # Light blue
                                {"range": [whatif_risk -0.25, whatif_risk + 0.25], "color": "#0055A4"},  # Blue dot as thin step
                                {"range": [whatif_risk + 0.25, upper_ci2], "color": "#E8E5F3"},  # Light blue
                            ] if (whatif_risk is not None and whatif_risk is not False) else []
                        ),

                        # Threshold marker (dark pink) - main point estimate
                        "threshold": {
                            "line": {"color": "#880E4F", "width": 6},
                            "thickness": 0.9,
                            "value": risk_percent
                        }
                    }
                ))

                # Add heart emoji - positioned in gauge coordinate system
                fig.add_annotation(
                    x=0.5,
                    y=0.25,
                    text="❤️",
                    font=dict(size=42, color="crimson"),
                    showarrow=False,
                    xref="paper",
                    yref="paper"
                )            

                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=450,
                    margin=dict(l=40, r=40, b=40, t=80),
                    # Remove gridlines
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                )

                # Show gauge in Streamlit
                st.plotly_chart(fig, width='stretch')
            # End Gauge chart

## Risk tiers -------------------------------------------------------
            if st.session_state["risk_value"] >= 25:
                st.error("🚨 High Risk! Please consult a healthcare professional for a comprehensive evaluation.")
            elif st.session_state["risk_value"] >= 10:
                st.warning("⚠️ Moderate Risk. Consider lifestyle changes and regular check-ups.") 
            else:
                st.info("✅ Low Risk. Maintain a healthy lifestyle to keep your risk low.") 

            st.write("-------------")

# start feature importance chart ------------------------------------
            # Get SHAP values for your user        
            model = st.session_state["model"]
            user = st.session_state["user_data"]
            background_sample = st.session_state["background_sample"]

            if st.session_state["risk_value2"] is not False:
                user2 = st.session_state["user_data2"]
                
                # Check if plot is already cached in session state
                cache_key = "shap_plot_with_whatif"
                if cache_key not in st.session_state:
                    with st.spinner('Updating importances...'):
                        fig, importance_df = compute_shap_plot_voting_cached(
                            model, user, user2, background_sample
                        )
                        st.session_state[cache_key] = (fig, importance_df)
                else:
                    fig, importance_df = st.session_state[cache_key]
            else:
                cache_key = "shap_plot_no_whatif"
                if cache_key not in st.session_state:
                    with st.spinner('Plotting importances...'):
                        fig, importance_df = compute_shap_plot_voting_cached(
                            model, user, None, background_sample
                        )
                        st.session_state[cache_key] = (fig, importance_df)
                else:
                    fig, importance_df = st.session_state[cache_key]
            
# divide based on feature importance-----------------------------------
            st.markdown("#### What Influenced Your Score")

            col1, col12 = st.columns(2)
            with col1:
                st.markdown("##### Priority actions: highly modifiable factors")
                high_risk = importance_df[
                    (importance_df['modifiability'] == 'highly') &
                    (importance_df['shap_value'] > 0)
                ].sort_values('shap_value', ascending=False)

                if len(high_risk):
                    for _, row in high_risk.iterrows():
                        st.write(f"🔴 **{row['feature']}**: +{row['shap_value']:.3f}")
                else:
                    st.success("No highly modifiable factors currently increasing risk.")
            with col12:
                st.markdown("##### Secondary actions: moderately modifiable factors")

                moderate_risk = importance_df[
                    (importance_df['modifiability'] == 'moderately') &
                    (importance_df['shap_value'] > 0)
                ].sort_values('shap_value', ascending=False)

                if len(moderate_risk):
                    for _, row in moderate_risk.iterrows():
                        st.write(f"🟠 **{row['feature']}**: +{row['shap_value']:.3f}")
                else:
                    st.info("No moderately modifiable factors currently increasing risk.")
            
            with st.expander("What Are Modifiable Factors?"):
                st.write("""
            Modifiable factors are health behaviors and conditions you have control over—like smoking cessation, physical activity levels, diet quality, alcohol consumption, and weight management. Unlike non-modifiable factors (age, sex, genetics), these can be changed to reduce your heart disease risk.")
                        """)
            # plotting
            with st.spinner("Visualizing all important factors..."):
                st.pyplot(fig, transparent=True, width='stretch')
            # end importance plot

## Shap explanation---------------------------------------------
            with st.expander("SHAP Values Explained"):
                st.write("""
SHAP values show how much each health factor (age, smoking, etc.) pushed the prediction higher or lower from the average baseline risk. Positive values increase heart disease risk, while negative values decrease it. All values add up to give the final prediction, explaining exactly why the model arrived at that specific percentage.
                         [→ Learn more about SHAP](https://shap.readthedocs.io/en/latest/index.html)
                         """)

# PDF Report dowload button ------------------------------------

            # Recommendations
            profile_text = "User profile information:\n"
            for k,v in user.items():
                profile_text += f"- {k}: {v}\n"

            profile_text += "\nUse this information when generating health recommendations.\n"

            # Initialize LLM
            llm = Groq(model="llama-3.3-70b-versatile", token=st.secrets["GROQ_API_KEY"])

            # Load embeddings & vector index (same as AI Assistant)
            embeddings = HuggingFaceEmbedding(model_name="sentence-transformers/distiluse-base-multilingual-cased-v1")
            # load Vector Database
            storage_context = StorageContext.from_defaults(persist_dir=VECTOR_INDEX_DIR)
            vector_index = load_index_from_storage(storage_context, embed_model=embeddings)
            retriever = vector_index.as_retriever(similarity_top_k=2)

            # Chat engine
            memory = ChatMemoryBuffer.from_defaults()
            rag_bot = ContextChatEngine(llm=llm, retriever=retriever, memory=memory, 
                                        prefix_messages=[ChatMessage(role=MessageRole.SYSTEM, content=profile_text)])

            # Generate initial recommendation
            answer = rag_bot.chat(
                "as bullet points with the following headings: Healthy Habits, Modifiable Risk Factors, Recommendations"
            )
            initial_recommendation = answer.response
           # Prepare the what-if data only if exists
            what_if_df = st.session_state.get("user_data2", None)
            what_if_risk = st.session_state.get("risk_value2", None)

            # Generate PDF
            pdf_buffer = create_pdf_report(
                user_df=st.session_state["user_data"],
                risk_value=st.session_state["risk_value"],
                ci_low=lower_ci,
                ci_high=upper_ci,
                what_if_df=what_if_df,
                what_if_risk=what_if_risk
            )

            # Red download button
            st.markdown("""
            <style>
            div.stDownloadButton > button {
                background-color: #f43f5e 
                color: white;
                font-weight: bold;
            }
            </style>
            """, unsafe_allow_html=True)

            st.download_button(
                label="📄 Download Heart Risk Report (PDF)",
                data=pdf_buffer,
                file_name="heart_risk_report.pdf",
                mime="application/pdf"
            )

# AI assistant button ------------------------------------------
            st.write("Head to the AI Assistant for personalized advice based on your profile.")              
            if st.button("Go to AI Assistant →", key="cta4"):
                st.switch_page("pages/02_AI_Assistance.py")




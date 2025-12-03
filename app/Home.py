import streamlit as st

# Page config
st.set_page_config(page_title="Home", page_icon="❤️", layout="wide")

# Custom CSS for modern look
st.markdown("""
    <style>
    .header-container {
        display: flex;
        align-items: center;
        justify-content: center;  /* Add this to center everything */
        gap: 1rem;
        margin-bottom: 0.5rem;
    }
    .big-title {
        font-size: 36;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
        /* Remove flex-grow: 1; */
        text-align: center;
    }
    .tagline {
        font-size: 2rem; 
        color: #444;
        margin-bottom: 1.5rem;
        text-align: center;
        font-weight: 500;
    }
    .intro-text {
        font-size: 2rem; 
        color: #555;
        line-height: 1.6;
        margin-bottom: 2rem;
        text-align: center;  
    }
    .stats-container {
        display: flex;
        gap: 2rem;
        justify-content: space-between;
        margin-top: 1.5rem;
    }
    .step-number {
        display: inline-block;
        width: 40px;
        height: 40px;
        background: #f43f5e;
        color: white;
        border-radius: 50%;
        text-align: center;
        line-height: 40px;
        font-weight: 700;
        margin-right: 1rem;
    }
    .step-container {
        margin-bottom: 1rem;
    }
    .feature-box {
        padding: 1.5rem;
        background: #f8f9fa;
        border-radius: 8px;
        border-left: 4px solid #f43f5e;
        margin-bottom: 1rem;
    }
    .metrics-box {
        padding: 1.5rem;
        background: #f8f9fa;
        margin-bottom: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f43f5e;
        border-right: 4px solid #f43f5e;
    }
    </style>
""", unsafe_allow_html=True)

# Hero section - full width header
st.markdown("#")
st.markdown("#")
st.markdown("""
    <div class="header-container">
        <h1 class="big-title">❤️ Love Your Heart</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown('<p class="tagline">Understand your heart health and reduce it with AI-powered insights</p>', unsafe_allow_html=True)
st.write("")

# Intro paragraph
st.markdown("""
    <p class="intro-text">
    Heart disease remains the leading cause of death globally, yet up to 90% of cases are preventable through lifestyle changes. 
    Understanding your personal risk factors is the first step toward a healthier heart. Our AI-powered tool provides you with 
    a personalized cardiovascular risk assessment and actionable guidance—empowering you to take control of your heart health today.
    </p>
""", unsafe_allow_html=True)
st.write("")

# Stats section
col_left, col_middle, col_right = st.columns([1, 1, 1], gap="large")

with col_left:
    st.markdown('''
    <div class="metrics-box">
        <strong>🌍 Global Impact</strong><br>
        10,000 deaths/day from heart disease (WHO Europe)
    </div>
    ''', unsafe_allow_html=True)
        
with col_middle:
    st.markdown('''
    <div class="metrics-box">
        <strong>✅ Preventable</strong><br>
        Up to 90% of heart disease can be prevented
    </div>
    ''', unsafe_allow_html=True)

with col_right:
    st.markdown('''
    <div class="metrics-box">
        <strong>💪 Take Action</strong><br>
        Know your risk, change your future
    </div>
    ''', unsafe_allow_html=True)

st.write("")
st.write("")
st.divider()
st.write("")
st.write("")

# Main content area
main_col, sidebar_col = st.columns([3, 2], gap="large")

with main_col:
    st.markdown("#### How It Works")

    # Step 1 - Assessment with expanded features
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([0.1, 0.9])
    with col1:
        st.markdown('<span class="step-number">1</span>', unsafe_allow_html=True)
    with col2:
        st.markdown("##### Get Your Personalized Risk Assessment")
        st.write("""
                 Enter your health and lifestyle information to receive an instant cardiovascular risk score.
            * **Visual Risk Score:** See your risk displayed as an easy-to-read gauge (Low, Moderate, or High)
            * **Understand Key Factors:** Discover which modifiable factors (smoking, exercise, diet, weight) 
            have the greatest impact on your heart health
            * **What-If Scenarios:** Explore how lifestyle changes could improve your risk score in real-time
            * **Download Results:** Save your assessment and recommendations to share with your healthcare provider
                 """)
        
        st.write("")
        if st.button("Start Assessment →", type="primary", key="cta1"):
            st.switch_page("pages/TestYourself.py")
  
    # Step 2 - AI Assistant
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([0.1, 0.9])
    with col1:
        st.markdown('<span class="step-number">2</span>', unsafe_allow_html=True)
    with col2:
        st.markdown("##### Chat with Your AI Health Assistant")
        st.write("Get personalized answers about your results, heart health strategies, and lifestyle modifications tailored to your profile.")
        st.write("")
        if st.button("Go to AI Assistant →", type="primary", key="cta2"):
            st.switch_page("pages/AIAssistance.py")
    st.markdown('</div>', unsafe_allow_html=True)

with sidebar_col:
    st.markdown("#### Why Use This Tool?")
    st.write("")
    
    st.markdown('''
    <div class="feature-box">
        <strong>📚 Research-Backed</strong><br>
        Built on peer-reviewed medical literature and clinical guidelines.
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
    <div class="feature-box">
        <strong>🔒 Privacy First</strong><br>
        Your health data stays with you. No storage, no sharing.
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
    <div class="feature-box">
        <strong>⚡ Instant Insights</strong><br>
        Get personalized answers in seconds, available 24/7.
    </div>
    ''', unsafe_allow_html=True)
        
    st.markdown('''
    <div class="feature-box">
        <strong>💡 Actionable Advice</strong><br>
        Receive clear, achievable steps to improve your heart health.
    </div>
    ''', unsafe_allow_html=True)
    
    if st.button("Learn more about the project →", key="cta3"):
        st.switch_page("pages/Background.py")

st.write("")
st.divider()

# Medical disclaimer
st.info("""
**Medical Disclaimer:** This tool provides educational information only and is not a substitute for professional medical advice. 
Always consult your healthcare provider for medical decisions. If you experience cardiac symptoms, seek immediate medical attention.
The AI assistant provides general guidance based on medical literature but cannot diagnose conditions or prescribe treatments.
""")

# Footer
col1, col2 = st.columns([3, 1])

with col1:
    st.caption("Powered by Llama 3.3 70B • Evidence-based • Privacy-focused • Free to use")

with col2:
    st.caption("Made with ❤️ for heart health")
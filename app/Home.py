
import streamlit as st

# Page config
st.set_page_config(page_title="Love Your Heart", page_icon="❤️", layout="wide")

# Custom CSS for modern look
st.markdown("""
    <style>
    .big-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 0.5rem;
    }
            .stats-container {
    display: flex;
    gap: 2rem;
    justify-content: space-between;
    margin-top: 1.5rem;
    }
    .stat-card {
        flex: 1;
        padding: 1.25rem;
        border-radius: 12px;
        background-color: var(--secondary-background-color);
    }
    .stat-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .stat-title {
        font-weight: 600;
        margin-top: 0.25rem;
    }
    .stat-subtitle {
        font-size: 0.9rem;
        opacity: 0.7;
    }
    .step-number {
        display: inline-block;
        width: 40px;
        height: 40px;
        background: #ff4b4b;
        color: white;
        border-radius: 50%;
        text-align: center;
        line-height: 40px;
        font-weight: 700;
        margin-right: 1rem;
    }
    .step-container {
        margin-bottom: 2rem;
    }
    .feature-box {
        padding: 1.5rem;
        background: #f8f9fa;
        border-radius: 8px;
        border-left: 4px solid #ff4b4b;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Hero section - asymmetric layout
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown('<h1 class="big-title">❤️ Love Your Heart</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">AI-powered insights for better cardiovascular health</p>',
        unsafe_allow_html=True
    )
    st.write("")
    # Horizontal layout
    col1, col2, col3 = st.columns(3, gap="medium")

    col1.metric(
        label="Global Impact",
        value="10,000",
        delta="deaths/day (WHO Europe)",
        delta_color="off",
        border=True
    )

    col2.metric(
        label="US Heart Attack Rate",
        value="Every 40 sec",
        delta="1 Heart Attack in the USA",
                delta_color="off",
        border=True
    )

    col3.metric(
        label="Preventable",
        value="Up to 90%",
        delta="of heart disease can be prevented",
        delta_color="off",
        border=True
    )


with col_right:
    st.markdown("#### Understand Your Risk")
    st.write("Quick cardiovascular risk assessment based on your lifestyle and health metrics.")
    st.write("")
    st.markdown("#### Evidence-Based Guidance")
    st.write("Answers from AI trained on medical literature, personalized to your profile.")

st.write("")
st.divider()
st.write("")

# Main content area
main_col, sidebar_col = st.columns([3, 2], gap="large")

with main_col:
    st.markdown("#### How It Works")

    # Step 1
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([0.1, 0.9])
    with col1:
        st.markdown('<span class="step-number">1</span>', unsafe_allow_html=True)
    with col2:
        st.markdown("##### Complete Your Profile")
        st.write("Share basic health information including age, BMI, diabetes status, and lifestyle factors. Takes about 2 minutes.")
        if st.button("Start Assessment →", type="primary", key="cta1"):
            st.switch_page("pages/TestYourself.py")
    # st.markdown('</div>', unsafe_allow_html=True)

    
    # Step 2
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([0.1, 0.9])
    with col1:
        st.markdown('<span class="step-number">2</span>', unsafe_allow_html=True)
    with col2:
        st.markdown("##### Review Your Metrics")
        st.write("See your calculated health metrics and cardiovascular risk profile based on clinical guidelines.")
    #st.markdown('</div>', unsafe_allow_html=True)

    
    # Step 3
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([0.1, 0.9])
    with col1:
        st.markdown('<span class="step-number">3</span>', unsafe_allow_html=True)
    with col2:
        st.markdown("##### Chat with AI Assistant")
        st.write("Ask questions about your results, heart health, and lifestyle changes tailored to your profile.")
        if st.button("Go to AI Assistant →", key="cta2"):
            st.switch_page("pages/AIAssistance.py")
    #st.markdown('</div>', unsafe_allow_html=True)

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
    if st.button("Learn more about the project →", type="primary", key="cta3"):
        st.switch_page("pages/Background.py")



st.write("")
st.divider()

# Medical disclaimer
st.info("""
**Medical Disclaimer:** This tool provides educational information only and is not a substitute for professional medical advice. 
Always consult your healthcare provider for medical decisions. If you experience cardiac symptoms, seek immediate medical attention.
The AI assistant provides general guidance based on medical literature but cannot diagnose conditions or prescribe treatments
""")


# Footer
col1, col2 = st.columns([3, 1])

with col1:
    st.caption("Powered by Llama 3.3 70B • Evidence-based • Privacy-focused • Free to use")

with col2:
    st.caption("Made with ❤️ for heart health")
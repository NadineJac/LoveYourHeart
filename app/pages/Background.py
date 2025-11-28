import streamlit as st
st.set_page_config(page_title="Background", page_icon="❤️", layout="wide")

with st.container():
    col_left, col_right = st.columns([2, 1], gap="large")

    with col_left:
        st.markdown("""
    # ❤️ Understanding Your Heart Health

    Cardiovascular diseases remain the leading cause of death worldwide, but here's the encouraging news: **many risk factors are within your control**. This tool helps you understand your personal risk profile and empowers you to take meaningful action toward a healthier heart.


    ### The Impact of Cardiovascular Disease

    #### A Global Health Priority

    Cardiovascular diseases claim one life every 34 seconds in the United States, accounting for 919,032 deaths in 2023 [[1]](https://www.cdc.gov/heart-disease/data-research/facts-stats/index.html). Globally, 10,000 people die from cardiovascular diseases every day in the WHO European Region [[2]](https://www.who.int/europe/news-room/15-05-2024-cardiovascular-diseases-kill-10-000-people-in-the-who-european-region-every-day--with-men-dying-more-frequently-than-women), representing over 42% of all deaths. In Germany specifically, cardiovascular disease has been the primary cause of death for decades, responsible for a significant portion of mortality and healthcare costs.

    Beyond the tragic loss of life, cardiovascular diseases place an enormous burden on healthcare systems and families. The economic impact extends beyond direct medical costs to include lost productivity, disability, and reduced quality of life for millions of individuals and their loved ones.


    ### Heart Attacks: A Critical Subset of Cardiovascular Disease

    #### What Is a Heart Attack?

    A heart attack, medically known as acute myocardial infarction, occurs when blood flow to part of the heart muscle becomes blocked, usually by a blood clot. This blockage prevents oxygen-rich blood from reaching heart tissue, causing damage or death to that portion of the heart muscle. Heart attacks are a subset of the broader category of cardiovascular diseases, which also includes stroke, heart failure, and other conditions affecting the heart and blood vessels.
""")
    with col_right:
        st.metric(
            label="Global Impact",
            value="10,000",
            delta="deaths/day (WHO Europe)",
            delta_color="off",
            border=True
        )
with st.container():
    col_left, col_right = st.columns([2, 1], gap="large")
    with col_left:
        st.markdown("""
    #### The Numbers Behind Heart Attacks

    **United States:**
    Approximately 805,000 Americans experience a heart attack each year[[3]](https://www.cdc.gov/heart-disease/data-research/facts-stats/index.html), with someone having a heart attack every 40 seconds. Of these, about 605,000 are experiencing their first heart attack, while 200,000 occur in people who have already had one. Notably, about one in five heart attacks are "silent"—the damage occurs, but the person isn't aware of it.

    **Europe and Germany:**
    In the European Union, there were 1.68 million deaths from diseases of the circulatory system in 2022, with ischemic heart diseases (which include heart attacks) being a leading cause. In Germany, the 12-month prevalence of coronary heart disease is 6.0% among men and 3.7% among women [[4]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10161269/). The prevalence increases dramatically with age, affecting up to 24.1% of men and 16.0% of women aged 75 years and older.

    While these statistics paint a serious picture, age-standardized mortality rates for acute myocardial infarction in Germany have fallen by an average of 4.2% per year for women and 4.1% per year for men between 1998 and 2023 [[5]](https://pmc.ncbi.nlm.nih.gov/articles/PMC12175194/), demonstrating that prevention and improved treatment are making a real difference.
    """)
        with st.spinner ("Plot data..."):
            import matplotlib.pyplot as plt
            import pandas as pd
            plot_data = pd.DataFrame({
                "Region": [
                    "United States",
                    "European Union"
                ],
                "value": [919_032, 1_680_000],  # numeric only
                "display": [
                    "919,032 deaths (2023)",
                    "1.68 million deaths (2022)"
                ],
                "info": [
                    "Leading cause of death for over 100 years",
                    "42.5% of all deaths annually in WHO European Region"
                ]
            })
            

            fig, ax = plt.subplots(figsize=(10, 2))

            # Reverse order so first item is on top
            plot_data = plot_data.iloc[::-1]

            bars = ax.barh(
                plot_data["Region"],
                plot_data["value"],
                color="#f43f5e",
                alpha=0.85
            )

            # Main value labels (on bars)
            for bar, value_label in zip(bars, plot_data["display"]):
                ax.text(
                    0.1e5,
                    bar.get_y() + bar.get_height() / 2,
                    f"   {value_label}",
                    va="center",
                    ha="left",
                    fontsize=10,
                    color="#FFFFFF",
                    fontweight="bold"
                )

        # Additional info BELOW each bar
        for bar, info in zip(bars, plot_data["info"]):
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f"   {info}",
                va="center",
                ha="left",
                fontsize=10,            
            )
        ax.set_xlabel("Absolute Number of Deaths")
        ax.set_ylabel("")

        # Remove spines
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)

        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

        plt.tight_layout()
        st.pyplot(fig, transparent=True, use_container_width=True)
    with col_right:
        st.metric(
            label="US Heart Attack Rate",
            value="Every 40 sec",
            delta="1 Heart Attack in the USA",
                    delta_color="off",
            border=True
        )

with st.container():
    col_left, col_right = st.columns([2, 1], gap="large")
    with col_left:
        st.markdown("""

    ## The Power of Prevention: Modifiable Risk Factors

    #### You Have More Control Than You Think

    Here's the truly empowering fact: research shows that strict control of five key modifiable risk factors could potentially prevent 57.2% of all cardiovascular disease cases in women and 52.6% in men globally[[6]](https://www.nejm.org/doi/full/10.1056/NEJMoa2206916). Another comprehensive study found that a large proportion of cardiovascular disease and premature deaths could be averted by targeting just a few modifiable risk factors [[7]](https://pmc.ncbi.nlm.nih.gov/articles/PMC8006904/).

    #### Key Modifiable Risk Factors

    The major risk factors you can influence include:

    - **High Blood Pressure (Hypertension):** Often called the "silent killer," high blood pressure is a leading risk factor for heart disease and can be managed through medication, diet, and lifestyle changes.

    - **High Cholesterol:** Elevated levels of certain cholesterol types contribute to plaque buildup in arteries, but this can be addressed through diet, exercise, and medication when needed.

    - **Smoking and Tobacco Use:** One of the most preventable risk factors, smoking more than doubles your risk of developing cardiovascular disease.

    - **Diabetes:** While diabetes increases heart disease risk, proper management and blood sugar control can significantly reduce this risk.

    - **Obesity and Physical Inactivity:** Regular physical activity and maintaining a healthy weight are powerful tools for reducing cardiovascular risk.

    - **Poor Diet:** A diet high in saturated fats, trans fats, and sodium increases risk, while a heart-healthy diet rich in fruits, vegetables, and whole grains protects your heart.

    - **Excessive Alcohol Consumption:** Drinking too much can raise blood pressure and increase heart disease risk.

    Up to 90% of heart disease is considered preventable with lifestyle changes and a proactive approach to prevention. Research consistently shows that maintaining good health through a nutrient-rich diet, managed weight, physical activity, and avoiding harmful habits like smoking can dramatically reduce your risk.
""")        
    with col_right:
        st.metric(
            label="Preventable",
            value="Up to 90%",
            delta="of heart disease can be prevented",
            delta_color="off",
            border=True
        )
with st.container():
    col_left, col_right = st.columns([2, 1], gap="large")
    with col_left:
        st.markdown("""

    ### Take Action Today

    Understanding your risk is the first step toward prevention. The good news is that it's never too late to make positive changes. Whether you're managing existing risk factors or working to prevent new ones from developing, small steps can lead to significant improvements in your heart health.

    **Remember:** Early detection and proactive management of risk factors can save lives. Use this tool to gain insights into your personal risk profile, and consider discussing the results with your healthcare provider to develop a personalized prevention plan.

    Your heart health is in your hands—and with the right knowledge and actions, you can significantly reduce your risk of heart attack and enjoy a longer, healthier life.
                """)

        # Data Sources section
        st.markdown("#### Data Sources")

        sources = """
        - **US Statistics**: [CDC Heart Disease Facts and Statistics](https://www.cdc.gov/heart-disease/data-research/facts-stats/index.html) 
        - **WHO European Region**: [WHO Europe News](https://www.who.int/europe/news-room/15-05-2024-cardiovascular-diseases-kill-10-000-people-in-the-who-european-region-every-day--with-men-dying-more-frequently-than-women)
        - **European Union Statistics**: [Eurostat - Cardiovascular Diseases Statistics](https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Cardiovascular_diseases_statistics)
        - **Germany CHD Prevalence**: [DEGS1 Study (PMC10161269)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10161269/)
        - **Germany Mortality Trends**: [German Cardiovascular Disease Trends (PMC12175194)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12175194/)
        """

        st.markdown(sources)

    with col_right:
        st.metric(
            label="12-month prevalence",
            value="6.0% men, 3.7% women",
            delta="of coronary heart disease in Germany",
                    delta_color="off",
            border=True
        )



import streamlit as st
import requests

st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    .main {max-width: 100vw !important; padding: 2rem 3rem !important;}
    .stNumberInput, .stSelectbox, .stSlider, .stRadio {margin-bottom: 1.5rem;}
    .header-style {font-size: 24px !important; color: #2c3e50 !important; border-bottom: 3px solid #3498db; padding-bottom: 0.4rem;}
    .prediction-card {padding: 2rem; border-radius: 12px; box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15); margin: 2rem 0;}
    .success {background-color: #e8f5e9; border-left: 6px solid #4caf50;}
    .danger {background-color: #ffebee; border-left: 6px solid #f44336;}
    .stForm {border: 1px solid #e0e0e0; border-radius: 12px; padding: 2rem;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("💳 Loan Default Prediction")
st.markdown(
    """
*Use this intelligent system to assess loan default risk. Complete the form below and click **Predict**.*
"""
)

with st.form("loan_form"):
    col_top1, col_top2, col_top3 = st.columns([1, 1, 1])

    with col_top1:
        with st.expander("**🧑 Personal Information**", expanded=True):
            person_age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=30,
                help="Applicant's age in years",
            )
            person_home_ownership = st.selectbox(
                "Home Ownership",
                ["RENT", "MORTGAGE", "OWN", "OTHER"],
                help="Current housing arrangement status",
            )

    with col_top2:
        with st.expander("**💵 Financial Status**", expanded=True):
            person_income = st.number_input(
                "Annual Income ($)",
                min_value=10000,
                max_value=200000,
                value=50000,
                step=1000,
                help="Gross annual income before taxes",
            )
            loan_amnt = st.number_input(
                "Loan Amount ($)",
                min_value=1000,
                max_value=50000,
                value=5000,
                step=500,
                help="Requested loan amount",
            )

    with col_top3:
        with st.expander("**📈 Loan Details**", expanded=True):
            loan_intent = st.selectbox(
                "Loan Purpose",
                [
                    "EDUCATION",
                    "MEDICAL",
                    "VENTURE",
                    "PERSONAL",
                    "DEBTCONSOLIDATION",
                    "HOMEIMPROVEMENT",
                ],
                help="Primary purpose for the loan",
            )
            loan_grade = st.selectbox(
                "Loan Grade",
                ["A", "B", "C", "D", "E", "F", "G"],
                help="Risk classification grade (A=best, G=worst)",
            )

    with st.expander("**📊 Additional Parameters**", expanded=True):
        col_bot1, col_bot2, col_bot3 = st.columns(3)

        with col_bot1:
            person_emp_length = st.slider(
                "Employment Length (years)",
                0,
                40,
                5,
                help="Total years of employment experience (0 = less than 1 year)",
            )
            loan_percent_income = st.slider(
                "Loan/Income Ratio (%)",
                5,
                50,
                15,
                help="Loan amount as percentage of annual income",
            )

        with col_bot2:
            cb_person_cred_hist_length = st.slider(
                "Credit History Length (years)",
                0,
                15,
                5,
                help="Duration of credit history in years",
            )
            loan_int_rate = st.slider(
                "Interest Rate (%)",
                5,
                30,
                10,
                help="Annual interest rate for the loan",
            )

        with col_bot3:
            cb_person_default_on_file = st.radio(
                "Previous Default History",
                ["N", "Y"],
                horizontal=True,
                help="Has the applicant defaulted on a loan before?",
            )

    submitted = st.form_submit_button("Predict Default Risk", use_container_width=True)

if submitted:
    input_data = {
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
    }

    with st.spinner("🔍 Analyzing application details..."):
        try:
            response = requests.post("http://localhost:8000/predict", json=input_data)
            if response.status_code == 200:
                prediction = response.json()
                pred_value = prediction["prediction"][0]
                result = "High Risk" if pred_value == 1 else "Low Risk"
                color_class = "danger" if pred_value == 1 else "success"
                emoji = "⚠️" if pred_value == 1 else "✅"

                st.markdown(
                    f"""
                <div class="prediction-card {color_class}">
                    <h3 style="margin:0;">{emoji} Prediction Result</h3>
                    <p style="font-size:26px; margin:1rem 0;">Risk Status: <strong>{result}</strong></p>
                    <p style="color:#666; margin:0;">Note: This prediction is based on machine learning analysis of historical data</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                st.subheader("📋 Application Summary")
                cols = st.columns(3)
                with cols[0]:
                    st.write("**Personal Details**")
                    st.metric("Age", person_age)
                    st.metric("Home Ownership", person_home_ownership)
                    st.metric("Employment Length", f"{person_emp_length} yrs")

                with cols[1]:
                    st.write("**Financial Details**")
                    st.metric("Annual Income", f"${person_income:,}")
                    st.metric("Loan Amount", f"${loan_amnt:,}")
                    st.metric("Interest Rate", f"{loan_int_rate}%")

                with cols[2]:
                    st.write("**Credit Details**")
                    st.metric("Credit History", f"{cb_person_cred_hist_length} yrs")
                    st.metric("Previous Defaults", cb_person_default_on_file)
                    st.metric("Loan Purpose", loan_intent.title())

            else:
                st.error("Error in prediction service. Please try again later.")

        except requests.exceptions.RequestException as e:
            st.error(f"Connection Error: Unable to reach prediction service. {str(e)}")
        except Exception as e:
            st.error(f"Unexpected Error: {str(e)}")

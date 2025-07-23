import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from io import BytesIO
from xhtml2pdf import pisa

# --- Load Model and Supporting Files ---
try:
    model = joblib.load('model/salary_prediction_model.joblib')
    feature_columns = joblib.load('model/feature_columns.joblib')
    categorical_values = joblib.load('model/categorical_values.joblib')
    df = pd.read_csv('indian_employee_salary_dataset.csv')
    df.rename(columns={'Monthly Salary (INR)': 'Salary', 'Experience (Years)': 'Experience'}, inplace=True)
except FileNotFoundError:
    st.error("Model files not found! Please run `model_training.py` first to generate them.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
def load_css():
    st.markdown("""
        <style>
            /* Main app background */
            .stApp {
                background-color: #f0f2f5;
            }
            
            /* Sidebar styling */
            [data-testid="stSidebar"] {
                background-color: #0d1b2a; /* Darker sidebar */
            }
            
            /* Form background */
            [data-testid="stForm"] {
                background-color: #ffffff;
                padding: 25px;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }

            /* Main Title styling */
            h1 {
                color: #1e2a38;
                text-align: center;
                font-weight: 700;
            }

            /* Form Header styling */
            .stForm h2 {
                color: #1e2a38;
                font-weight: 600;
                border-bottom: 2px solid #5e72e4;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }

            /* Button styling */
            .stButton>button {
                background-color: #5e72e4;
                color: white;
                border-radius: 8px;
                padding: 12px 20px;
                border: none;
                font-weight: 600;
                transition: all 0.3s ease;
                width: 100%;
            }
            .stButton>button:hover {
                background-color: #324cdd;
                box-shadow: 0 4px 8px rgba(94,114,228,0.4);
                transform: translateY(-2px);
            }

            /* Success Message Styling */
            [data-testid="stSuccess"] {
                background-color: #32cd32;
                color: #155724;
                border: 1px solid #c3e6cb;
                border-radius: 8px;
                font-weight: 600;
                text-align: center;
                padding: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)

load_css()

# --- PDF Report Generation Function ---
def create_pdf_report(inputs, prediction):
    html = f"""
    <html>
    <head>
        <style>
            @page {{ size: A4; margin: 2cm; }}
            body {{ font-family: 'Helvetica', 'Arial', sans-serif; color: #333; }}
            h1 {{ color: #1e2a38; text-align: center; }}
            h2 {{ color: #5e72e4; border-bottom: 1px solid #ddd; padding-bottom: 5px;}}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .result {{ background-color: #eef2ff; padding: 20px; text-align: center; margin-top: 30px; border-radius: 8px; }}
            .result h3 {{ color: #1e2a38; margin: 0; }}
            .result p {{ font-size: 24px; font-weight: bold; color: #5e72e4; margin: 5px 0; }}
        </style>
    </head>
    <body>
        <h1>Salary Prediction Report</h1>
        <p>This report was generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Input Details</h2>
        <table>
            <tr><th>Feature</th><th>Value</th></tr>
            <tr><td>Age</td><td>{inputs['Age']}</td></tr>
            <tr><td>Gender</td><td>{inputs['Gender']}</td></tr>
            <tr><td>Highest Education</td><td>{inputs['Education']}</td></tr>
            <tr><td>Job Title</td><td>{inputs['Job Title']}</td></tr>
            <tr><td>Years of Experience</td><td>{inputs['Experience']} years</td></tr>
        </table>
        
        <div class="result">
            <h3>Predicted Monthly Salary</h3>
            <p>{prediction['prediction']}</p>
            <p style="font-size: 16px; font-weight: normal; color: #6c757d;">{prediction['range']}</p>
        </div>
    </body>
    </html>
    """
    pdf_file = BytesIO()
    pisa_status = pisa.CreatePDF(BytesIO(html.encode('UTF-8')), dest=pdf_file)
    if pisa_status.err:
        return None
    pdf_file.seek(0)
    return pdf_file

# --- Sidebar Content ---
with st.sidebar:
    st.header("About")
    st.info("This app predicts employee salaries using a machine learning model trained on a dataset of Indian professionals.")
    st.header("Input Features")
    st.markdown("- **Age:** Employee's age\n- **Gender:** Male/Female/Other\n- **Education:** Highest qualification\n- **Job Title:** Current role\n- **Experience:** Total years of work experience")

# --- Main Application UI ---
st.title("💼 Employee Salary Predictor")
st.markdown("<p style='text-align: center; color: #6c757d;'>Enter employee details to get a salary prediction. Explore the data visualizations below to understand salary trends.</p>", unsafe_allow_html=True)

# --- Input Form ---
with st.form("prediction_form"):
    st.markdown("<h2>Employee Details</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 70, 30)
        gender = st.selectbox("Gender", sorted(categorical_values['Gender']))
    with col2:
        experience = st.number_input("Years of Experience", 0.0, 40.0, 5.0, 0.5)
        education = st.selectbox("Highest Education", sorted(categorical_values['Education']))
    
    job_title = st.selectbox("Job Title", sorted(categorical_values['Job Title']))
    
    submit_button = st.form_submit_button(label="✨ Predict Salary")

# --- Prediction Logic and Display ---
if submit_button:
    input_data = {'Age': age, 'Gender': gender, 'Education': education, 'Job Title': job_title, 'Experience': experience}
    input_df = pd.DataFrame([input_data], columns=feature_columns)
    
    with st.spinner('Analyzing data and making a prediction...'):
        prediction_val = model.predict(input_df)[0]
    
    lower_bound = prediction_val * 0.90
    upper_bound = prediction_val * 1.10
    
    prediction_result = {
        'prediction': f"₹{prediction_val:,.0f} per month",
        'range': f"Estimated Range: ₹{lower_bound:,.0f} to ₹{upper_bound:,.0f}"
    }

    st.success("Prediction Generated Successfully!")
    
    # Custom result display box
    st.markdown(f"""
    <div style="background-color: #ffffff; padding: 25px; border-radius: 10px; border-left: 10px solid #28a745; box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center;">
        <p style="font-size: 1.2rem; color: #6c757d; margin: 0;">Predicted Monthly Salary</p>
        <p style="font-size: 2.5rem; font-weight: 700; color: #5e72e4; margin: 0;">{prediction_result['prediction']}</p>
        <p style="font-size: 1.1rem; color: #6c757d; margin-top: 10px;">{prediction_result['range']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("") # Add some vertical space
    
    # --- PDF Download Button ---
    pdf_report = create_pdf_report(input_data, prediction_result)
    if pdf_report:
        st.download_button(
            label="📥 Download Prediction Report as PDF",
            data=pdf_report,
            file_name="salary_prediction_report.pdf",
            mime="application/pdf"
        )

st.markdown("---")
# --- Data Visualizations ---
st.markdown("<h2 style='text-align: center; color: #1e2a38;'>Dataset Insights</h2>", unsafe_allow_html=True)
c1, c2 = st.columns(2)

with c1:
    avg_salary_by_job = df.groupby('Job Title')['Salary'].mean().sort_values(ascending=False).head(10)
    fig1 = px.bar(avg_salary_by_job, x=avg_salary_by_job.values, y=avg_salary_by_job.index, orientation='h',
                  title='Top 10 Highest Average Salaries by Job Title', labels={'x': 'Average Monthly Salary (₹)', 'y': 'Job Title'},
                  color=avg_salary_by_job.values, color_continuous_scale=px.colors.sequential.Plasma)
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    education_dist = df['Education'].value_counts()
    fig2 = px.pie(education_dist, values=education_dist.values, names=education_dist.index,
                  title='Education Level Distribution in Dataset',
                  color_discrete_sequence=px.colors.sequential.Plasma_r)
    st.plotly_chart(fig2, use_container_width=True)

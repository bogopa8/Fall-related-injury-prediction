
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from streamlit_shap import st_shap
import shap
from PIL import Image


def calibration(data, train_pop, target_pop, sampled_train_pop, sampled_target_pop):
    calibrated_data =\
    ((data * (target_pop / train_pop) / (sampled_target_pop / sampled_train_pop)) /
    ((
        (1 - data) * (1 - target_pop / train_pop) / (1 - sampled_target_pop / sampled_train_pop)
     ) +
     (
        data * (target_pop / train_pop) / (sampled_target_pop / sampled_train_pop)
     )))
    return calibrated_data

def get_rank_by_pred_prob(df, value):
    #Filter the DataFrame to include only rows where pred_prob <= value
    df['predicted_probability'] = df['predicted_probability'].astype(float)
    sorted_df = df.sort_values(by='predicted_probability')
    filtered_df = sorted_df[sorted_df['predicted_probability'] >= value]
    if filtered_df.empty:
        return 0.1
    closest = filtered_df.iloc[0]
    return closest['rank']


def run_ml_app():
    st.title("Fall related injury prediction") 
    st.subheader("1. Prediction Results")   
    st.sidebar.header("Fall-related injury prediction")
    st.sidebar.subheader("Demographic")
    age_group=st.sidebar.radio("Age group", ('65~69', '70~74', '75+'), horizontal=True)
    Sex=st.sidebar.radio("Sex", ('Male', 'Female'), horizontal=True)
    medical_aid=st.sidebar.radio("medical aid", ('NHI', 'Medical aid'), horizontal=True)
    st.sidebar.subheader("Health care utilization")
    No_admission=st.sidebar.select_slider("No. of Admission or ED visits", options=np.arange(0,7))
    st.sidebar.subheader("Comorbidities")
    prior_fri=st.sidebar.radio("Prior fall-related injury",('No', 'Yes'), horizontal=True)
    dorsopathy=st.sidebar.radio("Dorsopathy", ('No', 'Yes'), horizontal=True)
    hyperlipidemia=st.sidebar.radio("Dyslipidemia", ('No', 'Yes'), horizontal=True)
    urinary=st.sidebar.radio("Urinary incontinence", ('No', 'Yes'), horizontal=True)
    parkinson=st.sidebar.radio("Parkinson disease", ('No', 'Yes'), horizontal=True)
    thyroid_dz=st.sidebar.radio("Thyroid disease", ('No', 'Yes'), horizontal=True)
    menopause=st.sidebar.radio("Menopause", ('No', 'Yes'), horizontal=True)     
    st.sidebar.subheader("Medications")
    medication=st.sidebar.select_slider("No.of medications", options=np.arange(0, 21))
    CNS_medication=st.sidebar.select_slider("No. of CNS active drugs", options=np.arange(0,7))
    loop=st.sidebar.radio("Loop diuretics", ('No', 'Yes'), horizontal=True)
    bb=st.sidebar.radio("Beta-blocker", ('No', 'Yes'), horizontal=True)
    rasi=st.sidebar.radio("RAS inhibitor", ('No', 'Yes'), horizontal=True)
    su=st.sidebar.radio("Sulfonylurea", ('No', 'Yes'), horizontal=True)
    tzd=st.sidebar.radio("Thiazolidinedione", ('No', 'Yes'), horizontal=True)
    achei=st.sidebar.radio("Acetylcholine esterase inhibitor", ('No', 'Yes'), horizontal=True)
    steroid=st.sidebar.radio("Steroid", ('No', 'Yes'), horizontal=True)
    bpn=st.sidebar.radio("Bisphosphonate", ('No', 'Yes'), horizontal=True)
    vitd=st.sidebar.radio("Vitamind D", ('No', 'Yes'), horizontal=True)
    hematopoietic=st.sidebar.radio("Hematopoietics", ('No', 'Yes'), horizontal=True)
    hormonal_chemo=st.sidebar.radio("Hormonal chemotherapy", ('No', 'Yes'), horizontal=True)
    st.sidebar.subheader("Drug-disease interaction")
    fracture_cns=st.sidebar.radio("CNS active drugs and prior fracture", ('No', 'Yes'), horizontal=True)
    cilo_hf=st.sidebar.radio("Cilostazol and heart failure", ('No', 'Yes'), horizontal=True)

    if age_group=='65~69':
        age_group=0
    elif age_group=='70~74':
        age_group=1
    else:
        age_group=2
    if Sex=='Male':
        Sex=1
    else:
        Sex=2
    if medical_aid=='NHI':
        medical_aid=0
    else:
        medical_aid=1
    var_list=[prior_fri, dorsopathy, hyperlipidemia, urinary, parkinson, thyroid_dz, menopause, loop, bb, rasi, su, tzd, achei, steroid, bpn, vitd, hematopoietic, hormonal_chemo, fracture_cns, cilo_hf]
    for i in range(len(var_list)):
        if var_list[i] == 'Yes':
            var_list[i] = 1
        else:
            var_list[i] = 0
    (prior_fri, dorsopathy, hyperlipidemia, urinary, parkinson, thyroid_dz, menopause, loop, bb, rasi, su, tzd, achei, steroid, bpn, vitd, hematopoietic, hormonal_chemo, fracture_cns, cilo_hf) = var_list

    
    sample = [Sex, age_group, medical_aid, No_admission, prior_fri, hyperlipidemia, dorsopathy, parkinson, menopause, thyroid_dz,  urinary, medication, CNS_medication, loop, bb, rasi, su, tzd, steroid, vitd, bpn,  achei, hormonal_chemo, hematopoietic,  cilo_hf, fracture_cns]
    new_df=np.array(sample).reshape(1,-1)       
    importance_PATH = 'model/shap_importance.png'
    model = joblib.load('model/cat_stream.pkl')
    explainer = joblib.load('model/explainer.pkl')
    rank = joblib.load('model/rank.pkl')
    pred_prob_o = model.predict_proba(new_df)[0][1]
    pred_prob_n = calibration(pred_prob_o, 520603, 5191, 520603, 9127)
    pred_prob_s = round(calibration(pred_prob_o, 520603, 5191, 520603, 9127)*100, 1)
    rank_o = round(get_rank_by_pred_prob(rank, pred_prob_n), 1)
    if pred_prob_o<0.01842620042670873:
        st.success("Low risk")
        st.write(f"Estimated 90-day risk of fall-related injury is: :blue[{pred_prob_s}%]")
        st.write(f"The patient's fall-related injury risk among community-dwelling older adults is in the top: :blue[{rank_o}%]")
    else:
        st.error("High risk")
        st.write(f"Estimated 90-day risk of fall-related injury is: :red[{pred_prob_s}%]")
        st.write(f"The patient's fall-related injury risk among community-dwelling older adults is in the top: :red[{rank_o}%]")
        
    st.subheader("2. Model Interpretation-Population level")
    with st.expander('Click to expand'):
        beeswarm = Image.open('model/shap_summary_cat.png')
        st.subheader("2.1 SHAP-beeswarm plot")
        st.markdown("Population-level prediction is depicted. Variables are ordered with respect to their importance on prediction. The color represents the value of each features, with red representing higher values and blue representing lower values. SHAP-value on x-axis explains the direction and degree of the model’s prediction where large positive values contribute to the prediction that the patient will fall, large negative values contribute to the prediction that the patient will not fall, and values close to zero contribute little to the prediction (Female sex is colored red(=2) and male is colored blue(=1)). "
, unsafe_allow_html=False)
        st.image(beeswarm)
        st.subheader("2.2 SHAP-variable importance")
        imp = Image.open('model/shap_importance.png')
        st.image(imp)
    st.subheader("3. Model Interpretation-Individual patient level")
    st.subheader("3.1 SHAP-waterfall")
    st.markdown("Patient-level prediction is depicted. Variables are ordered with respect to their importance on prediction. The SHAP value on the x-axis explains the direction and degree of the model’s prediction, where large positive values contribute to the prediction that a patient will experience fall-related injury, large negative values contribute to the prediction that a patient will not experience fall-related injury, and values close to zero contribute little to the prediction (Male sex is coded as 1 and female is coded as 2).")
    new_df=pd.DataFrame(new_df, columns=['Sex', 'Age group',  'Medical aid', 'No. of admission/ED visit', 'Prior FRI',  'Hyperlipidemia', 'Dorsopathy',  'Parkinson disease',  'Menoapuse', 'Thyroid disease', 'Urinary incontinence', 'No. of medication', 'No. of CNS depressant', 'Loop diuretic', 'Beta-blocker',  'RAS inhibitor', 'Sulfonylurea', 'Thiazolidinedione', 'Steroid','VitD', 'Bisphosphonate', 'Acetylcholine esterase inhibitor', 'Hormonal chemotherapy',  'Hematopoietic', 'Cilostazol and heart failure', 'Fracture and CNS depressant']) 
    shap_values=explainer(new_df)
    menu = ['Top 10', 'All']
    choice = st.selectbox("Feature display", menu)
    if choice == "Top 10":
        st_shap(shap.plots.waterfall(shap_values[0], max_display=10))
    else:
        st_shap(shap.plots.waterfall(shap_values[0], max_display=26))
    
        
            






def main():
    
    #st.markdown(html_temp, unsafe_allow_html=True)
    st.sidebar.header("Menu")
    menu = ['HOME', 'Fall-related injury prediction']
    choice = st.sidebar.selectbox("select menu", menu)
    if choice == 'HOME':
        st.subheader('HOME')
        st.subheader("Fall-related injury prediction")
        st.markdown("Falls in older adults are a major public health problem. They can occur in any age, but the incidence and severity of fall and fall-related injuries increase with age. More than one out of four older adults fall annually, 10% of older adults reported an injury from a fall, and falls are a leading cause of death from unintentional injury")
        st.markdown("In this regard, this tool was developed to predict the risk of fall-related injuries among community-dwelling elderly individuals. The development process and performance of the model is described in the following paper.")
        st.markdown("[Heo KN, et al. Development and validation of a machine learning-based fall-related injury risk prediction model using a nationwide claims database in the Korean community-dwelling older population. BMC Geriatr. 2023 Dec 11;23(1):830.]")
    
    else:
        run_ml_app()


if __name__=="__main__":
    main()
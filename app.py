import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(page_title="Employee Salary Prediction", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")


# ------------------------------------------------------------
# Load trained model (LabelEncoded categorical features)
# ------------------------------------------------------------
@st.cache_resource
def load_model(path: str = "best_model.pkl"):
    return joblib.load(path)

model = load_model()


# ------------------------------------------------------------
# LabelEncoder-style mappings (from your training data)
# IMPORTANT: Keys **must** match EXACT raw strings that appeared in training.
# Index values must match what LabelEncoder produced during training.
# ------------------------------------------------------------
workclass_mapping = {
    '?': 0,
    'Federal-gov': 1,
    'Local-gov': 2,
    'Never-worked': 3,
    'Private': 4,
    'Self-emp-inc': 5,
    'Self-emp-not-inc': 6,
    'State-gov': 7,
    'Without-pay': 8,
}

marital_status_mapping = {
    'Divorced': 0,
    'Married-AF-spouse': 1,
    'Married-civ-spouse': 2,
    'Married-spouse-absent': 3,
    'Never-married': 4,
    'Separated': 5,
    'Widowed': 6,
}

occupation_mapping = {
    '?': 0,
    'Adm-clerical': 1,
    'Armed-Forces': 2,
    'Craft-repair': 3,
    'Exec-managerial': 4,
    'Farming-fishing': 5,
    'Handlers-cleaners': 6,
    'Machine-op-inspct': 7,
    'Other-service': 8,
    'Priv-house-serv': 9,
    'Prof-specialty': 10,
    'Protective-serv': 11,
    'Sales': 12,
    'Tech-support': 13,
    'Transport-moving': 14,
}

relationship_mapping = {
    'Husband': 0,
    'Not-in-family': 1,
    'Other-relative': 2,
    'Own-child': 3,
    'Unmarried': 4,
    'Wife': 5,
}

race_mapping = {
    'Amer-Indian-Eskimo': 0,
    'Asian-Pac-Islander': 1,
    'Black': 2,
    'Other': 3,
    'White': 4,
}

gender_mapping = {
    'Female': 0,
    'Male': 1,
}

native_country_mapping = {
    '?': 0,
    'Cambodia': 1,
    'Canada': 2,
    'China': 3,
    'Columbia': 4,
    'Cuba': 5,
    'Dominican-Republic': 6,
    'Ecuador': 7,
    'El-Salvador': 8,
    'England': 9,
    'France': 10,
    'Germany': 11,
    'Greece': 12,
    'Guatemala': 13,
    'Haiti': 14,
    'Holand-Netherlands': 15,
    'Honduras': 16,
    'Hong': 17,
    'Hungary': 18,
    'India': 19,
    'Iran': 20,
    'Ireland': 21,
    'Italy': 22,
    'Jamaica': 23,
    'Japan': 24,
    'Laos': 25,
    'Mexico': 26,
    'Nicaragua': 27,
    'Outlying-US(Guam-USVI-etc)': 28,
    'Peru': 29,
    'Philippines': 30,
    'Poland': 31,
    'Portugal': 32,
    'Puerto-Rico': 33,
    'Scotland': 34,
    'South': 35,
    'Taiwan': 36,
    'Thailand': 37,
    'Trinadad&Tobago': 38,
    'United-States': 39,
    'Vietnam': 40,
    'Yugoslavia': 41,
}


# Utility: safe lookup with fallback to '?' (or 0)
def encode(value: str, mapping: dict, colname: str):
    """
    Convert category string to its integer code using mapping.
    If not found, use '?' if available; else 0; and warn user.
    """
    if value in mapping:
        return mapping[value]
    fallback = mapping.get('?', 0)
    st.warning(f"Unseen value '{value}' for {colname}; using '{list(mapping.keys())[fallback] if isinstance(fallback,int) else '?'}' fallback.")
    return fallback


# ------------------------------------------------------------
# Collect user inputs
# ------------------------------------------------------------
st.subheader("Input Features")

age = st.number_input("Age", min_value=17, max_value=90, value=30)

workclass = st.selectbox("Workclass", options=list(workclass_mapping.keys()), index=4 if 'Private' in workclass_mapping else 0)

fnlwgt = st.number_input("Fnlwgt", min_value=0, value=200000, step=1000)

# NOTE: 'education' was NOT used in the trained model; only 'educational-num'
educational_num = st.slider("Educational Num (years of education rank)", min_value=1, max_value=16, value=10)

marital_status = st.selectbox("Marital Status", options=list(marital_status_mapping.keys()), index=2 if 'Married-civ-spouse' in marital_status_mapping else 0)

occupation = st.selectbox("Occupation", options=list(occupation_mapping.keys()), index=4 if 'Exec-managerial' in occupation_mapping else 0)

relationship = st.selectbox("Relationship", options=list(relationship_mapping.keys()), index=1 if 'Not-in-family' in relationship_mapping else 0)

race = st.selectbox("Race", options=list(race_mapping.keys()), index=4 if 'White' in race_mapping else 0)

gender = st.selectbox("Gender", options=list(gender_mapping.keys()), index=1 if 'Male' in gender_mapping else 0)

capital_gain = st.number_input("Capital Gain", min_value=0, value=0, step=100)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0, step=100)

hours_per_week = st.slider("Hours per Week", min_value=1, max_value=99, value=40)

native_country = st.selectbox("Native Country", options=list(native_country_mapping.keys()), index=39 if 'United-States' in native_country_mapping else 0)


# ------------------------------------------------------------
# Encode user selections -> numeric values
# ------------------------------------------------------------
encoded_row = {
    'age': age,
    'workclass': encode(workclass, workclass_mapping, 'workclass'),
    'fnlwgt': fnlwgt,
    'educational-num': educational_num,
    'marital-status': encode(marital_status, marital_status_mapping, 'marital-status'),
    'occupation': encode(occupation, occupation_mapping, 'occupation'),
    'relationship': encode(relationship, relationship_mapping, 'relationship'),
    'race': encode(race, race_mapping, 'race'),
    'gender': encode(gender, gender_mapping, 'gender'),
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': encode(native_country, native_country_mapping, 'native-country'),
}

input_df = pd.DataFrame([encoded_row])


# ------------------------------------------------------------
# Display encoded inputs (for debugging)
# ------------------------------------------------------------
with st.expander("Show encoded input row"):
    st.write(input_df)


# ------------------------------------------------------------
# Predict
# ------------------------------------------------------------
if st.button("Predict Salary"):
    try:
        pred = model.predict(input_df)[0]
        # Many Adult models use strings: '>50K', '<=50K' (or '<=50K')
        if isinstance(pred, str):
            label = pred
        else:
            # If model encoded y, decode to friendly text
            # Assumes 1 => >50K, 0 => <=50K (adjust if different)
            label = ">50K" if pred == 1 else "<=50K"

        if ">50K" in label:
            st.success("Predicted Salary: **>50K**")
        else:
            st.warning("Predicted Salary: **≤50K**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

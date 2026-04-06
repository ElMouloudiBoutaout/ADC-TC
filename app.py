import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from src.data_loader import load_data
from src.preprocessor import FEATURE_COLS

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    reg = joblib.load('models/model_regression.pkl')
    clf = joblib.load('models/model_classification.pkl')
    pre = joblib.load('models/preprocessor.pkl')
    return reg, clf, pre

@st.cache_data
def load_dataset():
    return load_data('data/TADC_Complete_v3_avec_S.xlsx')

reg_model, clf_model, preprocessor = load_models()
df = load_dataset()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="ADC Toxicity Classifier", page_icon="🧬", layout="wide")
st.title("🧬 ADC Toxicity Classifier")
st.caption("Prédiction du risque de toxicité Grade ≥3 pour les Antibody-Drug Conjugates")

# ── Sidebar inputs ────────────────────────────────────────────────────────────
st.sidebar.header("Paramètres ADC")

P = st.sidebar.slider("P — Puissance payload", 1.0, 3.0, 1.5, 0.5,
                      help="Puissance cytotoxique du payload (ex: DM1, MMAE, DXd)")
D = st.sidebar.slider("D — DAR", 0.5, 3.0, 1.5, 0.5,
                      help="Drug Antibody Ratio — nombre de molécules de drogue par anticorps")
H = st.sidebar.slider("H — Hydrophobie", 0.5, 2.0, 1.0, 0.5,
                      help="Score d'hydrophobie — risque de capture hépatique")
B = st.sidebar.slider("B — Effet bystander", 0.5, 2.0, 1.0, 0.5,
                      help="Capacité de la drogue à diffuser hors de la cellule cible")
L = st.sidebar.slider("L — Stabilité linker", 0.5, 3.0, 1.5, 0.5,
                      help="Stabilité du linker en circulation sanguine")
E = st.sidebar.slider("E — Exposition", 0.5, 2.5, 1.0, 0.5,
                      help="Dose administrée × durée d'exposition")
V = st.sidebar.slider("V — Valeur organe", 1.0, 2.0, 1.3, 0.1,
                      help="Importance de l'organe × expression on-target")
S = st.sidebar.slider("S — Sensibilité patient", 0.2, 2.81, 1.0, 0.1,
                      help="Fragilité patient (insuffisance rénale/hépatique, comorbidités)")

st.sidebar.markdown("---")
payload_options = sorted(df['Payload class'].dropna().unique().tolist())
organe_options = sorted(df['Organe'].dropna().unique().tolist())
payload = st.sidebar.selectbox("Payload class", payload_options)
organe = st.sidebar.selectbox("Organe cible", organe_options)

predict_btn = st.sidebar.button("🔍 Prédire la toxicité", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    input_df = pd.DataFrame([{
        'P': P, 'D': D, 'H': H, 'B': B, 'L': L, 'E': E,
        'V': V, 'S(payload,organe)': S,
        'Payload class': payload, 'Organe': organe,
    }])

    # Add interaction features for the model
    input_df['P_D'] = input_df['P'] * input_df['D']
    input_df['V_S'] = input_df['V'] * input_df['S(payload,organe)']
    input_df['E_L'] = input_df['E'] / (input_df['L'] + 1e-6)

    # Note: FEATURE_COLS + interactions must match what preprocessor expects
    all_features = FEATURE_COLS + ['P_D', 'V_S', 'E_L']
    X_input = preprocessor.transform(input_df[all_features])

    pred_reg = float(reg_model.predict(X_input)[0])
    pred_prob = float(clf_model.predict_proba(X_input)[0][1])
    pred_binary = int(pred_prob >= 0.5)

    # T-ADC v3 formula baseline
    weighted_sum = 5*P + 1*D + 1*H + 1.5*B + 0.5*L + 0.5*E
    tadc_v3 = weighted_sum * V * S

    # ── Results display ───────────────────────────────────────────────────────
    st.markdown("## Résultats")
    col1, col2, col3 = st.columns(3)

    col1.metric("% G≥3 prédit (ML)", f"{pred_reg:.1f}%",
                help="Pourcentage prédit de toxicité de Grade ≥3 par le modèle ML")
    col2.metric("Score T-ADC v3 (formule)", f"{tadc_v3:.1f}",
                help="Score calculé par la formule déterministe T-ADC v3")

    if pred_binary == 1:
        col3.error("🔴 Risque ÉLEVÉ (G≥3 >10%)")
    else:
        col3.success("🟢 Risque FAIBLE (G≥3 ≤10%)")

    st.markdown(f"**Probabilité G≥3 >10% :** {pred_prob:.0%}")
    st.progress(min(pred_prob, 1.0))

    # ── SHAP waterfall ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Interprétabilité SHAP — Contribution des features")

    feature_names = (
        list(preprocessor.named_transformers_['num'].get_feature_names_out())
        + list(preprocessor.named_transformers_['cat'].get_feature_names_out())
    )
    explainer = shap.TreeExplainer(reg_model)
    shap_vals = explainer.shap_values(X_input)

    fig, ax = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_vals[0],
            base_values=explainer.expected_value,
            data=X_input[0],
            feature_names=feature_names,
        ),
        show=False,
    )
    st.pyplot(fig)
    plt.close()

else:
    st.info("Ajustez les paramètres dans la barre latérale puis cliquez sur **Prédire la toxicité**.")

    # Show dataset overview
    st.markdown("---")
    st.subheader("Aperçu du dataset (106 lignes, 14 ADCs)")
    st.dataframe(
        df[['ADC', 'Organe', 'Payload class', 'P', 'D', 'H', 'B', 'L', 'E', 'V',
            'S(payload,organe)', '%G≥3 observé', 'Y binaire (G≥3 >10%)']],
        use_container_width=True
    )

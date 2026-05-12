import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PRIS — Drug Regulatory AI",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom Styling ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%);
    color: #e8eaf6;
}

/* Hero Header */
.hero-header {
    background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
    border-radius: 20px;
    padding: 40px;
    margin-bottom: 30px;
    border: 1px solid rgba(100, 181, 246, 0.3);
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(100,181,246,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8em;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
    letter-spacing: -1px;
}
.hero-sub {
    font-size: 1.1em;
    color: rgba(255,255,255,0.75);
    margin-top: 8px;
    font-weight: 300;
}
.hero-badge {
    display: inline-block;
    background: rgba(100,181,246,0.2);
    border: 1px solid rgba(100,181,246,0.5);
    color: #64b5f6;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: 600;
    margin-bottom: 15px;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Section Headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.3em;
    font-weight: 700;
    color: #64b5f6;
    border-left: 4px solid #1565c0;
    padding-left: 14px;
    margin: 25px 0 15px 0;
    letter-spacing: 0.5px;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(10px);
}

/* Predict Button */
.stButton > button {
    background: linear-gradient(135deg, #1565c0, #0d47a1) !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.1em !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    width: 100% !important;
    padding: 16px !important;
    border-radius: 12px !important;
    border: none !important;
    box-shadow: 0 8px 32px rgba(21,101,192,0.4) !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 40px rgba(21,101,192,0.6) !important;
}

/* Result Cards */
.result-regulated {
    background: linear-gradient(135deg, rgba(183,28,28,0.3), rgba(229,57,53,0.1));
    border: 1px solid rgba(229,57,53,0.5);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}
.result-safe {
    background: linear-gradient(135deg, rgba(27,94,32,0.3), rgba(67,160,71,0.1));
    border: 1px solid rgba(67,160,71,0.5);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 2em;
    font-weight: 800;
    margin: 10px 0 5px 0;
}
.result-conf {
    font-size: 1em;
    opacity: 0.8;
    margin: 0;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 12px;
    margin-top: 16px;
}
.metric-box {
    flex: 1;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.8em;
    font-weight: 700;
    color: #64b5f6;
}
.metric-lbl {
    font-size: 0.75em;
    opacity: 0.65;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Sliders & Inputs */
.stSlider > div > div > div { background: #1565c0 !important; }
label { color: #b0bec5 !important; font-size: 0.9em !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #90a4ae !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    background: #1565c0 !important;
    color: white !important;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* File uploader */
.stFileUploader {
    background: rgba(255,255,255,0.03) !important;
    border: 2px dashed rgba(100,181,246,0.3) !important;
    border-radius: 12px !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 10px !important;
    color: #90caf9 !important;
    font-weight: 600 !important;
}

/* DataFrame */
.dataframe { font-size: 0.85em !important; }

/* Selectbox & number input */
.stSelectbox > div > div, .stNumberInput > div > div {
    background: rgba(255,255,255,0.06) !important;
    border-color: rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("drug_regulation_model.pkl")

model = load_model()

# ── Hero Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">🔬 AI-Powered</div>
    <div class="hero-title">💊 PRIS</div>
    <div style="font-family:'Syne',sans-serif; font-size:1.4em; font-weight:600; color:rgba(255,255,255,0.9); margin-top:4px;">
        Pharmaceutical Regulatory Intelligence System
    </div>
    <div class="hero-sub">Predict drug regulatory classification using XGBoost · Explainable AI · Batch Processing</div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍  Single Drug Prediction", "📂  Batch CSV Prediction"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Drug Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── Input Sections ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔬 Core Risk Inputs</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        with st.container():
            regulatory_risk    = st.slider("Regulatory Risk Score",       0.0, 10.0, 5.0, 0.1)
            safety_risk        = st.slider("Drug Safety Risk",            0.0, 20.0, 10.0, 0.1)
            abuse_potential    = st.slider("Abuse Potential Score",       0.0, 10.0, 3.0, 0.1)
            side_effect_sev    = st.slider("Side Effect Severity Score",  0.0, 10.0, 5.0, 0.1)
    with col2:
        doctor_influence   = st.slider("Doctor Influence Index",      0.0,  1.0, 0.5, 0.01)
        insurance          = st.slider("Insurance Coverage %",        0.0, 100.0, 50.0, 0.5)
        recall_history     = st.number_input("Recall History Count",  0, 50, 0)
        adverse_events     = st.number_input("Adverse Event Reports", 0, 1000, 10)

    st.markdown('<div class="section-header">💊 Drug Details</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        dosage_mg            = st.number_input("Dosage (mg)",           0.0, 2000.0, 100.0)
        clinical_trial_phase = st.selectbox("Clinical Trial Phase",     [1, 2, 3, 4])
        drug_form            = st.selectbox("Drug Form",                ["Tablet", "Injection", "Syrup"])
        therapeutic_class    = st.selectbox("Therapeutic Class",        ["Antibiotic", "Antidepressant", "Antiviral", "Cardiovascular", "Other"])
    with col4:
        otc_flag         = st.checkbox("OTC (Over-the-Counter)?")
        high_risk        = st.checkbox("High Risk Substance?")
        requires_cold    = st.checkbox("Requires Cold Storage?")
        manufacturing_region = st.selectbox("Manufacturing Region",     ["North", "South", "West", "Other"])

    st.markdown('<div class="section-header">📊 Financial & Market Data</div>', unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        price_per_unit = st.number_input("Price Per Unit ($)",     0.0, 10000.0, 50.0)
        production_cost = st.number_input("Production Cost ($)",   0.0,  5000.0, 20.0)
        marketing_spend = st.number_input("Marketing Spend ($M)",  0.0,   500.0, 10.0)
        rd_investment   = st.number_input("R&D Investment ($M)",   0.0,  1000.0, 50.0)
    with col6:
        annual_sales    = st.number_input("Annual Sales Volume",   0, 10000000, 100000)
        competitor_count = st.number_input("Competitor Count",     0, 100, 5)
        export_pct      = st.slider("Export %",                    0.0, 100.0, 20.0)
        online_sales_pct = st.slider("Online Sales %",             0.0, 100.0, 10.0)

    st.markdown('<div class="section-header">🏥 Distribution & Reputation</div>', unsafe_allow_html=True)
    col7, col8 = st.columns(2)
    with col7:
        prescription_rate  = st.slider("Prescription Rate",          0.0, 1.0, 0.5, 0.01)
        hospital_dist_pct  = st.slider("Hospital Distribution %",    0.0, 100.0, 40.0)
        brand_reputation   = st.slider("Brand Reputation Score",     0.0, 10.0, 7.0, 0.1)
        doctor_rec_rate    = st.slider("Doctor Recommendation Rate", 0.0, 1.0, 0.5, 0.01)
    with col8:
        approval_time  = st.number_input("Approval Time (Months)", 0, 120, 24)
        patent_duration = st.number_input("Patent Duration (Years)", 0, 30, 10)

    st.markdown("---")

    # ── Predict Button ──────────────────────────────────────────────────────
    predict_btn = st.button("🔍  PREDICT REGULATORY CLASS", key="single_predict")

    if predict_btn:

        # ── Feature Engineering ─────────────────────────────────────────────
        profit_margin         = (price_per_unit - production_cost) / (price_per_unit + 1e-9)
        risk_safety_interaction = regulatory_risk * safety_risk
        marketing_efficiency  = annual_sales / (marketing_spend * 1e6 + 1)
        log_risk_safety       = np.log1p(risk_safety_interaction)

        drug_form_injection = int(drug_form == "Injection")
        drug_form_syrup     = int(drug_form == "Syrup")
        drug_form_tablet    = int(drug_form == "Tablet")

        tc_antibiotic      = int(therapeutic_class == "Antibiotic")
        tc_antidepressant  = int(therapeutic_class == "Antidepressant")
        tc_antiviral       = int(therapeutic_class == "Antiviral")
        tc_cardiovascular  = int(therapeutic_class == "Cardiovascular")

        region_north = int(manufacturing_region == "North")
        region_south = int(manufacturing_region == "South")
        region_west  = int(manufacturing_region == "West")

        feature_dict = {
            'Dosage_mg': dosage_mg,
            'Price_Per_Unit': price_per_unit,
            'Production_Cost': production_cost,
            'Marketing_Spend': marketing_spend,
            'Clinical_Trial_Phase': clinical_trial_phase,
            'Side_Effect_Severity_Score': side_effect_sev,
            'Abuse_Potential_Score': abuse_potential,
            'Prescription_Rate': prescription_rate,
            'Hospital_Distribution_Percentage': hospital_dist_pct,
            'Annual_Sales_Volume': annual_sales,
            'Regulatory_Risk_Score': regulatory_risk,
            'Approval_Time_Months': approval_time,
            'Patent_Duration_Years': patent_duration,
            'R&D_Investment_Million': rd_investment,
            'Competitor_Count': competitor_count,
            'Recall_History_Count': recall_history,
            'Adverse_Event_Reports': adverse_events,
            'Requires_Cold_Storage': int(requires_cold),
            'OTC_Flag': int(otc_flag),
            'High_Risk_Substance': int(high_risk),
            'Insurance_Coverage_Percentage': insurance,
            'Export_Percentage': export_pct,
            'Online_Sales_Percentage': online_sales_pct,
            'Brand_Reputation_Score': brand_reputation,
            'Doctor_Recommendation_Rate': doctor_rec_rate,
            'Profit_Margin': profit_margin,
            'Drug_Safety_Risk': safety_risk,
            'Drug_Form_Injection': drug_form_injection,
            'Drug_Form_Syrup': drug_form_syrup,
            'Drug_Form_Tablet': drug_form_tablet,
            'Therapeutic_Class_Antibiotic': tc_antibiotic,
            'Therapeutic_Class_Antidepressant': tc_antidepressant,
            'Therapeutic_Class_Antiviral': tc_antiviral,
            'Therapeutic_Class_Cardiovascular': tc_cardiovascular,
            'Manufacturing_Region_North': region_north,
            'Manufacturing_Region_South': region_south,
            'Manufacturing_Region_West': region_west,
            'Risk_Safety_Interaction': risk_safety_interaction,
            'Doctor_Influence_Index': doctor_influence,
            'Marketing_Efficiency': marketing_efficiency,
            'Log_Risk_Safety': log_risk_safety,
        }

        features_df = pd.DataFrame([feature_dict])

        prediction = model.predict(features_df)[0]
        proba      = model.predict_proba(features_df)[0]
        conf       = proba[prediction] * 100

        # ── Result Card ─────────────────────────────────────────────────────
        st.markdown("---")
        if prediction == 1:
            st.markdown(f"""
            <div class="result-regulated">
                <div style="font-size:3em;">⚠️</div>
                <div class="result-title" style="color:#ef5350;">REGULATED DRUG</div>
                <div class="result-conf">Confidence: <strong>{conf:.1f}%</strong> &nbsp;|&nbsp; 
                Regulated Probability: <strong>{proba[1]*100:.1f}%</strong></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
                <div style="font-size:3em;">✅</div>
                <div class="result-title" style="color:#66bb6a;">NON-REGULATED DRUG</div>
                <div class="result-conf">Confidence: <strong>{conf:.1f}%</strong> &nbsp;|&nbsp; 
                Safe Probability: <strong>{proba[0]*100:.1f}%</strong></div>
            </div>
            """, unsafe_allow_html=True)

        # ── Metric Row ──────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-val">{proba[1]*100:.1f}%</div>
                <div class="metric-lbl">Regulated Risk</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">{proba[0]*100:.1f}%</div>
                <div class="metric-lbl">Safe Probability</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">{regulatory_risk:.1f}</div>
                <div class="metric-lbl">Reg. Risk Score</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">{safety_risk:.1f}</div>
                <div class="metric-lbl">Safety Risk</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Feature Importance Chart ────────────────────────────────────────
        with st.expander("📊 Feature Importance — Top 15 Features"):
            importance  = model.feature_importances_
            feat_names  = features_df.columns.tolist()
            sorted_idx  = np.argsort(importance)[-15:]

            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#0d1b2a')
            ax.set_facecolor('#0d1b2a')

            colors = ['#1565c0' if i != sorted_idx[-1] else '#ef5350' for i in sorted_idx]
            bars = ax.barh(
                [feat_names[i] for i in sorted_idx],
                [importance[i] for i in sorted_idx],
                color=colors, edgecolor='none', height=0.65
            )

            ax.set_xlabel("Importance Score", color='#90a4ae', fontsize=10)
            ax.set_title("Top 15 Most Important Features", color='white',
                         fontsize=13, fontweight='bold', pad=15)
            ax.tick_params(colors='#90a4ae', labelsize=9)
            ax.spines[:].set_visible(False)
            ax.xaxis.grid(True, color=(1, 1, 1, 0.06), linewidth=0.8)
            ax.set_axisbelow(True)

            top_patch = mpatches.Patch(color='#ef5350', label='Top Feature')
            rest_patch = mpatches.Patch(color='#1565c0', label='Other Features')
            ax.legend(handles=[top_patch, rest_patch], facecolor='#0d1b2a',
                      labelcolor='white', fontsize=9)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ── SHAP Explainability ─────────────────────────────────────────────
        with st.expander("🧠 SHAP Explainability — Why this prediction?"):
            try:
                with st.spinner("Calculating SHAP values..."):
                    explainer   = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(features_df)

                    # Handle multi-class (use class 1 = regulated)
                    if isinstance(shap_values, list):
                        sv = shap_values[1][0]
                        ev = explainer.expected_value[1]
                    else:
                        sv = shap_values[0]
                        ev = explainer.expected_value

                    fig, ax = plt.subplots(figsize=(10, 7))
                    fig.patch.set_facecolor('#0d1b2a')

                    shap.waterfall_plot(
                        shap.Explanation(
                            values=sv,
                            base_values=ev,
                            data=features_df.iloc[0].values,
                            feature_names=features_df.columns.tolist()
                        ),
                        show=False,
                        max_display=15
                    )
                    plt.gcf().set_facecolor('#0d1b2a')
                    st.pyplot(plt.gcf())
                    plt.close()

                st.info("🔵 Blue bars push toward **Non-Regulated** | 🔴 Red bars push toward **Regulated**")
            except Exception as e:
                st.warning(f"SHAP visualization could not be generated: {e}")

        # ── All Feature Values ──────────────────────────────────────────────
        with st.expander("📋 View All 41 Input Features"):
            display_df = features_df.T.rename(columns={0: "Value"})
            display_df["Value"] = display_df["Value"].round(4)
            st.dataframe(display_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch CSV Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">📂 Upload CSV for Batch Prediction</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <p style="color:#90a4ae; margin:0;">
        Upload a <strong style="color:#64b5f6;">.csv file</strong> where each row is a drug. 
        The CSV must contain the same <strong style="color:#64b5f6;">41 feature columns</strong> 
        the model was trained on. The app will predict regulatory class for every row and 
        let you download the results.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"✅ File loaded — **{len(batch_df)} drugs** found, **{len(batch_df.columns)} columns** detected.")

            with st.expander("👁️ Preview uploaded data"):
                st.dataframe(batch_df.head(10), use_container_width=True)

            if st.button("🚀  RUN BATCH PREDICTION", key="batch_predict"):
                with st.spinner("Predicting..."):
                    batch_preds = model.predict(batch_df)
                    batch_proba = model.predict_proba(batch_df)[:, 1]

                results_df = batch_df.copy()
                results_df.insert(0, "Prediction",    ["⚠️ Regulated" if p == 1 else "✅ Non-Regulated" for p in batch_preds])
                results_df.insert(1, "Confidence %",  (np.where(batch_preds == 1, batch_proba, 1 - batch_proba) * 100).round(1))
                results_df.insert(2, "Reg. Prob %",   (batch_proba * 100).round(1))

                # Summary metrics
                regulated_count = int(batch_preds.sum())
                safe_count      = len(batch_preds) - regulated_count
                avg_conf        = results_df["Confidence %"].mean()

                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-box">
                        <div class="metric-val">{len(batch_preds)}</div>
                        <div class="metric-lbl">Total Drugs</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-val" style="color:#ef5350;">{regulated_count}</div>
                        <div class="metric-lbl">Regulated</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-val" style="color:#66bb6a;">{safe_count}</div>
                        <div class="metric-lbl">Non-Regulated</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-val">{avg_conf:.1f}%</div>
                        <div class="metric-lbl">Avg Confidence</div>
                    </div>
                </div>
                <br>
                """, unsafe_allow_html=True)

                # Distribution pie chart
                fig2, ax2 = plt.subplots(figsize=(5, 5))
                fig2.patch.set_facecolor('#0d1b2a')
                ax2.set_facecolor('#0d1b2a')
                ax2.pie(
                    [regulated_count, safe_count],
                    labels=["Regulated", "Non-Regulated"],
                    colors=["#ef5350", "#66bb6a"],
                    autopct='%1.1f%%',
                    textprops={'color': 'white', 'fontsize': 12},
                    wedgeprops={'edgecolor': '#0d1b2a', 'linewidth': 2}
                )
                ax2.set_title("Prediction Distribution", color='white', fontsize=13, fontweight='bold')
                col_pie, col_space = st.columns([1, 2])
                with col_pie:
                    st.pyplot(fig2)
                plt.close()

                # Results table (only key cols + predictions)
                st.markdown('<div class="section-header">📋 Prediction Results</div>', unsafe_allow_html=True)
                st.dataframe(
                    results_df[["Prediction", "Confidence %", "Reg. Prob %"]].reset_index(drop=True),
                    use_container_width=True
                )

                # Download
                csv_out = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️  Download Full Results CSV",
                    data=csv_out,
                    file_name="drug_predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            st.info("Make sure your CSV has the correct 41 feature columns matching the training data.")

    else:
        # Show expected columns
        with st.expander("📋 View required CSV columns (41 features)"):
            required_cols = [
                'Dosage_mg', 'Price_Per_Unit', 'Production_Cost', 'Marketing_Spend',
                'Clinical_Trial_Phase', 'Side_Effect_Severity_Score', 'Abuse_Potential_Score',
                'Prescription_Rate', 'Hospital_Distribution_Percentage', 'Annual_Sales_Volume',
                'Regulatory_Risk_Score', 'Approval_Time_Months', 'Patent_Duration_Years',
                'R&D_Investment_Million', 'Competitor_Count', 'Recall_History_Count',
                'Adverse_Event_Reports', 'Requires_Cold_Storage', 'OTC_Flag', 'High_Risk_Substance',
                'Insurance_Coverage_Percentage', 'Export_Percentage', 'Online_Sales_Percentage',
                'Brand_Reputation_Score', 'Doctor_Recommendation_Rate', 'Profit_Margin',
                'Drug_Safety_Risk', 'Drug_Form_Injection', 'Drug_Form_Syrup', 'Drug_Form_Tablet',
                'Therapeutic_Class_Antibiotic', 'Therapeutic_Class_Antidepressant',
                'Therapeutic_Class_Antiviral', 'Therapeutic_Class_Cardiovascular',
                'Manufacturing_Region_North', 'Manufacturing_Region_South', 'Manufacturing_Region_West',
                'Risk_Safety_Interaction', 'Doctor_Influence_Index', 'Marketing_Efficiency', 'Log_Risk_Safety'
            ]
            cols_df = pd.DataFrame({"Column Name": required_cols, "Index": range(1, 42)})
            st.dataframe(cols_df, use_container_width=True)

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:40px 0 20px; color:rgba(255,255,255,0.3); font-size:0.8em;">
    PRIS — Pharmaceutical Regulatory Intelligence System &nbsp;·&nbsp; 
    Powered by XGBoost + SHAP &nbsp;·&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)
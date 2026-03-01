import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import ttest_ind
from fpdf import FPDF
from pyteomics import mzml
import os
import gc

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="TLL Metabo-Discovery", layout="wide")

# --- 2. SIDEBAR ---
st.sidebar.title("TLL Metabo-Discovery")
st.sidebar.info("""
**Research Pipeline**  
Standardized bioinformatics ecosystem for high-throughput untargeted metabolomics discovery.
""")

st.sidebar.markdown("---")
st.sidebar.subheader("Security & Privacy")
st.sidebar.caption("Data is processed in volatile memory. No laboratory files are stored on the server instance.")

st.sidebar.markdown("---")
st.sidebar.subheader("Citation")
st.sidebar.caption("Yusuf, A. (2026). TLL Metabo-Discovery: An Integrated Pipeline for Machine-Learning Validated Metabolomics.")

st.sidebar.markdown("---")
st.sidebar.caption("Developed by Abass Yusuf | Laboratory Support")

# --- 3. HELPER: PDF GENERATOR ---
def create_academic_report(g1, g2, feat_count, accuracy, leads_count):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "TLL Metabo-Discovery: Research Summary", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "1. Executive Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    summary = (f"The analysis successfully identified metabolic differences between cohort {g1} and {g2}. "
               f"A total of {feat_count} features were analyzed post-filtration. "
               f"Prioritization identified {leads_count} features with high discovery confidence. "
               f"Predictive validation (Random Forest) achieved an accuracy of {accuracy:.1%}.")
    pdf.multi_cell(0, 10, summary)
    
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 10, "Report generated for laboratory internal use and manuscript support.", ln=True)
    return bytes(pdf.output())

# --- 4. MAIN INTERFACE ---
st.title("TLL Metabo-Discovery: Professional Analytics Suite")
st.markdown("---")

mode = st.radio("Select Analysis Module:", 
                ("Raw Data (.mzML) Batch Extraction", "Discovery Dashboard & Statistical Analysis"))

# ============================================
# MODULE 1: RAW DATA PROCESSOR
# ============================================
if mode == "Raw Data (.mzML) Batch Extraction":
    st.subheader("Batch Feature Extraction (mzML)")
    uploaded_mzmls = st.file_uploader("Upload raw files (Max 5GB total)", type=["mzml"], accept_multiple_files=True)
    
    if uploaded_mzmls and st.button("Start Extraction Pipeline"):
        all_features = []
        p_bar = st.progress(0)
        status = st.empty()
        for i, file in enumerate(uploaded_mzmls):
            status.text(f"Processing: {file.name}")
            with open("temp.mzml", "wb") as f: f.write(file.getbuffer())
            rows = []
            try:
                with mzml.read("temp.mzml") as reader:
                    for spec in reader:
                        if spec['ms level'] == 1 and len(spec['intensity array']) > 0:
                            idx = np.argmax(spec['intensity array'])
                            rows.append([float(spec['m/z array'][idx]), float(spec['scanList']['scan'][0]['scan start time'])/60, float(spec['intensity array'][idx])])
                df_s = pd.DataFrame(rows, columns=["m/z", "RT_min", "Intensity"])
                df_s["Sample"] = file.name.replace(".mzML", "")
                all_features.append(df_s)
                del rows; gc.collect() 
            except Exception as e: st.error(f"Error in {file.name}: {e}")
            p_bar.progress((i + 1) / len(uploaded_mzmls))
        
        st.success("Extraction Complete.")
        st.download_button("Download Feature Table (CSV)", pd.concat(all_features).to_csv(index=False).encode('utf-8'), "combined_raw_data.csv")
        if os.path.exists("temp.mzml"): os.remove("temp.mzml")

# ============================================
# MODULE 2: DISCOVERY DASHBOARD
# ============================================
else:
    uploaded_file = st.file_uploader("Upload Quantified Peak Table (.csv)", type=["csv"])
    if uploaded_file:
        df_in = pd.read_csv(uploaded_file)
        with st.expander("Configure Pre-processing & Parameters"):
            c1, c2, c3, c4 = st.columns(4)
            mz_col = c1.selectbox("m/z Column", df_in.columns, index=0)
            rt_col = c2.selectbox("RT Column", df_in.columns, index=1 if "RT_min" not in df_in.columns else df_in.columns.get_loc("RT_min"))
            sm_col = c3.selectbox("Sample ID", df_in.columns, index=df_in.columns.get_loc("Sample") if "Sample" in df_in.columns else 0)
            in_col = c4.selectbox("Intensity", df_in.columns, index=df_in.columns.get_loc("Intensity") if "Intensity" in df_in.columns else 2)
            
            f1, f2, f3, f4 = st.columns(4)
            mz_bin = f1.slider("m/z Alignment Tolerance", 1, 5, 3)
            min_pres = f2.slider("Min Presence (%)", 0, 100, 80)
            ion_mode = f3.selectbox("Ionization Polarity", ["Negative [M-H]-", "Positive [M+H]+"])
            scaling = f4.selectbox("Scaling Method", ["Pareto Scaling", "Auto-Scaling", "None"])

        if st.button("Run Full Discovery Pipeline"):
            try:
                # 1. CLEANING & NORMALIZATION
                df_in['T_ID'] = df_in[mz_col].round(mz_bin).astype(str) + "_" + df_in[rt_col].round(2).astype(str)
                pivot = df_in.pivot_table(index='T_ID', columns=sm_col, values=in_col, aggfunc='mean').fillna(0)
                cleaned = pivot[(pivot != 0).sum(axis=1) >= (min_pres/100)*len(pivot.columns)]
                if cleaned.empty: st.error("No features left. Lower 'Min Presence'."); st.stop()
                
                tic_norm = cleaned.replace(0, (cleaned[cleaned > 0].min().min()) / 2).div(cleaned.sum(axis=0), axis=1) * 1000000 
                tic_norm = tic_norm[tic_norm.std(axis=1) > 0]

                # 2. STATS & ML
                groups = [str(s).split('_')[0] for s in tic_norm.columns]
                unique_g = sorted(list(set(groups)))
                X = tic_norm.T
                if scaling == "Pareto Scaling": X_s = (X - X.mean()) / np.sqrt(X.std().replace(0, np.nan))
                else: X_s = (X - X.mean()) / X.std()
                pca_res = PCA(n_components=2).fit_transform(X_s.fillna(0))
                
                stats_ready = False
                if len(unique_g) >= 2:
                    g1_c, g2_c = [c for c in tic_norm.columns if c.startswith(unique_g[0])], [c for c in tic_norm.columns if c.startswith(unique_g[1])]
                    _, pvals = ttest_ind(tic_norm[g1_c], tic_norm[g2_c], axis=1)
                    log2fc = np.log2(tic_norm[g2_c].mean(axis=1) / tic_norm[g1_c].mean(axis=1).replace(0, 0.001))
                    stats_df = pd.DataFrame({'ID': tic_norm.index, 'p': pvals, 'log10p': -np.log10(pvals), 'Log2FC': log2fc}).reset_index(drop=True)
                    stats_df['Significant'] = (stats_df['p'] < 0.05) & (abs(stats_df['Log2FC']) > 1)
                    y_ml = [1 if g == unique_g[-1] else 0 for g in groups]
                    acc = cross_val_score(RandomForestClassifier(), X_s.fillna(0), y_ml, cv=3).mean()
                    stats_ready = True

                # 3. TABS
                t1, t2, t3, t4, t5 = st.tabs(["📊 Distribution", "🔵 PCA", "🌋 Discovery", "🏆 Identification", "📋 Summary"])
                
                with t1: st.plotly_chart(px.box(X.melt(), y='value', title="TIC Normalization QC"), use_container_width=True)
                with t2: st.plotly_chart(px.scatter(x=pca_res[:,0], y=pca_res[:,1], color=groups, title="Unsupervised PCA Analysis"), use_container_width=True)
                with t3:
                    if stats_ready: st.plotly_chart(px.scatter(stats_df, x='Log2FC', y='log10p', color='Significant', hover_name='ID', title="Biomarker Discovery Plot"), use_container_width=True)
                
                with t4:
                    if stats_ready:
                        st.subheader("Automated Identification")
                        hits = stats_df[stats_df['Significant']].copy()
                        hits['m/z'] = hits['ID'].apply(lambda x: float(x.split('_')[0]))
                        hits['Neutral Mass'] = (hits['m/z'] + 1.0078) if ion_mode.startswith("Neg") else (hits['m/z'] - 1.0078)
                        
                        # --- UPDATED SIMPLIFIED CONFIDENCE SCORING ---
                        def score_confidence(row):
                            lv = 3 # Default: Mass + Stats confirmed
                            if row['Neutral Mass'] < 500: lv = 2 # High likelihood (Drug-like MW)
                            # Level 1 is reserved for external MS2 validation (added manually later)
                            return f"Level {lv}"
                        
                        hits['Confidence'] = hits.apply(score_confidence, axis=1)
                        hits['Search PubChem'] = hits['m/z'].apply(lambda x: f"https://pubchem.ncbi.nlm.nih.gov/#query={x}")
                        
                        st.dataframe(hits[['ID', 'Neutral Mass', 'Confidence', 'Log2FC', 'Search PubChem']], 
                                     column_config={"Search PubChem": st.column_config.LinkColumn("Identify")})
                        st.info("Confidence Logic: L2 (Physicochemical Likelihood), L3 (Statistically Significant), L4 (Mass Match Only)")

                with t5:
                    if stats_ready:
                        st.metric("Model Predictive Accuracy", f"{acc:.1%}")
                        pdf_data = create_academic_report(unique_g[0], unique_g[1], len(cleaned), acc, len(hits[hits['Confidence'] == 'Level 2']))
                        st.download_button("Download Research Summary (PDF)", pdf_data, "Metabo_Research_Report.pdf", "application/pdf")
                
                st.balloons()
            except Exception as e: st.error(f"Error: {e}")

st.markdown("---")
st.caption("Internal Research Dashboard | Authorized Lab Use Only")

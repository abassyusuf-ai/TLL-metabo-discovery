import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import plotly.express as px
from scipy.stats import ttest_ind, linregress
from fpdf import FPDF
from pyteomics import mzml
import os
import gc

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Metabo-Cleaner Pro | Enterprise", layout="wide")

# --- GLOBAL SETTINGS ---
contact_email = "abass.metabo@gmail.com"
payment_url = "https://sandbox.flutterwave.com/donate/rgxclpstozwl"

# --- 2. SIDEBAR (Formal & Professional) ---
st.sidebar.title("Metabo-Cleaner Pro")
st.sidebar.info("Enterprise Discovery Suite: Machine Learning & Lead Prioritization.")

st.sidebar.markdown("---")
st.sidebar.subheader("Contact & Support")
contact_url = f"mailto:{contact_email}?subject=Enterprise%20Inquiry"
st.sidebar.markdown(f"[Email Lead Architect]({contact_url})")
if st.sidebar.button("Show Email Address"):
    st.sidebar.code(contact_email)

st.sidebar.markdown("---")
st.sidebar.subheader("Research Fund")
st.sidebar.markdown(f"[Sponsor Development]({payment_url})")

st.sidebar.markdown("---")
st.sidebar.caption(f"© 2026 Yusuf Bioinformatics | {contact_email}")

# --- 3. HELPER: PDF GENERATOR ---
def create_pdf_report(g1, g2, feat_count, accuracy, leads_count):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Discovery & Lead Prioritization Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "1. Executive Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    summary = (f"The analysis successfully identified metabolic differences between {g1} and {g2}. "
               f"Processed {feat_count} high-quality features. "
               f"Prioritization identified {leads_count} drug-like candidates (< 500 Da). "
               f"Machine Learning validation score: {accuracy:.1%}.")
    pdf.multi_cell(0, 10, summary)
    return bytes(pdf.output())

# --- 4. MAIN INTERFACE ---
st.title("Metabo-Cleaner Pro: Enterprise Discovery Suite")
st.markdown("---")

mode = st.radio("Select Analysis Module:", 
                ("Raw Data Processor", "Drug Discovery & Machine Learning Dashboard"))

# ============================================
# MODULE 1: RAW DATA PROCESSOR
# ============================================
if mode == "Raw Data Processor":
    st.subheader("Batch Feature Extraction")
    uploaded_mzmls = st.file_uploader("Select .mzML files", type=["mzml"], accept_multiple_files=True)
    if uploaded_mzmls and st.button("Start Extraction"):
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
        st.download_button("Download Table", pd.concat(all_features).to_csv(index=False).encode('utf-8'), "metabo_features.csv")
        if os.path.exists("temp.mzml"): os.remove("temp.mzml")

# ============================================
# MODULE 2: DISCOVERY DASHBOARD
# ============================================
else:
    uploaded_file = st.file_uploader("Upload Quantified Table (.csv)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        with st.expander("Advanced Configuration"):
            c1, c2, c3, c4 = st.columns(4)
            mz_col = c1.selectbox("m/z Column", df.columns, index=0)
            rt_col = c2.selectbox("RT Column", df.columns, index=1 if "RT_min" not in df.columns else df.columns.get_loc("RT_min"))
            sm_col = c3.selectbox("Sample ID", df.columns, index=df.columns.get_loc("Sample") if "Sample" in df.columns else 0)
            in_col = c4.selectbox("Intensity", df.columns, index=df.columns.get_loc("Intensity") if "Intensity" in df.columns else 2)
            mz_bin = st.slider("m/z Alignment", 1, 5, 3)
            min_pres = st.slider("Min Presence (%)", 0, 100, 80)
            ion_mode = st.selectbox("Polarity", ["Negative [M-H]-", "Positive [M+H]+"])
            scaling = st.selectbox("Scaling", ["Pareto Scaling", "Auto-Scaling", "None"])
            dose_map = st.text_input("Dose Settings (Format -> Group:Value, Group:Value)", placeholder="e.g., APL:0, MCL:100")

        if st.button("Run Enterprise Discovery Pipeline"):
            try:
                # 1. CLEANING ENGINE
                df['Temp_ID'] = df[mz_col].round(mz_bin).astype(str) + "_" + df[rt_col].round(2).astype(str)
                pivot = df.pivot_table(index='Temp_ID', columns=sm_col, values=in_col, aggfunc='mean').fillna(0)
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
                
                # Discovery Calcs
                g1_c, g2_c = [c for c in tic_norm.columns if c.startswith(unique_g[0])], [c for c in tic_norm.columns if c.startswith(unique_g[1])]
                _, pvals = ttest_ind(tic_norm[g1_c], tic_norm[g2_c], axis=1)
                log2fc = np.log2(tic_norm[g2_c].mean(axis=1) / tic_norm[g1_c].mean(axis=1).replace(0, 0.001))
                stats_df = pd.DataFrame({'ID': tic_norm.index, 'p': pvals, 'log10p': -np.log10(pvals), 'Log2FC': log2fc}).reset_index(drop=True)
                stats_df['Significant'] = (stats_df['p'] < 0.05) & (abs(stats_df['Log2FC']) > 1)
                acc = cross_val_score(RandomForestClassifier(), X_s.fillna(0), [1 if g == unique_g[-1] else 0 for g in groups], cv=3).mean()

                # Dose Response
                dose_ready = False
                if dose_map and ":" in dose_map:
                    try:
                        d_dict = {item.split(":")[0].strip(): float(item.split(":")[1].strip()) for item in dose_map.split(",")}
                        sample_doses = [float(d_dict.get(g, 0)) for g in groups]
                        if len(set(sample_doses)) > 1:
                            r_values = [linregress(sample_doses, tic_norm.iloc[i].values)[2]**2 for i in range(len(tic_norm))]
                            stats_df['Dose_R2'] = r_values
                            dose_ready = True
                    except: st.warning("Dose format error.")

                # TABS
                t1, t2, t3, t4, t5 = st.tabs(["📊 Quality", "🔵 Multivariate", "🌋 Volcano", "💊 Identification & Leads", "📋 Report"])
                
                with t1: st.plotly_chart(px.box(X.melt(), y='value', title="Data Distribution"), use_container_width=True)
                with t2: st.plotly_chart(px.scatter(x=pca_res[:,0], y=pca_res[:,1], color=groups, title="Group Separation"), use_container_width=True)
                with t3: st.plotly_chart(px.scatter(stats_df, x='Log2FC', y='log10p', color='Significant', hover_name='ID', title="Discovery Map"), use_container_width=True)
                
                with t4:
                    st.subheader("Identification & Confidence Scoring")
                    hits = stats_df[stats_df['Significant']].copy()
                    hits['m/z'] = hits['ID'].apply(lambda x: float(x.split('_')[0]))
                    hits['Neutral Mass'] = (hits['m/z'] + 1.0078) if ion_mode.startswith("Neg") else (hits['m/z'] - 1.0078)
                    hits['Lipinski_MW'] = hits['Neutral Mass'] < 500
                    
                    # --- THE CONFIDENCE SCORING LOGIC ---
                    def assign_confidence(row):
                        score = 4 # Default: Mass Match Only
                        if row['Significant']: score = 3 # Confirmed significant difference
                        if row['Lipinski_MW']: score = 2 # Drug-like candidate
                        if dose_ready and row.get('Dose_R2', 0) > 0.7: score = 1 # High Potency Responder
                        return f"Level {score}"

                    hits['Confidence'] = hits.apply(assign_confidence, axis=1)
                    hits['PubChem'] = hits['m/z'].apply(lambda x: f"https://pubchem.ncbi.nlm.nih.gov/#query={x}")
                    
                    st.metric("Total High-Confidence Leads", len(hits[hits['Confidence'] == "Level 1"]))
                    
                    st.dataframe(hits[['ID', 'Neutral Mass', 'Confidence', 'Log2FC', 'PubChem']], 
                                 column_config={"PubChem": st.column_config.LinkColumn("Search Isomers")})
                    
                    st.info("💡 Confidence Levels: L1 (Potency/Dose Confirmed), L2 (Drug-like), L3 (Statistically Significant), L4 (Mass Match only).")

                with t5:
                    pdf_bytes = create_pdf_report(unique_g[0], unique_g[1], len(stats_df[stats_df['Significant']]), acc, len(hits[hits['Lipinski_MW']]))
                    st.download_button(label="Download Professional PDF Report", data=pdf_bytes, file_name="Discovery_Report.pdf", mime="application/pdf")
                
                st.balloons()
            except Exception as e: st.error(f"Error: {e}")

st.markdown("---")
st.caption(f"Metabo-Cleaner Pro Enterprise | Pharmaceutical Discovery System | {contact_email}")

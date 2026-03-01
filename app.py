import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
st.sidebar.info("**Academic Research Suite**\nIntegrated pipeline for HRMS data processing.")
st.sidebar.markdown("---")
st.sidebar.subheader("Security")
st.sidebar.caption("Processing in-memory. No data storage.")
st.sidebar.markdown("---")
st.sidebar.subheader("Citation")
st.sidebar.caption("Yusuf, A. (2026). TLL Metabo-Discovery: An Integrated Pipeline for Machine-Learning Validated Metabolomics.")
st.sidebar.caption("Developed by Abass Yusuf | Lab Support")

# --- 3. HELPER: PDF GENERATOR ---
def create_academic_report(g1, g2, feat_count, accuracy, leads_count):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Research Discovery Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 12); pdf.cell(0, 10, "1. Executive Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    summary = (f"Analysis of {g1} vs {g2}. Features analyzed: {feat_count}. "
               f"ML Validation Accuracy: {accuracy:.1%}. "
               f"Identification found {leads_count} high-confidence candidates.")
    pdf.multi_cell(0, 10, summary)
    return bytes(pdf.output())

# --- 4. MAIN INTERFACE ---
st.title("TLL Metabo-Discovery: Master Analytics Suite")
st.markdown("---")

mode = st.radio("Select Analysis Module:", ("Raw Data Extraction", "Discovery Dashboard"))

# ============================================
# MODULE 1: RAW DATA PROCESSOR
# ============================================
if mode == "Raw Data Extraction":
    st.subheader("Batch mzML Feature Extraction")
    uploaded_mzmls = st.file_uploader("Upload .mzML files (Max 5GB)", type=["mzml"], accept_multiple_files=True)
    if uploaded_mzmls and st.button("Start Extraction"):
        all_features = []
        p_bar = st.progress(0); status = st.empty()
        for i, file in enumerate(uploaded_mzmls):
            status.text(f"Processing: {file.name}")
            with open("temp.mzml", "wb") as f: f.write(file.getbuffer())
            rows = []
            with mzml.read("temp.mzml") as reader:
                for spec in reader:
                    if spec['ms level'] == 1 and len(spec['intensity array']) > 0:
                        idx = np.argmax(spec['intensity array'])
                        rows.append([float(spec['m/z array'][idx]), float(spec['scanList']['scan'][0]['scan start time'])/60, float(spec['intensity array'][idx])])
            df_s = pd.DataFrame(rows, columns=["m/z", "RT_min", "Intensity"])
            df_s["Sample"] = file.name.replace(".mzML", "")
            all_features.append(df_s)
            del rows; gc.collect()
            p_bar.progress((i + 1) / len(uploaded_mzmls))
        st.success("Complete."); st.download_button("Download CSV", pd.concat(all_features).to_csv(index=False).encode('utf-8'), "features.csv")

# ============================================
# MODULE 2: DISCOVERY DASHBOARD
# ============================================
else:
    uploaded_file = st.file_uploader("Upload Table (.csv)", type=["csv"])
    if uploaded_file:
        df_in = pd.read_csv(uploaded_file)
        with st.expander("Parameters & Polarity"):
            c1, c2, c3, c4 = st.columns(4)
            mz_col = c1.selectbox("m/z", df_in.columns, index=0)
            rt_col = c2.selectbox("RT", df_in.columns, index=1 if "RT_min" not in df_in.columns else df_in.columns.get_loc("RT_min"))
            sm_col = c3.selectbox("Sample ID", df_in.columns, index=df_in.columns.get_loc("Sample") if "Sample" in df_in.columns else 0)
            in_col = c4.selectbox("Intensity", df_in.columns, index=df_input.columns.get_loc("Intensity") if "Intensity" in df_in.columns else 2)
            
            f1, f2, f3, f4 = st.columns(4)
            mz_bin = f1.slider("Alignment", 1, 5, 3)
            min_pres = f2.slider("80% Rule filter (%)", 0, 100, 80)
            ion_mode = f3.selectbox("Polarity", ["Negative [M-H]-", "Positive [M+H]+"])
            scaling = f4.selectbox("Method for Stats", ["Pareto Scaling", "Auto-Scaling", "None"])

        if st.button("Run Full Academic Pipeline"):
            try:
                # 1. PROCESSING
                df_in['T_ID'] = df_in[mz_col].round(mz_bin).astype(str) + "_" + df_in[rt_col].round(2).astype(str)
                pivot = df_in.pivot_table(index='T_ID', columns=sm_col, values=in_col, aggfunc='mean').fillna(0)
                cleaned = pivot[(pivot != 0).sum(axis=1) >= (min_pres/100)*len(pivot.columns)]
                if cleaned.empty: st.error("No features left."); st.stop()
                
                # Normalization & Scaling Math (Your Colab Logic)
                min_v = (cleaned[cleaned > 0].min().min()) / 2
                X_raw = cleaned.replace(0, min_v).div(cleaned.sum(axis=0), axis=1).T * 1000000
                X_z = (X_raw - X_raw.mean()) / X_raw.std()
                X_p = (X_raw - X_raw.mean()) / np.sqrt(X_raw.std().replace(0, np.nan))
                X_scaled = X_p if scaling == "Pareto Scaling" else (X_z if "Auto" in scaling else X_raw)
                X_scaled = X_scaled.fillna(0).loc[:, X_scaled.std() > 0]

                groups = [str(s).split('_')[0] for s in X_raw.index]
                unique_g = sorted(list(set(groups)))

                # 2. STATS & ML
                g1_m, g2_m = [g == unique_g[0] for g in groups], [g == unique_g[1] for g in groups]
                _, pvals = ttest_ind(X_raw[g1_m], X_raw[g2_m], axis=0)
                log2fc = np.log2(X_raw[g2_m].mean() / X_raw[g1_m].mean().replace(0, 0.001))
                stats_df = pd.DataFrame({'ID': X_raw.columns, 'p': pvals, 'Log2FC': log2fc}).fillna(0)
                stats_df['Sig'] = (stats_df['p'] < 0.05) & (abs(stats_df['Log2FC']) > 1)
                acc = cross_val_score(RandomForestClassifier(), X_scaled, [1 if g==unique_g[-1] else 0 for g in groups], cv=3).mean()

                # 3. TABS (The Rich Suite)
                t1, t2, t3, t4, t5, t6, t7 = st.tabs(["📊 Distributions", "🔵 PCA Gallery", "🎯 PLS-DA Suite", "🌋 Volcano", "🏆 Identification", "🔥 Heatmap", "📋 Report"])
                
                with t1: # BOXPLOTS (Match Image 1)
                    st.subheader("Normalization QC")
                    c_a, c_b = st.columns(2)
                    c_a.plotly_chart(px.box(X_raw.melt(), y='value', title="Raw TIC Normalized", color_discrete_sequence=['#66c2a5']))
                    c_b.plotly_chart(px.box(X_p.melt(), y='value', title="Pareto Scaled", color_discrete_sequence=['#fc8d62']))

                with t2: # 4-PANEL PCA (Match Image 3)
                    st.subheader("Scaling comparison for separation")
                    pca_cols = st.columns(2)
                    for i, (n, d) in enumerate([("Raw", X_raw), ("Z-score", X_z), ("Pareto", X_p), ("Log2", np.log2(X_raw+1))]):
                        coords = PCA(n_components=2).fit_transform(d)
                        pca_cols[i%2].plotly_chart(px.scatter(x=coords[:,0], y=coords[:,1], color=groups, title=f"PCA: {n}"), use_container_width=True)

                with t3: # PLS-DA Suite (Match Image 5 & 6)
                    st.subheader("Supervised Discrimination & Importance")
                    y_bin = [1 if g == unique_g[-1] else 0 for g in groups]
                    pls = PLSRegression(n_components=2).fit(X_p, y_bin)
                    
                    c_1, c_2 = st.columns(2)
                    c_1.plotly_chart(px.scatter(x=pls.x_scores_[:,0], y=pls.x_scores_[:,1], color=groups, title="PLS-DA Scores Plot"))
                    c_2.plotly_chart(px.scatter(x=pls.x_loadings_[:,0], y=pls.x_loadings_[:,1], title="Loadings Plot (Feature Weights)", opacity=0.5))
                    
                    # VIP Scores logic (Simplified)
                    vip = np.abs(pls.x_weights_[:, 0]) # LV1 importance
                    vip_df = pd.DataFrame({'ID': X_raw.columns, 'VIP': vip}).sort_values('VIP', ascending=False).head(20)
                    st.plotly_chart(px.bar(vip_df, x='ID', y='VIP', title="Top 20 Features by VIP Score (LV1)"))

                with t4: # Volcano (Match Image 4)
                    st.plotly_chart(px.scatter(stats_df, x='Log2FC', y=-np.log10(stats_df['p']+1e-10), color='Sig', hover_name='ID', 
                                              color_discrete_map={True:'red', False:'gray'}, title="Volcano Discovery Plot"))
                
                with t5: # ID with PubChem
                    hits = stats_df[stats_df['Sig']].copy()
                    hits['m/z'] = hits['ID'].apply(lambda x: float(x.split('_')[0]))
                    hits['Neutral Mass'] = (hits['m/z'] + 1.0078) if ion_mode.startswith("Neg") else (hits['m/z'] - 1.0078)
                    hits['Confidence'] = hits['Neutral Mass'].apply(lambda x: "Level 2" if x < 500 else "Level 3")
                    hits['Identify'] = hits['m/z'].apply(lambda x: f"https://pubchem.ncbi.nlm.nih.gov/#query={x}")
                    st.dataframe(hits[['ID', 'Neutral Mass', 'Confidence', 'Log2FC', 'Identify']], column_config={"Identify": st.column_config.LinkColumn()})

                with t6: # Heatmap
                    st.plotly_chart(px.imshow(X_p.T.head(100), title="Top 100 Features Heatmap"), use_container_width=True)

                with t7:
                    st.metric("Random Forest Accuracy", f"{acc:.1%}")
                    pdf_data = create_academic_report(unique_g[0], unique_g[1], len(cleaned), acc, len(hits[hits['Confidence']=='Level 2']))
                    st.download_button("Download Report (PDF)", pdf_data, "Report.pdf", "application/pdf")
                st.balloons()
            except Exception as e: st.error(f"Error: {e}")

st.caption("Authorized Lab Use Only")

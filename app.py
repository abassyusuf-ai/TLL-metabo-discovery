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
st.sidebar.info("**Bioinformatics Hub**\nIntegrated pipeline for HRMS data processing.")
st.sidebar.markdown("---")
st.sidebar.subheader("External Links")
st.sidebar.markdown("[GNPS Dashboard](https://gnps.ucsd.edu/)")
st.sidebar.markdown("[SIRIUS Download](https://bright.cs.uni-jena.de/sirius/)")
st.sidebar.markdown("---")
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
               f"Significant hits identified: {leads_count}.")
    pdf.multi_cell(0, 10, summary)
    return bytes(pdf.output())

# --- 4. MAIN INTERFACE ---
st.title("TLL Metabo-Discovery: Professional Analytics Suite")
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
    c_u1, c_u2 = st.columns(2)
    with c_u1: uploaded_file = st.file_uploader("1. Upload Quantified Table (.csv)", type=["csv"])
    with c_u2: metadata_file = st.file_uploader("2. Optional: Upload Metadata (.csv or .txt)", type=["csv", "txt"])

    if uploaded_file:
        df_in = pd.read_csv(uploaded_file)
        df_meta = None
        if metadata_file:
            sep = '\t' if metadata_file.name.endswith('.txt') else ','
            df_meta = pd.read_csv(metadata_file, sep=sep)
            st.success("Metadata Loaded")

        with st.expander("Parameters & Mapping Configuration"):
            col1, col2, col3, col4 = st.columns(4)
            mz_col = col1.selectbox("m/z", df_in.columns, index=0)
            rt_col = col2.selectbox("RT", df_in.columns, index=1 if "RT_min" not in df_in.columns else df_in.columns.get_loc("RT_min"))
            sample_col = col3.selectbox("Sample ID", df_in.columns, index=df_in.columns.get_loc("Sample") if "Sample" in df_in.columns else 0)
            in_col = col4.selectbox("Intensity", df_in.columns, index=df_in.columns.get_loc("Intensity") if "Intensity" in df_in.columns else 2)
            
            st.markdown("---")
            meta_link, group_col = None, None
            if df_meta is not None:
                m1, m2 = st.columns(2)
                meta_link = m1.selectbox("Link Metadata via column:", df_meta.columns, index=0)
                group_col = m2.selectbox("Grouping column:", df_meta.columns, index=min(1, len(df_meta.columns)-1))
            
            f1, f2, f3, f4 = st.columns(4)
            mz_bin = f1.slider("Alignment", 1, 5, 3)
            min_pres = f2.slider("Min Presence %", 0, 100, 80)
            ion_mode = f3.selectbox("Polarity", ["Negative [M-H]-", "Positive [M+H]+"])
            scaling = f4.selectbox("Method", ["Pareto Scaling", "Auto-Scaling", "None"])

        if st.button("Run Full Academic Pipeline"):
            try:
                # 1. Processing
                df_in['T_ID'] = df_in[mz_col].round(mz_bin).astype(str) + "_" + df_in[rt_col].round(2).astype(str)
                pivot = df_in.pivot_table(index='T_ID', columns=sample_col, values=in_col, aggfunc='mean').fillna(0)
                cleaned = pivot[(pivot != 0).sum(axis=1) >= (min_pres/100)*len(pivot.columns)]
                if cleaned.empty: st.error("No features left. Lower Min Presence."); st.stop()
                
                min_v = (cleaned[cleaned > 0].min().min()) / 2
                X_raw = cleaned.replace(0, min_v).div(cleaned.sum(axis=0), axis=1).T * 1000000

                # 2. Metadata Assignment
                if df_meta is not None:
                    meta_indexed = df_meta.set_index(meta_link)
                    aligned_meta = meta_indexed.reindex(X_raw.index)
                    groups = aligned_meta[group_col].astype(str).tolist()
                else:
                    groups = [str(s).split('_')[0] for s in X_raw.index]
                
                unique_g = sorted(list(set(groups)))
                # Define y_bin for Supervised analysis
                y_bin = [1 if g == unique_g[-1] else 0 for g in groups]

                # 3. Scaling
                X_z = (X_raw - X_raw.mean()) / X_raw.std()
                X_p = (X_raw - X_raw.mean()) / np.sqrt(X_raw.std().replace(0, np.nan))
                X_scaled = X_p if scaling == "Pareto Scaling" else (X_z if "Auto" in scaling else X_raw)
                X_scaled = X_scaled.fillna(0).loc[:, X_scaled.std() > 0]

                # 4. Initialize result containers for Tabs
                stats_ready = False
                stats_df = pd.DataFrame()
                hits = pd.DataFrame()
                acc = 0.0

                if len(unique_g) >= 2:
                    g1_m, g2_m = [g == unique_g[0] for g in groups], [g == unique_g[1] for g in groups]
                    _, pvals = ttest_ind(X_raw.iloc[g1_m], X_raw.iloc[g2_m], axis=0)
                    log2fc = np.log2(X_raw.iloc[g2_m].mean() / X_raw.iloc[g1_m].mean().replace(0, 0.001))
                    stats_df = pd.DataFrame({'ID': X_raw.columns, 'p': pvals, 'Log2FC': log2fc}).fillna(0)
                    stats_df['Sig'] = (stats_df['p'] < 0.05) & (abs(stats_df['Log2FC']) > 1)
                    hits = stats_df[stats_df['Sig']].copy()
                    acc = cross_val_score(RandomForestClassifier(), X_scaled, y_bin, cv=3).mean()
                    stats_ready = True

                # 5. TABS
                t1, t2, t3, t4, t5, t6, t7 = st.tabs(["📊 Distribution", "🔵 PCA", "🎯 PLS-DA", "🌋 Volcano", "🏆 Identification", "🔗 Export", "📋 Report"])
                
                with t1: st.plotly_chart(px.box(X_raw.melt(), y='value', title="Data QC (TIC Normalized)"))
                with t2:
                    coords = PCA(n_components=2).fit_transform(X_scaled)
                    st.plotly_chart(px.scatter(x=coords[:,0], y=coords[:,1], color=groups, title="Unsupervised PCA"))
                
                with t3:
                    if stats_ready:
                        pls = PLSRegression(n_components=2).fit(X_scaled, y_bin)
                        st.plotly_chart(px.scatter(x=pls.x_scores_[:,0], y=pls.x_scores_[:,1], color=groups, title="Supervised PLS-DA"))
                    else: st.warning("Need 2 groups for PLS-DA.")

                with t4:
                    if stats_ready:
                        st.plotly_chart(px.scatter(stats_df, x='Log2FC', y=-np.log10(stats_df['p']+1e-10), color='Sig', hover_name='ID', title="Volcano Plot"))
                
                with t5:
                    if stats_ready and not hits.empty:
                        hits['m/z'] = hits['ID'].apply(lambda x: float(x.split('_')[0]))
                        hits['Neutral'] = (hits['m/z'] + 1.0078) if ion_mode.startswith("Neg") else (hits['m/z'] - 1.0078)
                        hits['Identify'] = hits['m/z'].apply(lambda x: f"https://pubchem.ncbi.nlm.nih.gov/#query={x}")
                        hits['Confidence'] = hits['Neutral'].apply(lambda x: "Level 2" if x < 500 else "Level 3")
                        st.dataframe(hits[['ID', 'Neutral', 'Confidence', 'Log2FC', 'p', 'Identify']], column_config={"Identify": st.column_config.LinkColumn()})

                with t6:
                    st.subheader("Platform Export")
                    gnps_table = X_raw.T.copy().reset_index()
                    gnps_table.insert(0, 'row ID', range(1, len(gnps_table) + 1))
                    gnps_table.insert(1, 'row m/z', gnps_table['ID'].apply(lambda x: x.split('_')[0]))
                    gnps_table.insert(2, 'row RT', gnps_table['ID'].apply(lambda x: x.split('_')[1]))
                    gnps_meta = pd.DataFrame({'filename': X_raw.index, 'Group': groups})
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.download_button("📥 Download GNPS Table", gnps_table.to_csv(index=False).encode('utf-8'), "gnps_quant.csv")
                        st.download_button("📥 Download GNPS Metadata", gnps_meta.to_csv(index=False, sep='\t').encode('utf-8'), "gnps_meta.txt")
                    with c2:
                        if stats_ready:
                            st.download_button("📥 Download SIRIUS List", hits[['m/z', 'Log2FC', 'p']].to_csv(index=False).encode('utf-8'), "sirius.csv")

                with t7:
                    if stats_ready:
                        st.metric("Model Accuracy", f"{acc:.1%}")
                        pdf_data = create_academic_report(unique_g[0], unique_g[1], len(cleaned), acc, len(hits))
                        st.download_button("Download Report (PDF)", pdf_data, "Report.pdf", "application/pdf")
                st.balloons()
            except Exception as e: st.error(f"Analysis Error: {e}")

st.caption("Authorized Lab Use Only")

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

st.set_page_config(page_title="TLL Metabo-Discovery", layout="wide")

# --- SIDEBAR ---
st.sidebar.title("TLL Metabo-Discovery")
st.sidebar.info("**Bioinformatics Hub**\nSupporting Custom Metadata Mapping.")
st.sidebar.markdown("---")
st.sidebar.markdown("[GNPS Dashboard](https://gnps.ucsd.edu/)")
st.sidebar.markdown("[SIRIUS Download](https://bright.cs.uni-jena.de/sirius/)")
st.sidebar.caption("Developed by Abass Yusuf | Lab Support")

# --- PDF GENERATOR ---
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
               f"Significant hits found: {leads_count}.")
    pdf.multi_cell(0, 10, summary)
    return bytes(pdf.output())

st.title("TLL Metabo-Discovery: Professional Analytics Suite")
st.markdown("---")

mode = st.radio("Select Analysis Module:", ("Raw Data Extraction", "Discovery Dashboard"))

# ============================================
# MODULE 1: RAW DATA PROCESSOR
# ============================================
if mode == "Raw Data Extraction":
    st.subheader("Batch mzML Feature Extraction")
    uploaded_mzmls = st.file_uploader("Upload .mzML files", type=["mzml"], accept_multiple_files=True)
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
    col_data, col_meta = st.columns(2)
    with col_data:
        uploaded_file = st.file_uploader("1. Upload Quantified Table (.csv)", type=["csv"])
    with col_meta:
        metadata_file = st.file_uploader("2. Optional: Upload Metadata (CSV/TXT)", type=["csv", "txt"])

    if uploaded_file:
        df_in = pd.read_csv(uploaded_file)
        df_meta = None
        if metadata_file:
            # Auto-detect separator for metadata
            sep = '\t' if metadata_file.name.endswith('.txt') else ','
            df_meta = pd.read_csv(metadata_file, sep=sep)
            st.success(" Metadata Loaded")

        with st.expander("Parameters & Mapping Configuration"):
            c1, c2, c3, c4 = st.columns(4)
            mz_col = c1.selectbox("m/z column", df_in.columns, index=0)
            rt_col = c2.selectbox("RT column", df_in.columns, index=1 if "RT_min" not in df_in.columns else df_in.columns.get_loc("RT_min"))
            sample_col = c3.selectbox("Sample ID (in data)", df_in.columns, index=df_in.columns.get_loc("Sample") if "Sample" in df_in.columns else 0)
            in_col = c4.selectbox("Intensity column", df_in.columns, index=df_in.columns.get_loc("Intensity") if "Intensity" in df_in.columns else 2)
            
            st.markdown("---")
            if df_meta is not None:
                m1, m2 = st.columns(2)
                meta_link_col = m1.selectbox("Link Metadata via this column:", df_meta.columns, index=0, help="This column must match the Sample IDs in your data.")
                group_col = m2.selectbox("Use this column for Grouping:", df_meta.columns, index=1 if len(df_meta.columns)>1 else 0)
            
            f1, f2, f3, f4 = st.columns(4)
            mz_bin = f1.slider("Alignment", 1, 5, 3)
            min_pres = f2.slider("Min Presence (%)", 0, 100, 80)
            ion_mode = f3.selectbox("Polarity", ["Negative [M-H]-", "Positive [M+H]+"])
            scaling = f4.selectbox("Scaling", ["Pareto Scaling", "Auto-Scaling", "None"])

        if st.button("Run Full Academic Pipeline"):
            try:
                # 1. PROCESSING
                df_in['T_ID'] = df_in[mz_col].round(mz_bin).astype(str) + "_" + df_in[rt_col].round(2).astype(str)
                pivot = df_in.pivot_table(index='T_ID', columns=sample_col, values=in_col, aggfunc='mean').fillna(0)
                cleaned = pivot[(pivot != 0).sum(axis=1) >= (min_pres/100)*len(pivot.columns)]
                if cleaned.empty: st.error("No features left."); st.stop()
                
                min_v = (cleaned[cleaned > 0].min().min()) / 2
                X_raw = cleaned.replace(0, min_v).div(cleaned.sum(axis=0), axis=1).T * 1000000
                
                # 2. MAPPING GROUPS
                if df_meta is not None:
                    # Merge metadata based on user selection
                    meta_indexed = df_meta.set_index(meta_link_col)
                    # Align metadata to the samples in the pivot table
                    aligned_meta = meta_indexed.reindex(X_raw.index)
                    groups = aligned_meta[group_col].astype(str).tolist()
                else:
                    # Fallback to original logic
                    groups = [str(s).split('_')[0] for s in X_raw.index]
                
                unique_g = sorted(list(set(groups)))

                # 3. SCALING
                X_z = (X_raw - X_raw.mean()) / X_raw.std()
                X_p = (X_raw - X_raw.mean()) / np.sqrt(X_raw.std().replace(0, np.nan))
                X_scaled = X_p if scaling == "Pareto Scaling" else (X_z if "Auto" in scaling else X_raw)
                X_scaled = X_scaled.fillna(0).loc[:, X_scaled.std() > 0]

                # 4. STATS & ML
                if len(unique_g) >= 2:
                    g1_mask = [g == unique_g[0] for g in groups]
                    g2_mask = [g == unique_g[1] for g in groups]
                    _, pvals = ttest_ind(X_raw.iloc[g1_mask], X_raw.iloc[g2_mask], axis=0)
                    log2fc = np.log2(X_raw.iloc[g2_mask].mean() / X_raw.iloc[g1_mask].mean().replace(0, 0.001))
                    stats_df = pd.DataFrame({'ID': X_raw.columns, 'p': pvals, 'Log2FC': log2fc}).fillna(0)
                    stats_df['Sig'] = (stats_df['p'] < 0.05) & (abs(stats_df['Log2FC']) > 1)
                    y_bin = [1 if g == unique_g[-1] else 0 for g in groups]
                    acc = cross_val_score(RandomForestClassifier(), X_scaled, y_bin, cv=3).mean()

                # 5. TABS
                t1, t2, t3, t4, t5, t6, t7 = st.tabs(["📊 Distribution", "🔵 PCA", "🎯 PLS-DA", "🌋 Volcano", "🏆 Identification", "🔗 Platform Export", "📋 Summary"])
                
                with t1: st.plotly_chart(px.box(X_raw.melt(), y='value', title="Data QC (TIC Normalized)"))
                with t2:
                    coords = PCA(n_components=2).fit_transform(X_p)
                    st.plotly_chart(px.scatter(x=coords[:,0], y=coords[:,1], color=groups, title="PCA: Pareto"))
                with t3:
                    pls = PLSRegression(n_components=2).fit(X_p, y_bin)
                    st.plotly_chart(px.scatter(x=pls.x_scores_[:,0], y=pls.x_scores_[:,1], color=groups, title="Supervised PLS-DA"))
                with t4:
                    st.plotly_chart(px.scatter(stats_df, x='Log2FC', y=-np.log10(stats_df['p']+1e-10), color='Sig', hover_name='ID', title="Volcano Plot"))
                with t5:
                    hits = stats_df[stats_df['Sig']].copy()
                    hits['m/z'] = hits['ID'].apply(lambda x: float(x.split('_')[0]))
                    hits['Neutral Mass'] = (hits['m/z'] + 1.0078) if ion_mode.startswith("Neg") else (hits['m/z'] - 1.0078)
                    hits['Identify'] = hits['m/z'].apply(lambda x: f"https://pubchem.ncbi.nlm.nih.gov/#query={x}")
                    st.dataframe(hits[['ID', 'Neutral Mass', 'Log2FC', 'p', 'Identify']], column_config={"Identify": st.column_config.LinkColumn()})
                with t6:
                    st.subheader("Generate Platform-Ready Files")
                    gnps_table = X_raw.T.copy().reset_index()
                    gnps_table.insert(0, 'row ID', range(1, len(gnps_table) + 1))
                    gnps_table.insert(1, 'row m/z', gnps_table['ID'].apply(lambda x: x.split('_')[0]))
                    gnps_table.insert(2, 'row retention time', gnps_table['ID'].apply(lambda x: x.split('_')[1]))
                    gnps_meta = pd.DataFrame({'filename': X_raw.index, 'ATTRIBUTE_Group': groups})
                    c1, c2 = st.columns(2)
                    with c1:
                        st.download_button("📥 Download GNPS Feature Table", gnps_table.to_csv(index=False).encode('utf-8'), "GNPS_quant.csv")
                        st.download_button("📥 Download GNPS Metadata", gnps_meta.to_csv(index=False, sep='\t').encode('utf-8'), "GNPS_metadata.txt")
                    with c2:
                        st.download_button("📥 Download SIRIUS Mass List", hits[['m/z', 'Log2FC', 'p']].to_csv(index=False).encode('utf-8'), "sirius_batch.csv")
                with t7:
                    st.metric("Model Predictive Accuracy", f"{acc:.1%}")
                    pdf_data = create_academic_report(unique_g[0], unique_g[1], len(cleaned), acc, len(hits))
                    st.download_button("Download Report (PDF)", pdf_data, "Report.pdf", "application/pdf")
                st.balloons()
            except Exception as e: st.error(f"Analysis Error: {e}")

st.caption("Internal Research Suite | Customized Metadata Support")

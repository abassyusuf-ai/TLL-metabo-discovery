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
            st.success("Metadata Loaded Successfully")

        with st.expander("Parameters & Mapping Configuration"):
            col1, col2, col3, col4 = st.columns(4)
            mz_col = col1.selectbox("m/z Column", df_in.columns, index=0)
            rt_col = col2.selectbox("RT Column", df_in.columns, index=1 if "RT_min" not in df_in.columns else df_in.columns.get_loc("RT_min"))
            sample_col = col3.selectbox("Sample ID", df_in.columns, index=df_in.columns.get_loc("Sample") if "Sample" in df_in.columns else 0)
            in_col = col4.selectbox("Intensity Column", df_in.columns, index=df_in.columns.get_loc("Intensity") if "Intensity" in df_in.columns else 2)
            
            st.markdown("---")
            meta_link, group_col_name = None, None
            if df_meta is not None:
                m1, m2 = st.columns(2)
                meta_link = m1.selectbox("Link Metadata via column:", df_meta.columns, index=0)
                group_col_name = m2.selectbox("Grouping column (e.g. CLASS_plant):", df_meta.columns, index=min(1, len(df_meta.columns)-1))
            
            f1, f2, f3, f4 = st.columns(4)
            mz_bin = f1.slider("Alignment Precision", 1, 5, 3)
            min_pres = f2.slider("Min Presence Filter %", 0, 100, 80)
            ion_mode = f3.selectbox("Ionization Polarity", ["Negative [M-H]-", "Positive [M+H]+"])
            scaling = f4.selectbox("Statistical Scaling", ["Pareto Scaling", "Auto-Scaling", "None"])

        if st.button("Run Full Academic Pipeline"):
            try:
                # 1. Processing Logic (Zero Reduction & Alignment)
                df_in['ID'] = df_in[mz_col].round(mz_bin).astype(str) + "_" + df_in[rt_col].round(2).astype(str)
                pivot = df_in.pivot_table(index='ID', columns=sample_col, values=in_col, aggfunc='mean').fillna(0)
                
                # Apply 80% Rule (Filter out features with too many zeros)
                thresh = (min_pres/100) * len(pivot.columns)
                cleaned = pivot[(pivot != 0).sum(axis=1) >= thresh]
                
                if cleaned.empty: 
                    st.error("Filtering error: Zero features remaining. Please lower 'Min Presence %'.")
                    st.stop()
                
                # Impute Gaps (Replacing 0s with 1/2 of minimum)
                min_v = (cleaned[cleaned > 0].min().min()) / 2
                X_raw_filled = cleaned.replace(0, min_v).div(cleaned.sum(axis=0), axis=1).T * 1000000

                # 2. Metadata Assignment
                if df_meta is not None:
                    meta_indexed = df_meta.set_index(meta_link)
                    aligned_meta = meta_indexed.reindex(X_raw_filled.index)
                    groups = aligned_meta[group_col_name].astype(str).tolist()
                else:
                    groups = [str(s).split('_')[0] for s in X_raw_filled.index]
                
                unique_g = sorted(list(set(groups)))
                y_bin = [1 if g == unique_g[-1] else 0 for g in groups]

                # 3. MULTI-SCALING MATH (For visual gallery)
                X_z = (X_raw_filled - X_raw_filled.mean()) / X_raw_filled.std()
                X_p = (X_raw_filled - X_raw_filled.mean()) / np.sqrt(X_raw_filled.std().replace(0, np.nan))
                X_scaled = X_p if scaling == "Pareto Scaling" else (X_z if "Auto" in scaling else X_raw_filled)
                X_scaled = X_scaled.fillna(0).loc[:, X_scaled.std() > 0]

                # 4. STATS & ML
                stats_ready = False
                if len(unique_g) >= 2:
                    g1_m, g2_m = [g == unique_g[0] for g in groups], [g == unique_g[1] for g in groups]
                    _, pvals = ttest_ind(X_raw_filled.iloc[g1_m], X_raw_filled.iloc[g2_m], axis=0)
                    log2fc = np.log2(X_raw_filled.iloc[g2_m].mean() / X_raw_filled.iloc[g1_m].mean().replace(0, 0.001))
                    stats_df = pd.DataFrame({'ID': X_raw_filled.columns, 'p': pvals, 'Log2FC': log2fc}).fillna(0)
                    stats_df['Sig'] = (stats_df['p'] < 0.05) & (abs(stats_df['Log2FC']) > 1)
                    hits = stats_df[stats_df['Sig']].copy()
                    acc = cross_val_score(RandomForestClassifier(), X_scaled, y_bin, cv=3).mean()
                    stats_ready = True

                # 5. TABS INTERFACE (The Rich Gallery)
                t1, t2, t3, t4, t5, t6, t7 = st.tabs(["📊 Distributions", "📈 Group Means", "🔵 PCA Analysis", "🎯 PLS-DA Suite", "🌋 Discovery", "🔗 Export", "📋 Report"])
                
                with t1: # BOXPLOTS (Match Image 1)
                    st.subheader("Intensity Distribution across Normalization Scales")
                    cols = st.columns(3)
                    cols[0].plotly_chart(px.box(X_raw_filled.melt(), y='value', title="TIC Normalized", color_discrete_sequence=['#66c2a5']))
                    cols[1].plotly_chart(px.box(X_z.melt(), y='value', title="Z-score Scaled", color_discrete_sequence=['#8da0cb']))
                    cols[2].plotly_chart(px.box(X_p.melt(), y='value', title="Pareto Scaled", color_discrete_sequence=['#fc8d62']))

                with t2: # BAR CHARTS (Match Image 2)
                    st.subheader("Comparison of Global Group Means")
                    # Calculate mean and SD per scaling method
                    m_df = pd.DataFrame({'Group': groups, 'Raw': X_raw_filled.mean(axis=1), 'Z': X_z.mean(axis=1), 'Pareto': X_p.mean(axis=1)})
                    cols_bar = st.columns(3)
                    cols_bar[0].plotly_chart(px.bar(m_df.groupby('Group').mean().reset_index(), x='Group', y='Raw', title="Raw Mean ± SD", error_y=m_df.groupby('Group').std()['Raw']))
                    cols_bar[1].plotly_chart(px.bar(m_df.groupby('Group').mean().reset_index(), x='Group', y='Z', title="Z-score Mean ± SD", error_y=m_df.groupby('Group').std()['Z']))
                    cols_bar[2].plotly_chart(px.bar(m_df.groupby('Group').mean().reset_index(), x='Group', y='Pareto', title="Pareto Mean ± SD", error_y=m_df.groupby('Group').std()['Pareto']))

                with t3: # PCA 4-PANEL (Match Image 4)
                    st.subheader("Multivariate Scaling Comparison (Unsupervised)")
                    p_cols = st.columns(2)
                    scaling_types = [("Raw (TIC)", X_raw_filled), ("Pareto", X_p), ("Z-score", X_z), ("Log2", np.log2(X_raw_filled + 1))]
                    for i, (name, d_set) in enumerate(scaling_types):
                        pca_c = PCA(n_components=2).fit_transform(d_set.fillna(0))
                        fig = px.scatter(x=pca_c[:,0], y=pca_c[:,1], color=groups, title=f"PCA: {name}", template="plotly_white")
                        p_cols[i%2].plotly_chart(fig, use_container_width=True)
                
                with t4: # PLS-DA Suite (Match Image 5 & 6)
                    if stats_ready:
                        st.subheader("Supervised Discrimination & Contribution")
                        pls = PLSRegression(n_components=2).fit(X_scaled, y_bin)
                        c1, c2 = st.columns(2)
                        c1.plotly_chart(px.scatter(x=pls.x_scores_[:,0], y=pls.x_scores_[:,1], color=groups, title="PLS-DA Scores (Group Discrimination)"), use_container_width=True)
                        c2.plotly_chart(px.scatter(x=pls.x_loadings_[:,0], y=pls.x_loadings_[:,1], title="PLS-DA Loadings (Feature Contributions)", opacity=0.4), use_container_width=True)
                    else: st.warning("Group labels required for supervised modeling.")

                with t5: # IDENTIFICATION (PubChem)
                    if stats_ready and not hits.empty:
                        st.subheader("Biomarker Discovery & Identification")
                        hits['m/z'] = hits['ID'].apply(lambda x: float(x.split('_')[0]))
                        hits['Neutral'] = (hits['m/z'] + 1.0078) if ion_mode.startswith("Neg") else (hits['m/z'] - 1.0078)
                        hits['Identify'] = hits['m/z'].apply(lambda x: f"https://pubchem.ncbi.nlm.nih.gov/#query={x}")
                        hits['Confidence'] = hits['Neutral'].apply(lambda x: "Level 2" if x < 500 else "Level 3")
                        st.dataframe(hits[['ID', 'Neutral', 'Confidence', 'Log2FC', 'p', 'Identify']], column_config={"Identify": st.column_config.LinkColumn()}, use_container_width=True)

                with t6: # EXPORT HUB
                    gnps_t = X_raw_filled.T.copy().reset_index()
                    gnps_t.insert(0, 'row ID', range(1, len(gnps_t) + 1))
                    gnps_t.insert(1, 'row m/z', gnps_t['ID'].apply(lambda x: x.split('_')[0]))
                    gnps_t.insert(2, 'row RT', gnps_t['ID'].apply(lambda x: x.split('_')[1]))
                    c_e1, c_e2 = st.columns(2)
                    with c_e1:
                        st.download_button("📥 GNPS Table", gnps_t.to_csv(index=False).encode('utf-8'), "gnps_quant.csv")
                    with c_e2:
                        if stats_ready:
                            st.download_button("📥 SIRIUS List", hits[['m/z', 'Log2FC', 'p']].to_csv(index=False).encode('utf-8'), "sirius.csv")

                with t7:
                    if stats_ready:
                        st.metric("Discovery Validation Accuracy", f"{acc:.1%}")
                        pdf_data = create_academic_report(unique_g[0], unique_g[1], len(cleaned), acc, len(hits))
                        st.download_button("Download Report (PDF)", pdf_data, "Report.pdf", "application/pdf")
                st.balloons()
            except Exception as e: st.error(f"Analysis Error: {e}")

st.caption("Authorized Lab Use | Standardized Research Pipeline")

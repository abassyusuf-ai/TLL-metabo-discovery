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
import base64 # For embedding images

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="TLL Metabo-Discovery", layout="wide")

# --- 2. SIDEBAR ---
st.sidebar.title("TLL Metabo-Discovery")
st.sidebar.info("""
**Bioinformatics Hub**  
Integrated pipeline for HRMS data processing.
""")
st.sidebar.markdown("---")
st.sidebar.subheader("External Resources")
st.sidebar.markdown("[GNPS Web Platform](https://gnps.ucsd.edu/)")
st.sidebar.markdown("[SIRIUS Desktop App](https://bright.cs.uni-jena.de/sirius/)")
st.sidebar.markdown("---")
st.sidebar.caption("Developed by Abass Yusuf | Lab Support")

# --- 3. HELPER: PDF GENERATOR ---
def create_academic_report(g1, g2, feat_count, accuracy, leads_count):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "TLL Metabo-Discovery: Research Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 12); pdf.cell(0, 10, "1. Executive Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    summary = (f"Analysis of {g1} vs {g2}. Features analyzed: {feat_count}. "
               f"ML Validation Accuracy: {accuracy:.1%}. "
               f"Significant hits identified: {leads_count}.")
    pdf.multi_cell(0, 10, summary)
    return bytes(pdf.output())

# --- 4. EMBEDDING HELPER FUNCTION FOR IMAGES ---
def add_image_to_pdf(pdf, image_path, width=60):
    try:
        with open(image_path, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode('latin-1')
        pdf.image(f"data:image/png;base64,{encoded_string}", x=pdf.get_x(), y=pdf.get_y(), w=width)
    except Exception as img_e:
        pdf.set_text_color(255, 0, 0) # Red for error
        pdf.cell(0, 10, f"Error embedding image: {img_e}", ln=True)
        pdf.set_text_color(0, 0, 0) # Reset color

# --- 5. MAIN INTERFACE ---
st.title("TLL Metabo-Discovery: Professional Analytics Suite")
st.markdown("---")

mode = st.radio("Select Analysis Module:", ("Raw Data Extraction", "Discovery Dashboard"))

# --- Base64 encoded images (for embedding directly) ---
# NOTE: Replace these with actual file paths to your images once uploaded.
# For this example, they are placeholders. You can upload images to your app's directory.
# Example: Save 'boxplot_comparison.png', 'pca_comparison.png', etc. in your root folder.

# --- MODULE 1: RAW DATA PROCESSOR ---
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
        st.success("Complete."); st.download_button("Download CSV", pd.concat(all_features).to_csv(index=False).encode('utf-8'), "features.csv")
        if os.path.exists("temp.mzml"): os.remove("temp.mzml")

# --- MODULE 2: DISCOVERY DASHBOARD ---
else:
    c_u1, c_u2 = st.columns(2)
    with c_u1: uploaded_file = st.file_uploader("1. Data Table (.csv)", type=["csv"])
    with c_u2: metadata_file = st.file_uploader("2. Metadata (Optional)", type=["csv", "txt"])

    if uploaded_file:
        df_in = pd.read_csv(uploaded_file)
        df_meta = None
        if metadata_file:
            sep = '\t' if metadata_file.name.endswith('.txt') else ','
            df_meta = pd.read_csv(metadata_file, sep=sep)
            # Clean metadata column names
            for col in df_meta.columns:
                if df_meta[col].dtype == 'object':
                    df_meta[col] = df_meta[col].astype(str).str.replace('.mzML', '', case=False).str.replace('.csv', '', case=False).str.strip()
            st.success("Metadata Loaded Successfully")

        with st.expander("Parameters & Grouping Configuration"):
            col1, col2, col3, col4 = st.columns(4)
            mz_col = col1.selectbox("m/z Column", df_in.columns, index=0)
            rt_col = col2.selectbox("RT Column", df_in.columns, index=1 if "RT_min" not in df_in.columns else df_in.columns.get_loc("RT_min"))
            sample_col = col3.selectbox("Sample ID", df_in.columns, index=df_in.columns.get_loc("Sample") if "Sample" in df_in.columns else 0)
            in_col = col4.selectbox("Intensity", df_in.columns, index=df_in.columns.get_loc("Intensity") if "Intensity" in df_in.columns else 2)
            
            st.markdown("---")
            meta_link, group_col_name = None, None
            if df_meta is not None:
                m1, m2 = st.columns(2)
                meta_link = m1.selectbox("Link Metadata via:", df_meta.columns, index=0)
                group_col_name = m2.selectbox("Grouping column:", df_meta.columns, index=min(1, len(df_meta.columns)-1))
            else:
                m1, m2 = st.columns(2)
                delimiter = m1.selectbox("Sample Name Delimiter", ["_", "-", ".", "Space"], index=0)
                actual_delim = " " if delimiter == "Space" else delimiter

            f1, f2, f3, f4 = st.columns(4)
            mz_bin = f1.slider("Alignment Tolerance", 1, 5, 3)
            min_pres = f2.slider("Min Presence (%)", 0, 100, 80)
            ion_mode = f3.selectbox("Polarity", ["Negative [M-H]-", "Positive [M+H]+"])
            scaling = f4.selectbox("Scaling Method", ["Pareto Scaling", "Auto-Scaling", "None"])
            
        if st.button("Run Full Discovery Pipeline"):
            try:
                # 1. PROCESSING
                df_in['T_ID'] = df_in[mz_col].round(mz_bin).astype(str) + "_" + df_in[rt_col].round(2).astype(str)
                pivot = df_in.pivot_table(index='T_ID', columns=sample_col, values=in_col, aggfunc='mean').fillna(0)
                cleaned = pivot[(pivot != 0).sum(axis=1) >= (min_pres/100)*len(pivot.columns)]
                if cleaned.empty: st.error("No features left. Lower Min Presence."); st.stop()
                
                min_v = (cleaned[cleaned > 0].min().min()) / 2
                X_raw = cleaned.replace(0, min_v).div(cleaned.sum(axis=0), axis=1).T * 1000000
                
                # 2. GROUP ASSIGNMENT
                if df_meta is not None:
                    meta_indexed = df_meta.set_index(meta_link)
                    aligned_meta = meta_indexed.reindex(X_raw.index)
                    groups = aligned_meta[group_col_name].astype(str).tolist()
                else:
                    groups = [str(s).split(actual_delim)[0] for s in X_raw.index]
                
                unique_g = sorted(list(set(groups)))
                y_bin = [1 if g == unique_g[-1] else 0 for g in groups]

                # 3. MATH & SCALING
                X_z = (X_raw - X_raw.mean()) / X_raw.std()
                X_p = (X_raw - X_raw.mean()) / np.sqrt(X_raw.std().replace(0, np.nan))
                X_scaled = X_p if scaling == "Pareto Scaling" else (X_z if "Auto" in scaling else X_raw)
                X_scaled = X_scaled.fillna(0).loc[:, X_scaled.std() > 0]

                # 4. STATS & ML
                stats_ready = False
                if len(unique_g) >= 2:
                    g1_mask, g2_mask = [g == unique_g[0] for g in groups], [g == unique_g[1] for g in groups]
                    _, pvals = ttest_ind(X_raw.iloc[g1_mask], X_raw.iloc[g2_mask], axis=0)
                    log2fc = np.log2(X_raw.iloc[g2_mask].mean() / X_raw.iloc[g1_mask].mean().replace(0, 0.001))
                    stats_df = pd.DataFrame({'ID': X_raw.columns, 'p': pvals, 'Log2FC': log2fc}).fillna(0)
                    stats_df['Sig'] = (stats_df['p'] < 0.05) & (abs(stats_df['Log2FC']) > 1)
                    hits = stats_df[stats_df['Sig']].copy()
                    
                    rf = RandomForestClassifier(n_estimators=100, random_state=42)
                    acc = cross_val_score(rf, X_scaled, y_bin, cv=min(3, len(X_raw))).mean()
                    
                    pls = PLSRegression(n_components=2).fit(X_scaled, y_bin)
                    stats_ready = True

                # 5. TABS FOR OUTPUT
                t1, t2, t3, t4, t5, t6, t7 = st.tabs(["📊 Distribution", "🔵 PCA", "🎯 PLS-DA", "🌋 Volcano", "🏆 Identification", "🔗 Export Hub", "📋 Summary"])
                
                with t1:
                    st.subheader("Intensity Distribution QC")
                    cols_dist = st.columns(3)
                    cols_dist[0].plotly_chart(px.box(X_raw.melt(), y='value', title="Raw Scale"), use_container_width=True)
                    cols_dist[1].plotly_chart(px.box(X_z.melt(), y='value', title="Z-score Scaled"), use_container_width=True)
                    cols_dist[2].plotly_chart(px.box(X_p.melt(), y='value', title="Pareto Scaled"), use_container_width=True)

                with t2:
                    st.subheader("Multivariate Analysis Comparison")
                    pca_cols = st.columns(2)
                    scaling_types = [("Raw", X_raw), ("Pareto", X_p), ("Z-score", X_z)] # Removed Log10 as it can distort variance
                    for i, (name, data) in enumerate(scaling_types):
                        coords = PCA(n_components=2).fit_transform(data.fillna(0))
                        pca_cols[i%2].plotly_chart(px.scatter(x=coords[:,0], y=coords[:,1], color=groups, title=f"PCA: {name}"), use_container_width=True)

                with t3:
                    if stats_ready:
                        st.subheader("Supervised Discriminant Analysis")
                        pl_cols = st.columns(2)
                        pl1 = PLSRegression(n_components=2).fit(X_scaled, y_bin)
                        pl1_scores, pl1_loadings = pl1.x_scores_, pl1.x_loadings_
                        pl1_cols[0].plotly_chart(px.scatter(x=pls.x_scores_[:,0], y=pls.x_scores_[:,1], color=groups, title="PLS-DA Scores"), use_container_width=True)
                        pl1_cols[1].plotly_chart(px.scatter(x=pls.x_loadings_[:,0], y=pls.x_loadings_[:,1], title="PLS-DA Loadings"), use_container_width=True)
                    else: st.warning("Supervised analysis requires at least 2 groups.")

                with t4:
                    if stats_ready:
                        st.subheader("Biomarker Significance Plot")
                        st.plotly_chart(px.scatter(stats_df, x='Log2FC', y=-np.log10(stats_df['p']+1e-10), color='Sig', hover_name='ID', title="Volcano Plot"), use_container_width=True)

                with t5:
                    if stats_ready and not hits.empty:
                        st.subheader("Candidate Identification & Confidence")
                        hits['m/z'] = hits['ID'].apply(lambda x: float(x.split('_')[0]))
                        hits['Neutral Mass'] = (hits['m/z'] + 1.0078) if ion_mode.startswith("Neg") else (hits['m/z'] - 1.0078)
                        hits['Identify'] = hits['m/z'].apply(lambda x: f"https://pubchem.ncbi.nlm.nih.gov/#query={x}")
                        
                        def score_confidence(row):
                            lv = 3 # Default: Statistically Significant
                            if row['Neutral Mass'] < 500: lv = 2 # Drug-like MW
                            return f"Level {lv}"
                        
                        hits['Confidence'] = hits.apply(score_confidence, axis=1)
                        st.dataframe(hits[['ID', 'Neutral Mass', 'Confidence', 'Log2FC', 'p', 'Identify']], column_config={"Identify": st.column_config.LinkColumn()}, use_container_width=True)
                    else: st.info("No significant biomarkers found with current settings.")

                with t6:
                    st.subheader("Data Export for External Platforms")
                    gnps_q = X_raw.T.copy().reset_index()
                    gnps_q.insert(0, 'row ID', range(1, len(gnps_q) + 1))
                    gnps_q.insert(1, 'row m/z', gnps_q['ID'].apply(lambda x: x.split('_')[0]))
                    gnps_q.insert(2, 'row RT', gnps_q['ID'].apply(lambda x: x.split('_')[1]))
                    
                    c_e1, c_e2 = st.columns(2)
                    with c_e1:
                        st.download_button("GNPS Feature Table", gnps_q.to_csv(index=False).encode('utf-8'), "GNPS_quant.csv")
                        if df_meta is not None:
                            gnps_meta = pd.DataFrame({'filename': X_raw.index, 'ATTRIBUTE_Group': groups})
                            st.download_button("GNPS Metadata", gnps_meta.to_csv(index=False, sep='\t').encode('utf-8'), "GNPS_metadata.txt")
                    with c_e2:
                        if stats_ready and not hits.empty:
                            st.download_button("SIRIUS Mass List", hits[['m/z', 'Log2FC', 'p']].to_csv(index=False).encode('utf-8'), "sirius_masslist.csv")
                
                with t7:
                    st.metric("ML Validation Accuracy", f"{acc:.1%}")
                    pdf_data = create_academic_report(unique_g[0], unique_g[1], len(cleaned), acc, len(hits[hits['Confidence']=='Level 2']))
                    st.download_button("Download Research Summary (PDF)", pdf_data, "Metabo_Report.pdf", "application/pdf")
                
                st.balloons()
            except Exception as e: st.error(f"Analysis Error: {e}")

st.caption("Internal Research Dashboard | Authorized Lab Use Only")

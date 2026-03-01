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

# --- SIDEBAR ---
st.sidebar.title("TLL Metabo-Discovery")
st.sidebar.info("**Bioinformatics Hub**\nAutomated Metadata Alignment Enabled.")
st.sidebar.markdown("---")
st.sidebar.markdown("[GNPS Web Platform](https://gnps.ucsd.edu/)")
st.sidebar.markdown("[SIRIUS Desktop App](https://bright.cs.uni-jena.de/sirius/)")

# --- PDF GENERATOR ---
def create_academic_report(g1, g2, feat_count, accuracy, leads_count):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "TLL Metabo-Discovery: Research Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 12); pdf.cell(0, 10, "1. Analysis Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    summary = (f"Comparison: {g1} vs {g2}\n"
               f"Features Processed: {feat_count}\n"
               f"ML Validation Accuracy: {accuracy:.1%}\n"
               f"Biomarkers Identified: {leads_count}")
    pdf.multi_cell(0, 10, summary)
    return bytes(pdf.output())

st.title("TLL Metabo-Discovery: Professional Analytics Suite")
st.markdown("---")

mode = st.radio("Select Module:", ("Raw Data Extraction", "Discovery Dashboard"))

# ============================================
# MODULE 1: RAW DATA PROCESSOR
# ============================================
if mode == "Raw Data Extraction":
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
        st.success("Complete."); st.download_button("Download CSV", pd.concat(all_features).to_csv(index=False).encode('utf-8'), "combined_features.csv")

# ============================================
# MODULE 2: DISCOVERY DASHBOARD
# ============================================
else:
    c_u1, c_u2 = st.columns(2)
    with c_u1: uploaded_file = st.file_uploader("1. Data Table (.csv)", type=["csv"])
    with c_u2: metadata_file = st.file_uploader("2. Metadata (Optional)", type=["csv", "txt"])

    if uploaded_file:
        df_in = pd.read_csv(uploaded_file)
        df_meta = None
        
        # --- IMPROVED METADATA LOADING ---
        if metadata_file:
            sep = '\t' if metadata_file.name.endswith('.txt') else ','
            df_meta = pd.read_csv(metadata_file, sep=sep)
            # CLEANING: Strip extensions and spaces from metadata filenames
            for col in df_meta.columns:
                if df_meta[col].dtype == 'object':
                    df_meta[col] = df_meta[col].astype(str).str.replace('.mzML', '', case=False).str.replace('.csv', '', case=False).str.strip()

        with st.expander("Parameters & Grouping Configuration"):
            col1, col2, col3, col4 = st.columns(4)
            mz_col = col1.selectbox("m/z Column", df_in.columns, index=0)
            rt_col = col2.selectbox("RT Column", df_in.columns, index=1 if "RT_min" not in df_in.columns else df_in.columns.get_loc("RT_min"))
            sample_col = col3.selectbox("Sample ID", df_in.columns, index=df_in.columns.get_loc("Sample") if "Sample" in df_in.columns else 0)
            in_col = col4.selectbox("Intensity Column", df_in.columns, index=df_in.columns.get_loc("Intensity") if "Intensity" in df_in.columns else 2)
            
            st.markdown("---")
            if df_meta is not None:
                m1, m2 = st.columns(2)
                meta_link = m1.selectbox("Link Metadata via:", df_meta.columns, index=0)
                group_col_name = m2.selectbox("Grouping column:", df_meta.columns, index=1 if len(df_meta.columns)>1 else 0)
            else:
                m1, m2 = st.columns(2)
                delimiter = m1.selectbox("Sample Name Delimiter", ["_", "-", ".", "Space"], index=0)
                actual_delim = " " if delimiter == "Space" else delimiter

            f1, f2, f3, f4 = st.columns(4)
            mz_bin = f1.slider("Alignment Tolerance", 1, 5, 3)
            min_pres = f2.slider("Min Presence %", 0, 100, 80)
            ion_mode = f3.selectbox("Polarity", ["Negative [M-H]-", "Positive [M+H]+"])
            scaling = f4.selectbox("Scaling", ["Pareto Scaling", "Auto-Scaling", "None"])

        # --- DYNAMIC GROUP PREVIEW ---
        sample_list = df_in[sample_col].unique()
        if df_meta is not None:
            meta_indexed = df_meta.set_index(meta_link)
            detected_groups = meta_indexed.reindex(sample_list)[group_col_name].dropna().unique()
        else:
            detected_groups = sorted(list(set([str(s).split(actual_delim)[0] for s in sample_list])))
        
        st.write(f"🔍 **Groups Found:** {', '.join(map(str, detected_groups))}")

        if st.button("Run Full Analytics Pipeline"):
            try:
                # --- A. PREPROCESSING ---
                df_in['ID'] = df_in[mz_col].round(mz_bin).astype(str) + "_" + df_in[rt_col].round(2).astype(str)
                pivot = df_in.pivot_table(index='ID', columns=sample_col, values=in_col, aggfunc='mean').fillna(0)
                cleaned = pivot[(pivot != 0).sum(axis=1) >= (min_pres/100)*len(pivot.columns)]
                if cleaned.empty: st.error("No features left. Lower Min Presence."); st.stop()
                
                min_v = (cleaned[cleaned > 0].min().min()) / 2
                X_raw = cleaned.replace(0, min_v).div(cleaned.sum(axis=0), axis=1).T * 1000000

                # --- B. GROUP ASSIGNMENT ---
                if df_meta is not None:
                    groups = meta_indexed.reindex(X_raw.index)[group_col_name].astype(str).tolist()
                else:
                    groups = [str(s).split(actual_delim)[0] for s in X_raw.index]
                
                unique_g = sorted(list(set(groups)))
                if len(unique_g) < 2:
                    st.error("Error: Only one group detected. Check if your metadata 'Link' column matches your data 'Sample ID' exactly.")
                    st.stop()
                
                y_labels = [1 if g == unique_g[-1] else 0 for g in groups]

                # --- C. MATH ---
                X_z = (X_raw - X_raw.mean()) / X_raw.std()
                X_p = (X_raw - X_raw.mean()) / np.sqrt(X_raw.std().replace(0, np.nan))
                X_scaled = X_p if scaling == "Pareto Scaling" else (X_z if "Auto" in scaling else X_raw)
                X_scaled = X_scaled.fillna(0).loc[:, X_scaled.std() > 0]

                # --- D. STATS ---
                g1_mask, g2_mask = [g == unique_g[0] for g in groups], [g == unique_g[1] for g in groups]
                _, pvals = ttest_ind(X_raw.iloc[g1_mask], X_raw.iloc[g2_mask], axis=0)
                log2fc = np.log2(X_raw.iloc[g2_mask].mean() / X_raw.iloc[g1_mask].mean().replace(0, 0.001))
                stats_df = pd.DataFrame({'ID': X_raw.columns, 'p': pvals, 'Log2FC': log2fc}).fillna(0)
                stats_df['Sig'] = (stats_df['p'] < 0.05) & (abs(stats_df['Log2FC']) > 1)
                hits = stats_df[stats_df['Sig']].copy()
                
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                acc = cross_val_score(rf, X_scaled, y_labels, cv=min(3, len(X_raw))).mean()
                pls = PLSRegression(n_components=2).fit(X_scaled, y_labels)
                pdf_report = create_academic_report(unique_g[0], unique_g[1], len(cleaned), acc, len(hits))

                # --- E. RENDERING ---
                t1, t2, t3, t4, t5, t6, t7 = st.tabs(["📊 Distribution", "🔵 PCA Gallery", "🎯 PLS-DA Suite", "🌋 Volcano Plot", "🏆 Identification", "🔗 Export Hub", "📋 Final Report"])
                
                with t1:
                    st.subheader("Normalization QC")
                    cols_box = st.columns(2)
                    cols_box[0].plotly_chart(px.box(X_raw.melt(), y='value', title="TIC Normalized"))
                    cols_box[1].plotly_chart(px.box(X_p.melt(), y='value', title="Pareto Scaled"))

                with t2:
                    pca_coords = PCA(n_components=2).fit_transform(X_scaled.fillna(0))
                    st.plotly_chart(px.scatter(x=pca_coords[:,0], y=pca_coords[:,1], color=groups, title=f"PCA: {scaling}"), use_container_width=True)

                with t3:
                    c_p1, c_p2 = st.columns(2)
                    c_p1.plotly_chart(px.scatter(x=pls.x_scores_[:,0], y=pls.x_scores_[:,1], color=groups, title="PLS-DA Scores"))
                    c_p2.plotly_chart(px.scatter(x=pls.x_loadings_[:,0], y=pls.x_loadings_[:,1], title="PLS-DA Loadings", opacity=0.4))

                with t4:
                    st.plotly_chart(px.scatter(stats_df, x='Log2FC', y=-np.log10(stats_df['p']+1e-10), color='Sig', hover_name='ID', title="Volcano Discovery Plot"))

                with t5:
                    if not hits.empty:
                        hits['m/z'] = hits['ID'].apply(lambda x: float(x.split('_')[0]))
                        hits['Neutral Mass'] = (hits['m/z'] + 1.0078) if ion_mode.startswith("Neg") else (hits['m/z'] - 1.0078)
                        hits['PubChem'] = hits['m/z'].apply(lambda x: f"https://pubchem.ncbi.nlm.nih.gov/#query={x}")
                        st.dataframe(hits[['ID', 'Neutral Mass', 'Log2FC', 'p', 'PubChem']], column_config={"PubChem": st.column_config.LinkColumn()})
                    else: st.info("No significant features identified.")

                with t6:
                    gnps_q = X_raw.T.copy().reset_index()
                    gnps_q.insert(0, 'row ID', range(1, len(gnps_q) + 1))
                    gnps_q.insert(1, 'row m/z', gnps_q['ID'].apply(lambda x: x.split('_')[0]))
                    gnps_q.insert(2, 'row RT', gnps_q['ID'].apply(lambda x: x.split('_')[1]))
                    st.download_button("📥 Download GNPS Table", gnps_q.to_csv(index=False).encode('utf-8'), "gnps.csv")
                    if not hits.empty: st.download_button("📥 Download SIRIUS List", hits[['m/z', 'Log2FC', 'p']].to_csv(index=False).encode('utf-8'), "sirius.csv")

                with t7:
                    st.metric("Model Accuracy", f"{acc:.1%}")
                    st.download_button("📥 Download Report (PDF)", pdf_report, "Report.pdf", "application/pdf")
                
                st.balloons()
            except Exception as e: st.error(f"Analysis Error: {e}")

st.caption("TLL Lab Discovery Hub | Yusuf Bioinformatics")

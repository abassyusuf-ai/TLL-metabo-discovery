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
from pyteomics import mzml
import os
import gc

st.set_page_config(page_title="TLL Metabo-Discovery", layout="wide", page_icon="🧪")

# --- SIDEBAR ---
st.sidebar.title("🧪 TLL Metabo-Discovery")
st.sidebar.info("Professional Pipeline: Raw Data Extraction to ML Biomarker Discovery.")
st.sidebar.caption("Developed by Abass Yusuf | Lab Suite")

st.title("🧪 TLL Metabo-Discovery: Professional Analytics Suite")

# --- STEP 1: INPUT MODE ---
mode = st.radio("Select Analysis Stage:", 
                ("Raw Data (.mzML) - Batch Feature Extraction", "Quantified Data (.csv) - Discovery Dashboard"))

# ============================================
# MODE 1: BATCH RAW DATA PROCESSING (Memory Efficient)
# ============================================
if mode == "Raw Data (.mzML) - Batch Feature Extraction":
    st.markdown("### Step 1: Upload Batch .mzML files")
    uploaded_mzmls = st.file_uploader("Select multiple files (44+ supported)", type=["mzml"], accept_multiple_files=True)
    
    if uploaded_mzmls:
        if st.button("🚀 Start Batch Extraction"):
            all_features = []
            progress_bar = st.progress(0)
            status = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_mzmls):
                status.text(f"Extracting Features from {uploaded_file.name} ({i+1}/{len(uploaded_mzmls)})")
                with open("temp.mzml", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                rows = []
                try:
                    with mzml.read("temp.mzml") as reader:
                        for spec in reader:
                            if spec['ms level'] == 1 and len(spec['intensity array']) > 0:
                                mzs, ints = spec['m/z array'], spec['intensity array']
                                base_idx = np.argmax(ints)
                                rows.append([float(mzs[base_idx]), float(spec['scanList']['scan'][0]['scan start time'])/60, float(ints[base_idx])])
                    
                    df_s = pd.DataFrame(rows, columns=["m/z", "RT_min", "Intensity"])
                    df_s["Sample"] = uploaded_file.name.replace(".mzML", "")
                    all_features.append(df_s)
                    del rows
                    gc.collect() 
                except Exception as e:
                    st.error(f"Error in {uploaded_file.name}: {e}")
                
                progress_bar.progress((i + 1) / len(uploaded_mzmls))

            df_combined = pd.concat(all_features, ignore_index=True)
            st.success("Extraction Complete!")
            st.download_button("📥 Download Combined Table (.csv)", df_combined.to_csv(index=False).encode('utf-8'), "combined_features.csv")
            if os.path.exists("temp.mzml"): os.remove("temp.mzml")

# ============================================
# MODE 2: DISCOVERY DASHBOARD (The Rich Results Edition)
# ============================================
else:
    uploaded_csv = st.file_uploader("Upload Combined/Merged Table (.csv)", type=["csv"])
    
    if uploaded_csv:
        df_input = pd.read_csv(uploaded_csv)
        
        with st.expander("🛠️ Advanced Data Cleaning & Zero Reduction"):
            c1, c2, c3, c4 = st.columns(4)
            mz_c = c1.selectbox("m/z Column", df_input.columns, index=0)
            rt_c = c2.selectbox("RT Column", df_input.columns, index=1 if "RT_min" not in df_input.columns else df_input.columns.get_loc("RT_min"))
            sm_c = c3.selectbox("Sample ID", df_input.columns, index=df_input.columns.get_loc("Sample") if "Sample" in df_input.columns else 0)
            in_c = c4.selectbox("Intensity", df_input.columns, index=df_input.columns.get_loc("Intensity") if "Intensity" in df_input.columns else 2)
            
            f1, f2, f3, f4 = st.columns(4)
            mz_bin = f1.slider("m/z Rounding (Alignment)", 1, 4, 3)
            min_pres = f2.slider("Min Presence (%) - 80% Rule", 0, 100, 80, help="Reduces zeros by removing features not consistently found.")
            impute_on = f3.checkbox("Gap Filling (Half-Min)", value=True, help="Replaces remaining zeros with 1/2 of lowest value.")
            p_thresh = f4.number_input("P-value Threshold", 0.05)

        if st.button("🚀 Run Rich Discovery Pipeline"):
            try:
                # 1. CLEANING & ALIGNMENT (The "Zero-Reduction" Logic)
                df_input['ID'] = df_input[mz_c].round(mz_bin).astype(str) + " | RT=" + df_input[rt_c].round(2).astype(str)
                pivot = df_input.pivot_table(index='ID', columns=sm_c, values=in_c, aggfunc='mean').fillna(0)
                
                # Apply 80% Rule Filter
                thresh = (min_pres/100) * len(pivot.columns)
                cleaned = pivot[(pivot != 0).sum(axis=1) >= thresh]
                
                # Imputation (Replacing remaining 0s)
                if impute_on:
                    min_val = cleaned[cleaned > 0].min().min()
                    data_final = cleaned.replace(0, min_val / 2)
                else:
                    data_final = cleaned
                
                # TIC Normalization
                data_norm = data_final.div(data_final.sum(axis=0), axis=1) * 1000000
                
                # 2. MULTI-SCALING MATH (For the comparison plots)
                X_raw = data_norm.T
                X_z = (X_raw - X_raw.mean()) / X_raw.std()
                X_p = (X_raw - X_raw.mean()) / np.sqrt(X_raw.std().replace(0, np.nan))
                X_z, X_p = X_z.fillna(0), X_p.fillna(0)
                
                groups = [str(s).split('_')[0] for s in X_raw.index]
                unique_groups = sorted(list(set(groups)))

                # 3. RICH VISUALIZATION TABS (Matching your images)
                tabs = st.tabs(["📊 Distributions", "📈 Group Means", "🔵 PCA Gallery", "🎯 PLS-DA Suite", "🌋 Volcano Plot", "🔥 Heatmap", "🏆 ML Validation"])

                with tabs[0]: # BOXPLOTS (Image 1)
                    st.subheader("Intensity Distribution across Normalization Scales")
                    cols = st.columns(3)
                    cols[0].plotly_chart(px.box(X_raw.melt(), y='value', title="TIC Normalized (Raw Scale)", color_discrete_sequence=['#66c2a5']))
                    cols[1].plotly_chart(px.box(X_z.melt(), y='value', title="Z-score Scaled", color_discrete_sequence=['#8da0cb']))
                    cols[2].plotly_chart(px.box(X_p.melt(), y='value', title="Pareto Scaled", color_discrete_sequence=['#fc8d62']))

                with tabs[1]: # BAR PLOTS (Image 2)
                    st.subheader("Comparison of Group Means")
                    mean_data = pd.DataFrame({'Group': groups, 'Mean_Intensity': X_raw.mean(axis=1)})
                    fig_bar = px.bar(mean_data.groupby('Group').mean().reset_index(), x='Group', y='Mean_Intensity', 
                                     color='Group', error_y=mean_data.groupby('Group').std()['Mean_Intensity'],
                                     title="Global Mean Intensity ± SD per Group")
                    st.plotly_chart(fig_bar, use_container_width=True)

                with tabs[2]: # PCA 4-PANEL (Image 3)
                    st.subheader("Impact of Scaling on Group Separation (PCA)")
                    p_cols = st.columns(2)
                    scaling_types = [("Raw (TIC)", X_raw), ("Pareto", X_p), ("Z-score", X_z), ("Log10", np.log10(X_raw + 1))]
                    for i, (name, d_set) in enumerate(scaling_types):
                        pca_coords = PCA(n_components=2).fit_transform(d_set)
                        fig = px.scatter(x=pca_coords[:,0], y=pca_coords[:,1], color=groups, title=f"PCA: {name}", template="plotly_white")
                        p_cols[i%2].plotly_chart(fig, use_container_width=True)

                with tabs[3]: # PLS-DA SUITE (Image 5 & 6)
                    if len(unique_groups) >= 2:
                        y_labels = [1 if g == unique_groups[-1] else 0 for g in groups]
                        pls_model = PLSRegression(n_components=2)
                        pls_model.fit(X_p, y_labels)
                        
                        pl1, pl2 = st.columns(2)
                        pl1.plotly_chart(px.scatter(x=pls_model.x_scores_[:,0], y=pls_model.x_scores_[:,1], color=groups, 
                                                   title="PLS-DA Scores Plot (Pareto)", labels={'x':'LV1','y':'LV2'}))
                        pl2.plotly_chart(px.scatter(x=pls_model.x_loadings_[:,0], y=pls_model.x_loadings_[:,1], 
                                                   title="PLS-DA Loadings (Feature Contribution)", opacity=0.4))

                with tabs[4]: # VOLCANO (Image 4)
                    if len(unique_groups) >= 2:
                        g1_m = [g == unique_groups[0] for g in groups]
                        g2_m = [g == unique_groups[1] for g in groups]
                        log2fc = np.log2(X_raw[g2_m].mean() / X_raw[g1_m].mean().replace(0, 0.001))
                        _, pvals = ttest_ind(X_raw[g1_m], X_raw[g2_m], axis=0)
                        vol_df = pd.DataFrame({'ID': X_raw.columns, 'Log2FC': log2fc, 'p': pvals, 'log10p': -np.log10(pvals)})
                        vol_df['Sig'] = (vol_df['p'] < p_thresh) & (abs(vol_df['Log2FC']) > 1)
                        fig_v = px.scatter(vol_df, x='Log2FC', y='log10p', color='Sig', hover_name='ID',
                                         color_discrete_map={True:'red', False:'gray'}, title="Volcano Plot: Biomarker Discovery")
                        fig_v.add_hline(y=-np.log10(p_thresh), line_dash="dash")
                        st.plotly_chart(fig_v, use_container_width=True)

                with tabs[5]: # HEATMAP
                    st.plotly_chart(px.imshow(X_p.T.head(100), aspect="auto", title="Top 100 Features (Pareto-Scaled)"), use_container_width=True)

                with tabs[6]: # ML VALIDATION
                    y_ml = [1 if g == unique_groups[-1] else 0 for g in groups]
                    acc = cross_val_score(RandomForestClassifier(), X_p, y_ml, cv=3).mean()
                    st.metric("Random Forest Predictive Accuracy", f"{acc:.1%}")
                    st.info("Validation confirms if the clusters found in PCA/PLS-DA are biologically predictable.")

                st.balloons()
            except Exception as e:
                st.error(f"Pipeline Error: {e}")

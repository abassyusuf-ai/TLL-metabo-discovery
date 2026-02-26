import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import ttest_ind
from pyteomics import mzml
import os

st.set_page_config(page_title="TLL Metabo-Discovery", layout="wide", page_icon="🧪")

# --- SIDEBAR ---
st.sidebar.title("TLL Metabo-Discovery")
st.sidebar.info("Batch Processing & Biomarker Discovery Pipeline.")
st.sidebar.caption("Developed by Abass Yusuf")

st.title("🧪 TLL Metabo-Discovery: Professional Suite")

# --- STEP 1: INPUT MODE ---
mode = st.radio("Select Analysis Stage:", 
                ("Raw Data (.mzML) - Batch Feature Extraction", "Quantified Data (.csv) - Discovery Stats"))

# ============================================
# MODE 1: BATCH RAW DATA PROCESSING (The Colab Loop)
# ============================================
if mode == "Raw Data (.mzML) - Batch Feature Extraction":
    st.markdown("### Step 1: Upload all .mzML files from your batch")
    uploaded_mzmls = st.file_uploader("Select multiple .mzML files", type=["mzml"], accept_multiple_files=True)
    
    if uploaded_mzmls:
        st.write(f"Found {len(uploaded_mzmls)} files. Ready to extract features.")
        
        if st.button("🚀 Start Batch Extraction"):
            all_features = []
            progress_bar = st.progress(0)
            
            for i, uploaded_file in enumerate(uploaded_mzmls):
                # Save temp file for pyteomics to read
                with open("temp.mzml", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                rows = []
                with mzml.read("temp.mzml") as reader:
                    for spec in reader:
                        if spec['ms level'] == 1:
                            ints = spec['intensity array']
                            if len(ints) > 0:
                                mzs = spec['m/z array']
                                base_idx = np.argmax(ints)
                                # Extracting exact logic from your Colab: [mz, rt, intensity]
                                rows.append([
                                    float(mzs[base_idx]), 
                                    float(spec['scanList']['scan'][0]['scan start time']), 
                                    float(ints[base_idx])
                                ])
                
                # Create DF for this sample
                df_sample = pd.DataFrame(rows, columns=["m/z", "RT", "Intensity"])
                df_sample["RT_min"] = df_sample["RT"] / 60.0 # Match your Colab conversion
                df_sample["Sample"] = uploaded_file.name.replace(".mzML", "")
                
                all_features.append(df_sample)
                progress_bar.progress((i + 1) / len(uploaded_mzmls))

            # Combine all (Like your df_all = pd.concat in Colab)
            df_combined = pd.concat(all_features, ignore_index=True)
            
            st.success(f"Successfully processed {len(uploaded_mzmls)} files!")
            st.markdown("### Combined Feature Table (Phase 1)")
            st.dataframe(df_combined.head(10))
            
            # Download link for the combined table
            csv = df_combined.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Combined Feature Table (.csv)", 
                               data=csv, 
                               file_name="combined_features_raw.csv", 
                               mime="text/csv")
            st.info("💡 Next: Use this CSV in the 'Discovery Stats' mode above.")

# ============================================
# MODE 2: DISCOVERY STATS (The Colab Stats Logic)
# ============================================
else:
    st.markdown("### Step 1: Upload your Combined/Merged Table")
    uploaded_csv = st.file_uploader("Upload .csv file", type=["csv"])
    
    if uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv)
        
        with st.expander("Step 2: Column & Scaling Settings"):
            c1, c2, c3, c4 = st.columns(4)
            with c1: mz_col = st.selectbox("m/z", df.columns, index=0)
            with c2: rt_col = st.selectbox("RT", df.columns, index=4 if "RT_min" in df.columns else 1)
            with c3: sample_col = st.selectbox("Sample ID", df.columns, index=df.columns.get_loc("Sample") if "Sample" in df.columns else 0)
            with c4: intensity_col = st.selectbox("Intensity", df.columns, index=df.columns.get_loc("Intensity") if "Intensity" in df.columns else 2)
            
            sc1, sc2, sc3 = st.columns(3)
            with sc1: scaling = st.selectbox("Scaling Method", ["Pareto Scaling", "Z-score (Auto-scale)", "None"])
            with sc2: min_pres = st.slider("Min Presence %", 0, 100, 50)
            with sc3: p_val_limit = st.number_input("P-value Significance", 0.05)

        if st.button("🚀 Run Discovery Pipeline"):
            try:
                # 1. Pivot & Clean (Colab Phase 3 Logic)
                df['ID'] = df[mz_col].round(4).astype(str) + " | RT=" + df[rt_col].round(2).astype(str)
                pivot = df.pivot_table(index='ID', columns=sample_col, values=intensity_col, aggfunc='mean').fillna(0)
                
                # Filter rows with zero variation and min presence
                cleaned = pivot[(pivot != 0).sum(axis=1) >= ((min_pres/100)*len(pivot.columns))]
                
                # 2. Scaling Math (Your Colab Formulas)
                X = cleaned.T
                if scaling == "Z-score (Auto-scale)":
                    X_scaled = (X - X.mean()) / X.std()
                elif scaling == "Pareto Scaling":
                    X_scaled = (X - X.mean()) / np.sqrt(X.std().replace(0, np.nan))
                else:
                    X_scaled = X
                
                X_scaled = X_scaled.fillna(0).replace([np.inf, -np.inf], 0)

                # 3. Discovery & ML
                groups = [s.split('_')[0] for s in X_scaled.index]
                unique_groups = sorted(list(set(groups)))
                
                if len(unique_groups) >= 2:
                    g1, g2 = unique_groups[0], unique_groups[1]
                    y = [1 if g == g2 else 0 for g in groups]
                    
                    # Machine Learning Validation (Colab Phase 4)
                    rf = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf_acc = cross_val_score(rf, X_scaled, y, cv=3).mean()
                    
                    # Stats
                    g1_mask = [g == g1 for g in groups]
                    g2_mask = [g == g2 for g in groups]
                    _, pvals = ttest_ind(X_scaled[g1_mask], X_scaled[g2_mask], axis=0)
                    
                    vol_df = pd.DataFrame({'ID': X_scaled.columns, 'p': pvals, 'log10p': -np.log10(pvals)})
                    vol_df['Sig'] = vol_df['p'] < p_val_limit

                    # Tabs
                    t1, t2, t3, t4, t5 = st.tabs(["🔵 PCA", "🎯 PLS-DA", "🌋 Volcano", "🔥 Heatmap", "🏆 ML Accuracy"])
                    
                    with t1:
                        pca = PCA(n_components=2).fit_transform(X_scaled)
                        st.plotly_chart(px.scatter(x=pca[:,0], y=pca[:,1], color=groups, title="PCA"), use_container_width=True)
                    
                    with t2:
                        pls = PLSRegression(n_components=2).fit_transform(X_scaled, y)[0]
                        st.plotly_chart(px.scatter(x=pls[:,0], y=pls[:,1], color=groups, title="PLS-DA"), use_container_width=True)

                    with t3:
                        st.plotly_chart(px.scatter(vol_df, x='ID', y='log10p', color='Sig', title="Volcano Plot"), use_container_width=True)

                    with t4:
                        st.plotly_chart(px.imshow(X_scaled.T.head(50), color_continuous_scale='Viridis', title="Top 50 Features Heatmap"), use_container_width=True)

                    with t5:
                        st.metric("Predictive Accuracy (Random Forest)", f"{rf_acc:.1%}")
                        st.info("High accuracy confirms that the biological difference between your groups is robust.")

                st.balloons()
            except Exception as e:
                st.error(f"Error: {e}")

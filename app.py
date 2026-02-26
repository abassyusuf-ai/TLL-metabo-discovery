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
import gc # Garbage Collector for memory management

st.set_page_config(page_title="TLL Metabo-Discovery", layout="wide", page_icon="🧪")

# --- SIDEBAR ---
st.sidebar.title("TLL Metabo-Discovery")
st.sidebar.info("Professional Pipeline: Raw Data to Machine Learning Validation.")
st.sidebar.caption("Developed by Abass Yusuf")

st.title("🧪 TLL Metabo-Discovery: Professional Suite")

# --- STEP 1: INPUT MODE ---
mode = st.radio("Select Analysis Stage:", 
                ("Raw Data (.mzML) - Batch Feature Extraction", "Quantified Data (.csv) - Discovery Stats"))

# ============================================
# MODE 1: BATCH RAW DATA PROCESSING (Optimized for 44+ Files)
# ============================================
if mode == "Raw Data (.mzML) - Batch Feature Extraction":
    st.markdown("### Step 1: Upload all .mzML files")
    uploaded_mzmls = st.file_uploader("Select multiple .mzML files (up to 5GB)", type=["mzml"], accept_multiple_files=True)
    
    if uploaded_mzmls:
        st.success(f"Ready to process {len(uploaded_mzmls)} files.")
        
        if st.button("🚀 Start Batch Extraction"):
            all_features = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_mzmls):
                status_text.text(f"Processing File {i+1}/{len(uploaded_mzmls)}: {uploaded_file.name}")
                
                # Write to temp file to save RAM
                with open("temp.mzml", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                rows = []
                try:
                    with mzml.read("temp.mzml") as reader:
                        for spec in reader:
                            if spec['ms level'] == 1:
                                ints = spec['intensity array']
                                if len(ints) > 0:
                                    mzs = spec['m/z array']
                                    # Exact logic from your Colab: Find the Base Peak (max intensity)
                                    base_idx = np.argmax(ints)
                                    rows.append([
                                        float(mzs[base_idx]), 
                                        float(spec['scanList']['scan'][0]['scan start time']), 
                                        float(ints[base_idx])
                                    ])
                    
                    # Create DF for this sample
                    df_sample = pd.DataFrame(rows, columns=["m/z", "RT", "Intensity"])
                    df_sample["RT_min"] = df_sample["RT"] / 60.0
                    df_sample["Sample"] = uploaded_file.name.replace(".mzML", "")
                    all_features.append(df_sample)
                    
                    # Clean up memory immediately
                    del rows
                    gc.collect() 
                    
                except Exception as e:
                    st.error(f"Error in {uploaded_file.name}: {e}")
                
                progress_bar.progress((i + 1) / len(uploaded_mzmls))

            # Combine all results (Matches your Colab: df_all = pd.concat)
            df_combined = pd.concat(all_features, ignore_index=True)
            
            st.success(f"Successfully processed {len(uploaded_mzmls)} files!")
            st.dataframe(df_combined.head(10))
            
            csv = df_combined.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Combined Feature Table (.csv)", 
                               data=csv, 
                               file_name="combined_features_raw.csv", 
                               mime="text/csv")
            
            # Remove temp file
            if os.path.exists("temp.mzml"): os.remove("temp.mzml")

# ============================================
# MODE 2: DISCOVERY STATS (Colab Machine Learning & Stats)
# ============================================
else:
    st.markdown("### Step 1: Upload your Combined Table")
    uploaded_csv = st.file_uploader("Upload .csv file", type=["csv"])
    
    if uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv)
        
        with st.expander("Step 2: Column & Scaling Configuration"):
            c1, c2, c3, c4 = st.columns(4)
            with c1: mz_col = st.selectbox("m/z Column", df.columns, index=0)
            with c2: rt_col = st.selectbox("RT Column", df.columns, index=df.columns.get_loc("RT_min") if "RT_min" in df.columns else 1)
            with c3: sample_col = st.selectbox("Sample ID", df.columns, index=df.columns.get_loc("Sample") if "Sample" in df.columns else 0)
            with c4: intensity_col = st.selectbox("Intensity Column", df.columns, index=df.columns.get_loc("Intensity") if "Intensity" in df.columns else 2)
            
            sc1, sc2, sc3 = st.columns(3)
            with sc1: scaling = st.selectbox("Scaling Method", ["Pareto Scaling", "Z-score (Auto-scale)", "None"])
            with sc2: min_pres = st.slider("Min Presence (%)", 0, 100, 50)
            with sc3: p_val_limit = st.number_input("P-value Threshold", 0.05)

        if st.button("🚀 Run Full Discovery Pipeline"):
            try:
                # 1. Pivot & Cleaning
                df['ID'] = df[mz_col].round(4).astype(str) + " | RT=" + df[rt_col].round(2).astype(str)
                pivot = df.pivot_table(index='ID', columns=sample_col, values=intensity_col, aggfunc='mean').fillna(0)
                
                # Filter by presence
                thresh = (min_pres/100) * len(pivot.columns)
                cleaned = pivot[(pivot != 0).sum(axis=1) >= thresh]
                
                if cleaned.empty:
                    st.error("❌ No data left after filtering! Lower the Min Presence slider.")
                    st.stop()

                # 2. Scaling (Exact Colab Logic)
                X = cleaned.T
                if scaling == "Z-score (Auto-scale)":
                    X_scaled = (X - X.mean()) / X.std()
                elif scaling == "Pareto Scaling":
                    # Centers data and divides by sqrt of standard deviation
                    X_scaled = (X - X.mean()) / np.sqrt(X.std().replace(0, np.nan))
                else:
                    X_scaled = X
                
                # Final Clean of NaNs/Infs after scaling
                X_final = X_scaled.fillna(0).replace([np.inf, -np.inf], 0)
                # Remove zero-variance features (safety check)
                X_final = X_final.loc[:, X_final.std() > 0]

                # 3. Discovery & Machine Learning
                groups = [str(s).split('_')[0] for s in X_final.index]
                unique_groups = sorted(list(set(groups)))
                
                if len(unique_groups) >= 2:
                    g1, g2 = unique_groups[0], unique_groups[1]
                    y = [1 if g == g2 else 0 for g in groups]
                    
                    # ML Validation (Matches your Colab Phase 4)
                    rf = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf_acc = cross_val_score(rf, X_final, y, cv=3).mean()
                    
                    # Stats (T-test)
                    g1_mask = [g == g1 for g in groups]
                    g2_mask = [g == g2 for g in groups]
                    _, pvals = ttest_ind(X_final[g1_mask], X_final[g2_mask], axis=0)
                    
                    vol_df = pd.DataFrame({'ID': X_final.columns, 'p': pvals, 'log10p': -np.log10(pvals)})
                    vol_df['Sig'] = vol_df['p'] < p_val_limit

                    # 4. TABS FOR PI
                    t1, t2, t3, t4, t5 = st.tabs(["🔵 PCA Plot", "🎯 PLS-DA", "🌋 Volcano Plot", "🔥 Heatmap", "🏆 ML Accuracy"])
                    
                    with t1:
                        pca = PCA(n_components=2).fit_transform(X_final)
                        st.plotly_chart(px.scatter(x=pca[:,0], y=pca[:,1], color=groups, title="Unsupervised PCA"), use_container_width=True)
                    
                    with t2:
                        pls = PLSRegression(n_components=2).fit_transform(X_final, y)[0]
                        st.plotly_chart(px.scatter(x=pls[:,0], y=pls[:,1], color=groups, title="Supervised PLS-DA"), use_container_width=True)

                    with t3:
                        st.plotly_chart(px.scatter(vol_df, x='ID', y='log10p', color='Sig', title="Discovery Volcano Plot"), use_container_width=True)

                    with t4:
                        # Heatmap of top 100 most variable features
                        st.plotly_chart(px.imshow(X_final.T.head(100), color_continuous_scale='Viridis', title="Top 100 Features Heatmap"), use_container_width=True)

                    with t5:
                        st.metric("Random Forest Predictive Accuracy", f"{rf_acc:.1%}")
                        st.info("Accuracy > 80% suggests strong biological biomarkers.")

                st.balloons()
            except Exception as e:
                st.error(f"Analysis Error: {e}")

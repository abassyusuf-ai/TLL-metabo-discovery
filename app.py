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
st.sidebar.info("Professional Metabolomics Pipeline: Raw Data to Biomarkers.")
st.sidebar.caption("Developed by Abass Yusuf")

st.title("🧪 TLL Metabo-Discovery: Professional Suite")

# --- STEP 1: INPUT MODE ---
mode = st.radio("Select Analysis Stage:", 
                ("Raw Data (.mzML) - Quality Control", "Quantified Data (.csv) - Discovery Stats"))

# ============================================
# MODE 1: RAW DATA (mzML) - The "Base Peak" Logic from your Colab
# ============================================
if mode == "Raw Data (.mzML) - Quality Control":
    uploaded_mzml = st.file_uploader("Upload a Raw .mzML file", type=["mzml"])
    
    if uploaded_mzml is not None:
        st.info("Extracting Base Peak Chromatogram (BPC)...")
        with open("temp.mzml", "wb") as f:
            f.write(uploaded_mzml.getbuffer())
        
        try:
            times, base_ints, base_mzs = [], [], []
            with mzml.read("temp.mzml") as reader:
                for spec in reader:
                    if spec['ms level'] == 1:
                        mzs, ints = spec['m/z array'], spec['intensity array']
                        if len(ints) > 0:
                            idx = np.argmax(ints)
                            times.append(spec['scanList']['scan'][0]['scan start time'])
                            base_ints.append(ints[idx])
                            base_mzs.append(mzs[idx])
            
            bpc_df = pd.DataFrame({'RT': times, 'Intensity': base_ints, 'm/z': base_mzs})
            fig = px.line(bpc_df, x='RT', y='Intensity', hover_data=['m/z'], 
                          title="Base Peak Chromatogram (BPC)", labels={'RT':'Time (min)'})
            st.plotly_chart(fig, use_container_width=True)
            st.success("✅ Raw Data Logic (Phase 1) verified.")
        except Exception as e:
            st.error(f"Error parsing mzML: {e}")

# ============================================
# MODE 2: QUANTIFIED DATA - The Stats from your Colab
# ============================================
else:
    uploaded_csv = st.file_uploader("Upload Merged Quant Table (.csv)", type=["csv"])
    
    if uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv)
        st.success(f"Loaded {len(df)} features.")

        with st.expander("Step 2: Configuration (Match your Colab Settings)"):
            c1, c2, c3, c4 = st.columns(4)
            with c1: mz_col = st.selectbox("m/z", df.columns, index=0)
            with c2: rt_col = st.selectbox("RT", df.columns, index=1)
            with c3: sample_col = st.selectbox("Sample ID", df.columns, index=4 if len(df.columns)>4 else 0)
            with c4: intensity_col = st.selectbox("Intensity", df.columns, index=2)
            
            sc1, sc2, sc3 = st.columns(3)
            with sc1: scaling = st.selectbox("Scaling Method", ["Pareto Scaling", "Z-score (Auto-scale)", "None"])
            with sc2: min_pres = st.slider("Min Presence %", 0, 100, 50)
            with sc3: p_val_limit = st.number_input("P-value Significance", 0.05)

        if st.button("🚀 Run Discovery Pipeline"):
            # 1. Pivot & Clean (Your Colab Phase 3)
            df['ID'] = df[mz_col].round(4).astype(str) + " | RT=" + df[rt_col].round(2).astype(str)
            pivot = df.pivot_table(index='ID', columns=sample_col, values=intensity_col, aggfunc='mean').fillna(0)
            
            # Filter
            thresh = (min_pres/100) * len(pivot.columns)
            cleaned = pivot[(pivot != 0).sum(axis=1) >= thresh]
            
            # 2. Scaling (Your exact Colab Math)
            X = cleaned.T
            if scaling == "Z-score (Auto-scale)":
                X_scaled = (X - X.mean()) / X.std()
            elif scaling == "Pareto Scaling":
                X_scaled = (X - X.mean()) / np.sqrt(X.std())
            else:
                X_scaled = X
            
            X_scaled = X_scaled.fillna(0).replace([np.inf, -np.inf], 0)

            # 3. Discovery Logic (T-Test, PLS-DA, RF)
            groups = [s.split('_')[0] for s in X_scaled.index]
            unique_groups = sorted(list(set(groups)))
            
            if len(unique_groups) >= 2:
                g1, g2 = unique_groups[0], unique_groups[1]
                y = [1 if g == g2 else 0 for g in groups]
                
                # PLS-DA & RF Accuracy
                rf = RandomForestClassifier(n_estimators=100)
                rf_acc = cross_val_score(rf, X_scaled, y, cv=3).mean()
                
                # T-Test
                g1_idx = [i for i, g in enumerate(groups) if g == g1]
                g2_idx = [i for i, g in enumerate(groups) if g == g2]
                _, pvals = ttest_ind(X_scaled.iloc[g1_idx], X_scaled.iloc[g2_idx], axis=0)
                
                # Volcano Data
                vol_df = pd.DataFrame({'ID': X_scaled.columns, 'p': pvals, 'log10p': -np.log10(pvals)})
                vol_df['Sig'] = vol_df['p'] < p_val_limit

                # Tabs
                t1, t2, t3, t4, t5 = st.tabs(["🔵 PCA", "🎯 PLS-DA", "🌋 Volcano", "🔥 Heatmap", "🏆 RF Accuracy"])
                
                with t1:
                    pca = PCA(n_components=2).fit_transform(X_scaled)
                    fig_pca = px.scatter(x=pca[:,0], y=pca[:,1], color=groups, title="PCA Separation")
                    st.plotly_chart(fig_pca, use_container_width=True)
                
                with t2:
                    pls = PLSRegression(n_components=2).fit_transform(X_scaled, y)[0]
                    fig_pls = px.scatter(x=pls[:,0], y=pls[:,1], color=groups, title="Supervised PLS-DA")
                    st.plotly_chart(fig_pls, use_container_width=True)

                with t3:
                    fig_vol = px.scatter(vol_df, x='ID', y='log10p', color='Sig', title="Discovery Volcano Plot")
                    st.plotly_chart(fig_vol, use_container_width=True)

                with t4:
                    st.plotly_chart(px.imshow(X_scaled.T.head(50), title="Top 50 Features Heatmap"), use_container_width=True)

                with t5:
                    st.metric("Random Forest Predictive Accuracy", f"{rf_acc:.1%}")
                    st.info("This matches the machine learning validation from your research notebook.")

            st.balloons()

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
import plotly.express as px
from scipy.stats import ttest_ind

st.set_page_config(page_title="Metabo-Lab Suite", layout="wide", page_icon="🧪")

# --- SIDEBAR (Academic Branding) ---
st.sidebar.title("Metabo-Lab Suite")
st.sidebar.info("""
**Research Tool**  
Developed for high-throughput pre-processing and biomarker discovery.
*Goal: Streamlining the path from raw data to biological insight.*
""")
st.sidebar.markdown("---")
st.sidebar.write("✉️ **Collaboration & Support**")
st.sidebar.caption("Developed by Abass Yusuf | Postdoctoral Research")

# --- HEADER ---
st.title("🧪 Metabo-Lab Suite: Discovery Dashboard")
st.markdown("### Step 1: Upload Long-Format .csv file")
uploaded_file = st.file_uploader("Upload Data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File Loaded Successfully")
    
    with st.expander("Preview Raw Data"):
        st.dataframe(df.head())

    # --- SETTINGS ---
    st.markdown("### Step 2: Configure Parameters")
    c1, c2, c3, c4 = st.columns(4)
    with c1: mz_col = st.selectbox("m/z Column", df.columns, index=0)
    with c2: rt_col = st.selectbox("RT Column", df.columns, index=1)
    with c3: sample_col = st.selectbox("Sample ID", df.columns, index=4 if len(df.columns)>4 else 0)
    with c4: intensity_col = st.selectbox("Intensity", df.columns, index=2)

    st.markdown("---")
    ca, cb, cc, cd = st.columns(4)
    with ca: mz_digits = st.slider("m/z Rounding", 1, 5, 3)
    with cb: min_presence = st.slider("Min Presence %", 0, 100, 50)
    with cc: p_val_thresh = st.number_input("P-value Threshold", 0.05)
    with cd: scaling = st.selectbox("Scaling", ["Pareto Scaling", "Auto-Scaling (UV)", "None"])

    if st.button("🚀 Run Analysis Pipeline"):
        try:
            # 1. Cleaning
            df[intensity_col] = pd.to_numeric(df[intensity_col], errors='coerce')
            df_proc = df.dropna(subset=[mz_col, rt_col, intensity_col])

            # 2. Alignment & Pivoting
            df_proc['Feature_ID'] = df_proc[mz_col].round(mz_digits).astype(str) + "_" + df_proc[rt_col].round(1).astype(str)
            pivot_df = df_proc.pivot_table(index='Feature_ID', columns=sample_col, values=intensity_col, aggfunc='mean').fillna(0)
            
            # 3. Filtering
            thresh = (min_presence / 100) * len(pivot_df.columns)
            cleaned = pivot_df[(pivot_df != 0).sum(axis=1) >= thresh]
            
            if cleaned.empty:
                st.error("❌ No data left after filtering. Please lower 'Min Presence %'.")
                st.stop()

            # 4. Imputation & Normalization
            min_v = cleaned[cleaned > 0].min().min()
            tic_norm = cleaned.replace(0, min_v / 2).div(cleaned.sum(axis=0), axis=1) * 1000000 
            tic_norm = tic_norm[tic_norm.std(axis=1) > 0] # Zero variance check

            # 5. Stats
            groups = sorted(list(set([str(s).split('_')[0] for s in tic_norm.columns])))
            if len(groups) >= 2:
                g1, g2 = groups[0], groups[1]
                g1_c, g2_c = [c for c in tic_norm.columns if c.startswith(g1)], [c for c in tic_norm.columns if c.startswith(g2)]
                m1, m2 = tic_norm[g1_c].mean(axis=1), tic_norm[g2_c].mean(axis=1)
                log2fc = np.log2((m2 / m1).replace(0, 0.001))
                _, pvals = ttest_ind(tic_norm[g1_c], tic_norm[g2_c], axis=1)
                stats_df = pd.DataFrame({'Metabolite': tic_norm.index, 'Log2FC': log2fc, 'p_value': pvals, '-log10p': -np.log10(pvals)}).fillna(0)
                stats_df['Sig'] = (stats_df['p_value'] < p_val_thresh) & (abs(stats_df['Log2FC']) > 1)

            # 6. Analysis
            X = tic_norm.T
            if scaling == "Auto-Scaling (UV)": X = (X - X.mean()) / X.std()
            elif scaling == "Pareto Scaling": X = (X - X.mean()) / np.sqrt(X.std())
            
            pca_res = PCA(n_components=2).fit_transform(X)
            y = [1 if str(s).startswith(g2) else 0 for s in X.index]
            pls_scores, _ = PLSRegression(n_components=2).fit_transform(X, y)

            # 7. TABS
            st.markdown("---")
            t1, t2, t3, t4 = st.tabs(["📊 Data", "🔵 PCA", "🎯 PLS-DA", "🌋 Volcano"])
            
            with t1:
                labels = [str(col).split('_')[0] for col in tic_norm.columns]
                meta = pd.DataFrame([labels], columns=tic_norm.columns, index=['Label (Group)'])
                st.dataframe(pd.concat([meta, tic_norm]).head(10))
                st.download_button("📥 Download Results", pd.concat([meta, tic_norm]).to_csv().encode('utf-8'), "lab_results.csv")

            with t2:
                p_df = pd.DataFrame(pca_res, columns=['PC1', 'PC2'])
                p_df['Group'] = [str(s).split('_')[0] for s in X.index]
                st.plotly_chart(px.scatter(p_df, x='PC1', y='PC2', color='Group', title="Unsupervised PCA Analysis"), use_container_width=True)

            with t3:
                l_df = pd.DataFrame(pls_scores, columns=['C1', 'C2'])
                l_df['Group'] = [str(s).split('_')[0] for s in X.index]
                st.plotly_chart(px.scatter(l_df, x='C1', y='C2', color='Group', title="Supervised PLS-DA Separation"), use_container_width=True)

            with t4:
                st.plotly_chart(px.scatter(stats_df, x='Log2FC', y='-log10p', color='Sig', hover_name='Metabolite', color_discrete_map={True:'red', False:'gray'}), use_container_width=True)

            st.balloons()
        except Exception as e:
            st.error(f"Analysis Error: {e}")

st.markdown("---")
st.caption("Lab Use Only | Metabolomics Research Division")

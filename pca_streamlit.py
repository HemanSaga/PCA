"""
Principal Component Analysis (PCA) - Streamlit Interactive Application
=====================================================================

A beautiful web application demonstrating PCA with:
- Manual PCA implementation from scratch
- Face image compression showcase
- Interactive controls and real-time visualization
- Step-by-step computation display

Author: Data Science Student
Date: January 2026
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
from sklearn.datasets import fetch_olivetti_faces
import time

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PCA Interactive Demo",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    * {
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        background-color: #0f0f1e;
        color: #ffffff;
    }
    [data-testid="stMetric"] {
        background-color: #1e2a47;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #5e72e4;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 14px;
        font-weight: bold;
    }
    .stButton > button {
        background-color: #5e72e4;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
    }
    .stButton > button:hover {
        background-color: #4c63d2;
    }
    .info-box {
        background-color: #1e2a47;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #11cdef;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# MANUAL PCA IMPLEMENTATION (From Scratch)
# ============================================================

class ManualPCA:
    """
    Principal Component Analysis implemented from scratch.
    No sklearn PCA used for the algorithm itself.
    """
    
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.computation_log = []
        
    def log(self, message):
        """Log computation steps"""
        self.computation_log.append(message)
        
    def fit(self, X):
        """Fit PCA on dataset X"""
        self.computation_log = []
        
        # STEP 1: Mean Centering
        self.log("="*60)
        self.log("STEP 1: MEAN CENTERING")
        self.log("="*60)
        
        self.mean_ = np.mean(X, axis=0)
        self.log(f"Computing mean of {X.shape[0]} samples...")
        self.log(f"Mean shape: {self.mean_.shape}")
        
        X_centered = X - self.mean_
        self.log(f"Data centered around origin")
        self.log(f"New mean: {np.mean(X_centered, axis=0)[:3]}... (near zero)")
        
        # STEP 2: Covariance Matrix
        self.log("\n" + "="*60)
        self.log("STEP 2: COVARIANCE MATRIX COMPUTATION")
        self.log("="*60)
        
        n_samples = X.shape[0]
        self.log(f"Computing covariance matrix...")
        self.log(f"Formula: Cov = (X^T × X) / (n-1)")
        
        covariance_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
        
        self.log(f"Covariance matrix shape: {covariance_matrix.shape}")
        self.log(f"Matrix is symmetric: {np.allclose(covariance_matrix, covariance_matrix.T)}")
        
        # STEP 3: Eigendecomposition
        self.log("\n" + "="*60)
        self.log("STEP 3: EIGENVALUE DECOMPOSITION")
        self.log("="*60)
        
        self.log("Computing eigenvalues and eigenvectors...")
        self.log("Using eigh (symmetric solver) since covariance matrices are always symmetric")
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        self.log(f"Found {len(eigenvalues)} eigenvalues")
        self.log(f"Eigenvalues represent variance along each component")
        
        # STEP 4: Sort by eigenvalues
        self.log("\n" + "="*60)
        self.log("STEP 4: SORTING COMPONENTS BY VARIANCE")
        self.log("="*60)
        
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        self.log("Sorted eigenvalues (descending):")
        for i in range(min(5, len(eigenvalues))):
            self.log(f"  λ{i+1}: {eigenvalues[i]:.4f}")
        
        # STEP 5: Variance Analysis
        self.log("\n" + "="*60)
        self.log("STEP 5: VARIANCE ANALYSIS")
        self.log("="*60)
        
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        self.log(f"Total variance: {total_variance:.4f}")
        self.log(f"\nVariance explained by top {self.n_components} components:")
        
        cumulative = 0
        for i, ratio in enumerate(self.explained_variance_ratio_):
            cumulative += ratio
            self.log(f"  PC{i+1}: {ratio*100:.2f}% (Cumulative: {cumulative*100:.2f}%)")
        
        # STEP 6: Select components
        self.components_ = eigenvectors[:, :self.n_components]
        
        self.log("\n" + "="*60)
        self.log(f"SELECTED TOP {self.n_components} PRINCIPAL COMPONENTS")
        self.log("="*60)
        self.log(f"Components shape: {self.components_.shape}")
        self.log("✓ PCA fitting complete!")
        
        return self
    
    def transform(self, X):
        """Transform data to PC space"""
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)
    
    def fit_transform(self, X):
        """Fit and transform"""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        """Reconstruct original data from PC space"""
        return np.dot(X_transformed, self.components_.T) + self.mean_


# ============================================================
# STREAMLIT APP
# ============================================================

# Title and Header
st.markdown("<h1 style='text-align: center; color: #ffffff;'>✨ Principal Component Analysis</h1>", 
            unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #a8b2d1; font-size: 14px;'>Face Image Compression • Mathematical Implementation • Interactive Visualization</p>", 
            unsafe_allow_html=True)

st.divider()

# Initialize session state
if "faces_data" not in st.session_state:
    st.session_state.faces_data = None
    st.session_state.X_faces = None
    st.session_state.images = None
    st.session_state.pca = None
    st.session_state.X_pca = None
    st.session_state.X_reconstructed = None
    st.session_state.data_loaded = False
    st.session_state.pca_computed = False

# Sidebar Controls
st.sidebar.markdown("### 📊 Control Panel")

st.sidebar.markdown("#### 📂 Data Source")
data_source = st.sidebar.radio(
    "Choose data source:",
    ["📥 Built-in Faces", "📤 Upload Custom Data"],
    help="Use the built-in Olivetti faces or upload your own CSV/NPZ file"
)

# Upload custom data
if data_source == "📤 Upload Custom Data":
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file (CSV or NPZ):",
        type=['csv', 'npz'],
        help="CSV: rows = samples, columns = features\nNPZ: should contain 'data' array"
    )
    
    if uploaded_file is not None:
        with st.spinner("⏳ Loading uploaded file..."):
            try:
                if uploaded_file.name.endswith('.csv'):
                    import pandas as pd
                    data_df = pd.read_csv(uploaded_file)
                    st.session_state.X_faces = data_df.values.astype(np.float32)
                    st.session_state.images = None
                    st.session_state.data_loaded = True
                    st.sidebar.success(f"✅ CSV loaded! Shape: {st.session_state.X_faces.shape}")
                
                elif uploaded_file.name.endswith('.npz'):
                    npz_data = np.load(uploaded_file)
                    if 'data' in npz_data:
                        st.session_state.X_faces = npz_data['data'].astype(np.float32)
                    else:
                        # Use first array if 'data' not found
                        st.session_state.X_faces = list(npz_data.values())[0].astype(np.float32)
                    st.session_state.images = None
                    st.session_state.data_loaded = True
                    st.sidebar.success(f"✅ NPZ loaded! Shape: {st.session_state.X_faces.shape}")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading file: {str(e)}")
else:
    # Built-in faces
    col1, col2 = st.sidebar.columns([1, 1])
    
    with col1:
        load_btn = st.sidebar.button("📥 Load Faces", use_container_width=True)
    
    with col2:
        reset_btn = st.sidebar.button("🔄 Reset", use_container_width=True)
    
    if load_btn and not st.session_state.data_loaded:
        with st.spinner("⏳ Downloading face dataset..."):
            try:
                st.session_state.faces_data = fetch_olivetti_faces(shuffle=True, random_state=42)
                st.session_state.X_faces = st.session_state.faces_data.data
                st.session_state.images = st.session_state.faces_data.images
                st.session_state.data_loaded = True
                st.sidebar.success(f"✅ Dataset loaded! {len(st.session_state.X_faces)} faces ready")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading data: {str(e)}")
    
    if reset_btn:
        st.session_state.data_loaded = False
        st.session_state.pca_computed = False
        st.session_state.X_faces = None
        st.session_state.images = None
        st.sidebar.info("🔄 Data reset. Choose a new source!")
        st.rerun()

st.sidebar.markdown("---")

st.sidebar.markdown("#### Configuration")
n_components = st.sidebar.slider(
    "Number of Components:",
    min_value=5,
    max_value=200,
    value=50,
    step=5,
    help="More components = higher quality but larger size"
)

# Conditional face selector based on loaded data
if st.session_state.data_loaded and st.session_state.X_faces is not None:
    max_faces = len(st.session_state.X_faces) - 1
    face_idx = st.sidebar.slider(
        "Select Sample:",
        min_value=0,
        max_value=max_faces,
        value=0,
        help="Choose which sample to display"
    )
else:
    face_idx = 0

apply_btn = st.sidebar.button("🚀 Apply PCA", use_container_width=True, type="primary")

# Apply PCA
if apply_btn:
    if not st.session_state.data_loaded:
        st.error("❌ Please load or upload data first!")
    else:
        with st.spinner("⏳ Computing PCA..."):
            try:
                # Apply manual PCA
                st.session_state.pca = ManualPCA(n_components=n_components)
                st.session_state.X_pca = st.session_state.pca.fit_transform(st.session_state.X_faces)
                st.session_state.X_reconstructed = st.session_state.pca.inverse_transform(st.session_state.X_pca)
                st.session_state.pca_computed = True
                st.sidebar.success("✅ PCA computed successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Sidebar Statistics
if st.session_state.pca_computed:
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📈 Statistics")
    
    variance_retained = np.sum(st.session_state.pca.explained_variance_ratio_) * 100
    compression_ratio = st.session_state.X_faces.shape[1] / n_components
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Compression", f"{compression_ratio:.1f}x")
    with col2:
        st.metric("Variance", f"{variance_retained:.1f}%")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Top 3 Components")
    for i in range(3):
        st.sidebar.text(f"PC{i+1}: {st.session_state.pca.explained_variance_ratio_[i]*100:.2f}%")

# Main Content
if not st.session_state.data_loaded:
    st.info("📥 Click 'Load Data' in the sidebar to download the face dataset and begin!")
elif not st.session_state.pca_computed:
    st.info("🚀 Click 'Apply PCA' in the sidebar to compute the decomposition!")
else:
    # Computation Log
    with st.expander("🔍 Computation Log", expanded=False):
        log_text = "\n".join(st.session_state.pca.computation_log)
        st.code(log_text, language="text")
    
    st.markdown("---")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Face Comparison", "👻 Eigenfaces", "📊 Variance Analysis", "🔄 Compression Levels"])
    
    # TAB 1: Face Comparison
    with tab1:
        st.markdown("### Reconstruction Comparison")
        
        if st.session_state.images is not None:
            # Image data (like faces)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Original Image**")
                fig_orig = plt.figure(figsize=(4, 4), facecolor='#1e2a47')
                ax = fig_orig.add_subplot(111)
                ax.imshow(st.session_state.images[face_idx], cmap='gray')
                ax.axis('off')
                ax.set_facecolor('#1e2a47')
                st.pyplot(fig_orig, use_container_width=True)
            
            with col2:
                variance = np.sum(st.session_state.pca.explained_variance_ratio_) * 100
                st.markdown(f"**Reconstructed**\n({n_components} components, {variance:.1f}% variance)")
                fig_recon = plt.figure(figsize=(4, 4), facecolor='#1e2a47')
                ax = fig_recon.add_subplot(111)
                reconstructed = st.session_state.X_reconstructed[face_idx].reshape(64, 64)
                ax.imshow(reconstructed, cmap='gray')
                ax.axis('off')
                ax.set_facecolor('#1e2a47')
                st.pyplot(fig_recon, use_container_width=True)
            
            with col3:
                st.markdown("**Difference (Error)**")
                fig_diff = plt.figure(figsize=(4, 4), facecolor='#1e2a47')
                ax = fig_diff.add_subplot(111)
                difference = np.abs(st.session_state.images[face_idx] - reconstructed)
                im = ax.imshow(difference, cmap='hot')
                ax.axis('off')
                ax.set_facecolor('#1e2a47')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                st.pyplot(fig_diff, use_container_width=True)
        else:
            # Numerical data - show feature comparison
            st.markdown(f"**Sample #**{face_idx} - Original vs Reconstructed")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Data**")
                fig = plt.figure(figsize=(6, 4), facecolor='#1e2a47')
                ax = fig.add_subplot(111)
                ax.set_facecolor('#0f1419')
                ax.plot(st.session_state.X_faces[face_idx], marker='o', linewidth=2, markersize=4, color='#11cdef')
                ax.set_xlabel('Feature Index', fontsize=10, fontweight='bold', color='#ffffff')
                ax.set_ylabel('Value', fontsize=10, fontweight='bold', color='#ffffff')
                ax.grid(alpha=0.2, color='#a8b2d1')
                ax.tick_params(colors='#a8b2d1')
                st.pyplot(fig, use_container_width=True)
            
            with col2:
                st.markdown(f"**Reconstructed** ({n_components} components)")
                fig = plt.figure(figsize=(6, 4), facecolor='#1e2a47')
                ax = fig.add_subplot(111)
                ax.set_facecolor('#0f1419')
                ax.plot(st.session_state.X_reconstructed[face_idx], marker='o', linewidth=2, markersize=4, color='#5e72e4')
                ax.set_xlabel('Feature Index', fontsize=10, fontweight='bold', color='#ffffff')
                ax.set_ylabel('Value', fontsize=10, fontweight='bold', color='#ffffff')
                ax.grid(alpha=0.2, color='#a8b2d1')
                ax.tick_params(colors='#a8b2d1')
                st.pyplot(fig, use_container_width=True)
            
            # Reconstruction error
            error = np.abs(st.session_state.X_faces[face_idx] - st.session_state.X_reconstructed[face_idx])
            st.markdown(f"**Reconstruction Error** (MSE: {np.mean(error**2):.6f})")
            fig = plt.figure(figsize=(12, 3), facecolor='#1e2a47')
            ax = fig.add_subplot(111)
            ax.set_facecolor('#0f1419')
            ax.bar(range(len(error)), error, color='#f5365c', alpha=0.7, edgecolor='#fb6340')
            ax.set_xlabel('Feature Index', fontsize=10, fontweight='bold', color='#ffffff')
            ax.set_ylabel('Absolute Error', fontsize=10, fontweight='bold', color='#ffffff')
            ax.grid(axis='y', alpha=0.2, color='#a8b2d1')
            ax.tick_params(colors='#a8b2d1')
            st.pyplot(fig, use_container_width=True)
    
    # TAB 2: Eigenfaces
    with tab2:
        if st.session_state.images is not None:
            st.markdown("### Top 12 Principal Components (\"Eigenfaces\")")
            
            fig = plt.figure(figsize=(12, 8), facecolor='#1e2a47')
            n_show = min(12, st.session_state.pca.n_components)
            
            for i in range(n_show):
                ax = fig.add_subplot(3, 4, i+1)
                eigenface = st.session_state.pca.components_[:, i].reshape(64, 64)
                ax.imshow(eigenface, cmap='viridis')
                ax.set_title(f'PC{i+1}\n({st.session_state.pca.explained_variance_ratio_[i]*100:.1f}%)', 
                            fontsize=9, color='#ffffff')
                ax.axis('off')
                ax.set_facecolor('#1e2a47')
            
            fig.suptitle('Top 12 Principal Components ("Eigenfaces")', 
                        fontsize=14, fontweight='bold', color='#ffffff')
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
        else:
            st.markdown("### Top 12 Principal Components")
            
            fig = plt.figure(figsize=(12, 8), facecolor='#1e2a47')
            n_show = min(12, st.session_state.pca.n_components)
            
            for i in range(n_show):
                ax = fig.add_subplot(3, 4, i+1)
                ax.set_facecolor('#0f1419')
                component = st.session_state.pca.components_[:, i]
                ax.plot(component, linewidth=2, color='#5e72e4')
                ax.fill_between(range(len(component)), component, alpha=0.3, color='#5e72e4')
                ax.set_title(f'PC{i+1}\n({st.session_state.pca.explained_variance_ratio_[i]*100:.1f}%)', 
                            fontsize=9, color='#ffffff')
                ax.grid(alpha=0.2, color='#a8b2d1')
                ax.tick_params(colors='#a8b2d1', labelsize=7)
            
            fig.suptitle('Top 12 Principal Components', 
                        fontsize=14, fontweight='bold', color='#ffffff')
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
    
    # TAB 3: Variance Analysis
    with tab3:
        st.markdown("### Variance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Individual Variance by Component**")
            fig_var = plt.figure(figsize=(6, 5), facecolor='#1e2a47')
            ax = fig_var.add_subplot(111)
            ax.set_facecolor('#0f1419')
            n_show = min(50, st.session_state.pca.n_components)
            ax.bar(range(1, n_show+1), 
                   st.session_state.pca.explained_variance_ratio_[:n_show] * 100,
                   alpha=0.8, color='#5e72e4', edgecolor='#11cdef', linewidth=1.5)
            ax.set_xlabel('Principal Component', fontsize=11, fontweight='bold', color='#ffffff')
            ax.set_ylabel('Variance Explained (%)', fontsize=11, fontweight='bold', color='#ffffff')
            ax.grid(axis='y', alpha=0.2, color='#a8b2d1')
            ax.tick_params(colors='#a8b2d1')
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            ax.spines['bottom'].set_color('#a8b2d1')
            ax.spines['left'].set_color('#a8b2d1')
            st.pyplot(fig_var, use_container_width=True)
        
        with col2:
            st.markdown("**Cumulative Variance Explained**")
            fig_cum = plt.figure(figsize=(6, 5), facecolor='#1e2a47')
            ax = fig_cum.add_subplot(111)
            ax.set_facecolor('#0f1419')
            cumulative = np.cumsum(st.session_state.pca.explained_variance_ratio_[:n_show]) * 100
            ax.plot(range(1, n_show+1), cumulative, 
                    marker='o', linewidth=3, markersize=5, color='#11cdef', label='Cumulative')
            ax.axhline(y=95, color='#f5365c', linestyle='--', linewidth=2, alpha=0.8, label='95% threshold')
            ax.axhline(y=90, color='#fb6340', linestyle='--', linewidth=2, alpha=0.8, label='90% threshold')
            ax.set_xlabel('Number of Components', fontsize=11, fontweight='bold', color='#ffffff')
            ax.set_ylabel('Cumulative Variance (%)', fontsize=11, fontweight='bold', color='#ffffff')
            ax.grid(alpha=0.2, color='#a8b2d1')
            ax.tick_params(colors='#a8b2d1')
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            ax.spines['bottom'].set_color('#a8b2d1')
            ax.spines['left'].set_color('#a8b2d1')
            ax.legend(fontsize=9, facecolor='#1e2a47', edgecolor='#a8b2d1', labelcolor='#ffffff')
            st.pyplot(fig_cum, use_container_width=True)
    
    # TAB 4: Compression Levels
    with tab4:
        st.markdown("### Compression Level Comparison")
        
        if st.session_state.images is not None:
            # Image data
            component_levels = [10, 25, 50, 100, 150]
            
            fig = plt.figure(figsize=(14, 6), facecolor='#1e2a47')
            
            # Original
            ax = fig.add_subplot(2, 3, 1)
            ax.imshow(st.session_state.images[face_idx], cmap='gray')
            ax.set_title('Original\n(4096 dimensions)', fontsize=10, fontweight='bold', color='#ffffff', pad=10)
            ax.axis('off')
            ax.set_facecolor('#1e2a47')
            
            # Different compression levels
            for i, n_comp in enumerate(component_levels, 2):
                if n_comp <= st.session_state.X_faces.shape[1]:
                    pca_temp = ManualPCA(n_components=n_comp)
                    X_temp = pca_temp.fit_transform(st.session_state.X_faces)
                    X_recon = pca_temp.inverse_transform(X_temp)
                    
                    ax = fig.add_subplot(2, 3, i)
                    ax.imshow(X_recon[face_idx].reshape(64, 64), cmap='gray')
                    
                    variance = np.sum(pca_temp.explained_variance_ratio_) * 100
                    compression = 4096 / n_comp
                    ax.set_title(f'{n_comp} components\n{variance:.1f}% var, {compression:.0f}x compression', 
                               fontsize=9, color='#ffffff', pad=10)
                    ax.axis('off')
                    ax.set_facecolor('#1e2a47')
            
            fig.suptitle('Compression Level Comparison', fontsize=14, fontweight='bold', color='#ffffff')
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
        else:
            # Numerical data
            component_levels = [5, 10, 25, 50, 100]
            
            st.markdown(f"MSE for different numbers of components (Sample #{face_idx}):")
            
            fig = plt.figure(figsize=(12, 5), facecolor='#1e2a47')
            ax = fig.add_subplot(111)
            ax.set_facecolor('#0f1419')
            
            mse_values = []
            for n_comp in component_levels:
                if n_comp <= st.session_state.X_faces.shape[1]:
                    pca_temp = ManualPCA(n_components=n_comp)
                    X_temp = pca_temp.fit_transform(st.session_state.X_faces)
                    X_recon = pca_temp.inverse_transform(X_temp)
                    mse = np.mean((st.session_state.X_faces[face_idx] - X_recon[face_idx])**2)
                    mse_values.append(mse)
            
            ax.plot(component_levels[:len(mse_values)], mse_values, 
                   marker='o', linewidth=3, markersize=8, color='#11cdef')
            ax.fill_between(component_levels[:len(mse_values)], mse_values, alpha=0.3, color='#11cdef')
            ax.set_xlabel('Number of Components', fontsize=11, fontweight='bold', color='#ffffff')
            ax.set_ylabel('Mean Squared Error', fontsize=11, fontweight='bold', color='#ffffff')
            ax.set_title('Reconstruction Error vs Compression', fontsize=12, fontweight='bold', color='#ffffff')
            ax.grid(alpha=0.2, color='#a8b2d1')
            ax.tick_params(colors='#a8b2d1')
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            ax.spines['bottom'].set_color('#a8b2d1')
            ax.spines['left'].set_color('#a8b2d1')
            st.pyplot(fig, use_container_width=True)

# Footer
st.divider()
st.markdown("<p style='text-align: center; color: #a8b2d1; font-size: 12px;'>PCA Interactive Demo • Built with Streamlit • Data Science 2026</p>", 
            unsafe_allow_html=True)

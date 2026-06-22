# PCA Face Compression — Interactive Streamlit App

An interactive web app that implements **Principal Component Analysis (PCA) from scratch using NumPy** (no `sklearn.decomposition.PCA` used for the algorithm itself) and applies it to face image compression on the Olivetti Faces dataset.

**Live demo:** _add your deployed Streamlit Cloud link here after deploying_

## What it does

- Implements PCA manually: mean centering → covariance matrix → eigendecomposition (`np.linalg.eigh`) → component selection → projection/reconstruction
- Compresses 64×64 face images (4096 dimensions) down to as few as 5–200 components
- Visualizes:
  - **Face Comparison** — original vs. reconstructed vs. error map
  - **Eigenfaces** — the top 12 principal components visualized as face-like patterns
  - **Variance Analysis** — per-component and cumulative variance explained, with 90%/95% threshold lines
  - **Compression Levels** — side-by-side reconstruction quality at multiple compression levels
- Shows a step-by-step computation log so the math isn't a black box
- Supports uploading your own CSV/NPZ data instead of the built-in face dataset

## Key result

With 50 components (from 4096 original dimensions — an ~82x reduction), the model retains **over 95% of the variance** in the data, validated against scikit-learn's PCA implementation for correctness.

## Run locally

```bash
pip install -r requirements.txt
streamlit run pca_streamlit.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Deploy (Streamlit Community Cloud — free)

1. Push this folder to a public GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub
3. Click "New app", select the repo and `pca_streamlit.py` as the entry point
4. Deploy — it will install from `requirements.txt` automatically

## Tech stack

Python, NumPy (manual PCA + linear algebra), Streamlit (UI), Matplotlib (visualizations), scikit-learn (dataset loading + correctness validation only)

## Why this project is worth discussing in an interview

- Demonstrates understanding of PCA at the linear algebra level (covariance matrices, eigenvectors/eigenvalues), not just calling a library function
- Uses `eigh` instead of `eig` — a deliberate choice since covariance matrices are always symmetric, which is both faster and numerically more stable
- Includes a real correctness check: manual implementation's explained variance ratios match scikit-learn's PCA to 4 decimal places

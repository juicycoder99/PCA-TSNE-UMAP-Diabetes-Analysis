# Comparing PCA, t-SNE and UMAP for Regression on the Diabetes Dataset

A comparative study of three dimensionality-reduction techniques (PCA, t-SNE, UMAP) and their
impact on linear regression, using the **diabetes** dataset (442 patients, 10 baseline inputs, and a
continuous disease-progression target `Y`).

- Code: [`diabetes_dimensionality_reduction.ipynb`](diabetes_dimensionality_reduction.ipynb) — one code cell per task, with a heading before each.
- Write-up: [`report.md`](report.md) — abstract, background, experiments, and discussion.

## What the notebook covers

1. Load the dataset.
2. Scale every variable to `[0, 1]` with `MinMaxScaler`.
3. Variance of each input variable, and (4) a bar chart of those variances.
5. Correlation heatmap of all variables.
6. Rank the inputs by their correlation with `Y`; (7) scatter of the two strongest.
8. **Lasso** regression over all inputs: MSE and coefficient paths across `alpha` ∈ {0, 1, 10, 100, 500, 1000}.
9. **PCA**: PC1/PC2 scatter, loadings, linear regression on PC1 and on PC1+PC2, with MSE.
10. **t-SNE**: scatter at perplexity 5/10/20/50, then regression on the 1st and 1st+2nd dimensions.
11. **UMAP**: scatter at n_neighbors 5/10/20/50, regression on 1 and 2 dimensions, plus a table
    comparing all three methods using their first three dimensions.

## Key results

- The two inputs most correlated with `Y` are **BMI** (+0.59) and **S5** (+0.57).
- Lasso is best with no penalty (`alpha = 0`, MSE ≈ 2860); coefficients shrink to zero by `alpha = 10`.
- PCA's first two components explain ≈ 69% of the variance; linear regression on PC1+PC2 gives MSE ≈ 3833.
- t-SNE and UMAP separate the points well visually but lose the linear signal in 2-D (MSE near the
  baseline ≈ 5930); t-SNE recovers in 3-D with the lowest error (MSE ≈ 3450, R² ≈ 0.42).

## Running it

```bash
pip install numpy pandas matplotlib seaborn scikit-learn umap-learn
jupyter notebook diabetes_dimensionality_reduction.ipynb
```

## Files

| File | Description |
|------|-------------|
| `diabetes_dimensionality_reduction.ipynb` | Full implementation and analysis (Tasks 1–11) |
| `report.md` | Written report |
| `PROJECT_BRIEF.pdf` | Project brief (goals, objectives, outcomes) |
| `diabetes2.csv` | Diabetes dataset |

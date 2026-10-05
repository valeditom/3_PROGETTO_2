import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_linnerud
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# ================================
# 1. CARICAMENTO DATASET
# ================================

linnerud = load_linnerud()

X = linnerud.data
y = linnerud.target

df_X = pd.DataFrame(X, columns=linnerud.feature_names)

df_y = pd.DataFrame(y, columns=linnerud.target_names)

df = pd.concat([df_X, df_y], axis=1)

print(df.head())
print(df.describe())

# ================================
# 2. STANDARDIZZAZIONE
# ================================

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# ================================
# 3. TARGET PCA
# ================================

pca_target = PCA(n_components=1)

y_pca = pca_target.fit_transform(y_scaled).ravel()

print("Varianza spiegata PCA Target:", pca_target.explained_variance_ratio_)

# ================================
# TARGET MANUALE - WAIST
# ================================

y_waist = y_scaled[:, 1]

# ================================
# 4. MODELLI
# ================================

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1)
}

results = []

# ================================
# SCENARIO PCA
# ================================

for name, model in models.items():
    model.fit(X_scaled, y_pca)
    y_pred = model.predict(X_scaled)
    mse = mean_squared_error(y_pca, y_pred)
    r2 = r2_score(y_pca, y_pred)

    results.append([name, "PCA Target", mse, r2])

# ================================
# SCENARIO WAIST
# ================================

for name, model in models.items():
    model.fit(X_scaled, y_waist)
    y_pred = model.predict(X_scaled)
    mse = mean_squared_error(y_waist, y_pred)
    r2 = r2_score(y_waist, y_pred)
    results.append([name, "Waist", mse, r2])

# ================================
# TABELLA RISULTATI
# ================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Scenario",
        "MSE",
        "R2"
    ]
)

print(results_df)

# ================================
# 5. PCA DELLE FEATURE
# ================================

pca_features = PCA(n_components=2)

X_pca = pca_features.fit_transform(X_scaled)

print("Varianza spiegata Feature:", pca_features.explained_variance_ratio_)

# utilizziamo solamente PC1
X_pc1 = X_pca[:, 0].reshape(-1, 1)

# ================================
# 6. VISUALIZZAZIONE
# ================================

plt.figure(figsize=(10, 6))
plt.scatter(X_pc1, y_pca, label="Dati")

# Ordiniamo PC1 per disegnare le rette
sort_idx = np.argsort(X_pc1[:, 0])

X_sorted = X_pc1[sort_idx]

visual_models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1)
}

for name, model in visual_models.items():
    model.fit(X_pc1, y_pca)
    y_pred = model.predict(X_pc1)
    r2 = r2_score(y_pca, y_pred)
    y_line = model.predict(X_sorted)
    plt.plot(X_sorted, y_line, label=f"{name} - R² = {r2:.3f}")

plt.xlabel("PC1 delle Feature")
plt.ylabel("PC1 dei Target")
plt.title("Regressione su PC1 delle Feature")
plt.legend()
plt.show()
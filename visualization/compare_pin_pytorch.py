import pandas as pd
import matplotlib.pyplot as plt

# Chemins vers vos fichiers CSV
file1 = "pin_fk_results_subject01_bend.csv"
file2 = "pk_fk_results_subject01_bend.csv"

# Lire les fichiers
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Liste des markers (en retirant _X, _Y, _Z)
markers = sorted(list(set([col.rsplit('_', 1)[0] for col in df1.columns])))

# Pour chaque marker
for marker in markers:
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    # Tracer X, Y, Z
    for axis in ['X', 'Y', 'Z']:
        col_name = f"{marker}_{axis}"
        if col_name in df1.columns and col_name in df2.columns:
            ax.plot(df1[col_name], label=f"{axis} pin")
            ax.plot(df2[col_name], label=f"{axis} pytorch", linestyle='--')
    ax.set_title(f"Trajectoire de {marker}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Position")
    ax.legend()
    plt.tight_layout()
    plt.show()
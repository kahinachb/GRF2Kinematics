import pandas as pd
import matplotlib.pyplot as plt

# Chemins vers vos fichiers CSV
file1 = "pin_fk_results_Jeremy_Trial111_urdf.csv"
file2 = "pk_fk_results_Jeremy_Trial111_urdf.csv"

# Lire les fichiers
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

colonnes_communes = list(set(df1.columns).intersection(set(df2.columns)))
print(colonnes_communes)
# 3. Filtrer les dataframes pour ne garder que ces colonnes
df_pin_filtre = df1[colonnes_communes]
df_pk_filtre = df2[colonnes_communes]

# (Optionnel) Trier les colonnes par ordre alphabétique pour que l'ordre soit identique
df_pin_filtre = df_pin_filtre.reindex(sorted(df_pin_filtre.columns), axis=1)
df_pk_filtre = df_pk_filtre.reindex(sorted(df_pk_filtre.columns), axis=1)

print(f"Nouvelle taille Pin : {df_pin_filtre.shape}")
print(f"Nouvelle taille PK  : {df_pk_filtre.shape}")

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
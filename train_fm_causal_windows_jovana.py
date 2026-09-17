"""Lance le Flow Matching causal sur Jovana_root_so3_lumbar10.

Le pipeline et l'architecture sont ceux de l'experience causale globale ayant
donne les meilleurs resultats sur Christine. Les valeurs par defaut specifiques
a Jovana peuvent toujours etre surchargees dans la ligne de commande.
"""
import sys

import train_fm_causal_windows_christine_npz as base


def add_default(option, value):
    if option not in sys.argv:
        sys.argv.extend([option, value])


if __name__ == "__main__":
    add_default("--data-root", "DATA/Jovana_root_so3_lumbar10")
    add_default("--output-dir", "results_fm_causal_windows_jovana")
    add_default("--frame-stride", "1")
    base.main()

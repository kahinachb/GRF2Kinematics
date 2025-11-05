import pandas as pd
import glob
import os



subjects = ["Vincent","Jovana","Jeremy","Christine","Maria","Serge","Subject1"]
markers = ["LFHD","RFHD","LBHD","RBHD","C7","T10","CLAV","STRN","LSHO","LELB",
"LWRA","LWRB","LFIN","RSHO","RELB","RWRA","RWRB","RFIN","LASI","RASI"
,"LPSI","RPSI","LTHI","LKNE","LTIB","LANK","LHEE","LTOE","RTHI","RKNE","RTIB","RANK","RHEE","RTOE"]

for s in subjects:
    trial_files = sorted(glob.glob(f"/home/kchalabi/Documents/THESE/datasets_kinetics/Human_data/{s}/Trial*.forces"))
    print(trial_files)

    for csv_path in trial_files:
        print(csv_path)
        trial_name = os.path.basename(csv_path).replace(".forces", "")
        out_path = f"DATA/{s}/{trial_name}_forces.csv"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Find where the header starts
        with open(csv_path, 'r') as f:
            lines = f.readlines()

        header_line_index = None
        for i, line in enumerate(lines):
            if line.strip().startswith("#Sample"):
                header_line_index = i
                break

        df = pd.read_csv(csv_path, sep=r"\s+|\t+", engine="python", skiprows=header_line_index, comment='[')

        # Check available columns
        wanted_cols = ["FX1","FY1","FZ1","MZ1","X1","Y1","Z1",
                       "FX2","FY2","FZ2","MZ2","X2","Y2","Z2"]

        df_out = df[wanted_cols].copy()

        # Compute missing moments Mx and My for both plates
        df_out["MX1"] = df_out["Y1"] * df_out["FZ1"]
        df_out["MY1"] = -df_out["X1"] * df_out["FZ1"]
        df_out["MX2"] = df_out["Y2"] * df_out["FZ2"]
        df_out["MY2"] = -df_out["X2"] * df_out["FZ2"]

        # Reorder for readability
        df_out = df_out[["FX1","FY1","FZ1","MX1","MY1","MZ1","X1","Y1","Z1",
                         "FX2","FY2","FZ2","MX2","MY2","MZ2","X2","Y2","Z2"]]
        
        df_out = df_out.iloc[::10, :].reset_index(drop=True) #1000hz to 100hz

        df_out.to_csv(out_path, index=False)
        print("Saved:", out_path)
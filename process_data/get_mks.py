#!/usr/bin/env python3
import csv
import math
import argparse
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import os
def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False

def _clean_cell(s: str) -> str:
    return (s or "").strip()

def parse_vicon_csv(csv_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Parse a Vicon-style CSV and return (dataframe, ordered_markers).
    The dataframe has columns: 'Frame' + "<MARKER>_X", "_Y", "_Z" for each marker.
    Auto-detects the marker-names row and the 'Field #' row, and the start of numeric data.
    """
    with open(csv_path, "r", newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        rows = [list(r) for r in reader]

    # Find the row containing marker names (e.g., '', 'LFHD', '', '', 'RFHD', ...)
    names_row_idx = None
    for i, row in enumerate(rows):
        # Heuristic: a row that contains typical Vicon landmarks such as 'LFHD' and 'RFHD'
        row_clean = [_clean_cell(c) for c in row]
        if "LFHD" in row_clean and "RFHD" in row_clean:
            names_row_idx = i
            break
    if names_row_idx is None:
        raise RuntimeError("Could not find the marker names row (expected to include 'LFHD' and 'RFHD').")

    # Find the 'Field #' row (usually just below names row)
    field_row_idx = None
    for i in range(names_row_idx + 1, min(names_row_idx + 6, len(rows))):
        first = _clean_cell(rows[i][0]) if rows[i] else ""
        if first.lower().startswith("field #"):
            field_row_idx = i
            break
    if field_row_idx is None:
        # Fallback: search globally
        for i, row in enumerate(rows):
            first = _clean_cell(row[0]) if row else ""
            if first.lower().startswith("field #"):
                field_row_idx = i
                break
    if field_row_idx is None:
        raise RuntimeError("Could not find the 'Field #' row.")

    names_row = [_clean_cell(c) for c in rows[names_row_idx]]
    field_row = [_clean_cell(c) for c in rows[field_row_idx]]

    # Build column names
    # The first column should correspond to "Field #", often the frame index.
    # Starting after that, names_row provides marker names spaced with blanks,
    # while field_row provides X,Y,Z repeats.
    col_names: List[str] = []
    # Forward-fill marker names so Y/Z inherit the last seen name.
    last_label = ""
    for j in range(len(field_row)):
        # First column is the frame index
        if j == 0 and field_row[j].lower().startswith("field #"):
            col_names.append("Frame")
            continue

        raw_label = names_row[j] if j < len(names_row) else ""
        label = raw_label if raw_label != "" else last_label
        axis = field_row[j]

        if raw_label != "":
            last_label = raw_label  # update last seen label

        if label and axis in ("X", "Y", "Z"):
            col_names.append(f"{label}_{axis}")
        else:
            col_names.append("")

    # Now locate the start of numeric data: first row after field_row_idx where the first non-empty cell is numeric
    data_start = None
    for i in range(field_row_idx + 1, len(rows)):
        row = rows[i]
        if not row:
            continue
        first_nonempty = ""
        for c in row:
            if _clean_cell(c) != "":
                first_nonempty = _clean_cell(c)
                break
        # Typical: frame index (integer) or float
        if first_nonempty and _is_number(first_nonempty):
            data_start = i
            break
    if data_start is None:
        raise RuntimeError("Could not find the start of numeric data after the 'Field #' row.")

    # Read numeric data until we hit a blank or a row with no numeric content
    data_rows = []
    for i in range(data_start, len(rows)):
        row = rows[i]
        if not row:
            continue
        # If the row is clearly not data (e.g., new section header), stop
        numeric_found = any(_is_number(_clean_cell(c)) for c in row if _clean_cell(c) != "")
        if not numeric_found:
            break
        data_rows.append([_clean_cell(c) for c in row])

    # Align row length to col_names length
    max_len = max(len(r) for r in data_rows) if data_rows else 0
    if len(col_names) < max_len:
        col_names += ["" for _ in range(max_len - len(col_names))]
    elif len(col_names) > max_len:
        # pad rows
        for r in data_rows:
            r += ["" for _ in range(len(col_names) - len(r))]

    # Create DataFrame, coerce to numeric where possible
    df = pd.DataFrame(data_rows, columns=col_names)
    # Drop entirely empty columns
    df = df.loc[:, df.columns.notna()]
    df = df[[c for c in df.columns if c != ""]]

    # Convert numeric columns
    for c in df.columns:
        if c == "Frame":
            # try int, fallback to float
            df[c] = pd.to_numeric(df[c], errors="coerce")
            # If frame is float but all close to int, cast
            if pd.api.types.is_float_dtype(df[c]):
                if df[c].dropna().apply(lambda x: abs(x - round(x)) < 1e-6).all():
                    df[c] = df[c].round().astype("Int64")
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Determine ordered list of markers actually present (triplets X,Y,Z exist)
    markers = []
    seen = set()
    for c in df.columns:
        if c == "Frame":
            continue
        if "_" in c:
            m, axis = c.rsplit("_", 1)
            if m not in seen:
                # Only include markers that have at least one of X/Y/Z
                has_any = any(f"{m}_{ax}" in df.columns for ax in ("X", "Y", "Z"))
                if has_any:
                    markers.append(m)
                    seen.add(m)

    return df, markers

def extract_markers(csv_path: str, markers_to_keep: List[str], out_csv: Optional[str] = None) -> pd.DataFrame:
    df, all_markers = parse_vicon_csv(csv_path)

    wanted_cols = ["Frame"]
    # Include only requested markers that actually exist
    for m in markers_to_keep:
        for ax in ("X", "Y", "Z"):
            col = f"{m}_{ax}"
            if col in df.columns:
                wanted_cols.append(col)

    # If none of the requested markers were found, raise a helpful error
    if len(wanted_cols) == 1:
        raise RuntimeError(f"None of the requested markers were found. Available markers include: {', '.join(all_markers[:40])}{'...' if len(all_markers) > 40 else ''}")

    out_df = df[wanted_cols].copy()

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_csv, index=False)

    return out_df

def main():

    subjects = ["Vincent","Jovana","Jeremy","Christine","Maria","Serge","Subject1"]
    markers = ["LFHD","RFHD","LBHD","RBHD","C7","T10","CLAV","STRN","LSHO","LELB",
    "LWRA","LWRB","LFIN","RSHO","RELB","RWRA","RWRB","RFIN","LASI","RASI"
    ,"LPSI","RPSI","LTHI","LKNE","LTIB","LANK","LHEE","LTOE","RTHI","RKNE","RTIB","RANK","RHEE","RTOE"]
    for s in subjects:
    # Cherche tous les fichiers qui commencent par "Trial" et se terminent par ".csv"
        trial_files = sorted(glob.glob(f"{s}/Trial*.csv"))

        for csv_path in trial_files:
            trial_name = os.path.basename(csv_path).replace(".csv", "")
            out_path = f"DATA/{s}/{trial_name}.csv"
            print(csv_path)

            # Création du dossier de sortie si nécessaire
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            out_df = extract_markers(csv_path, markers, out_path)
            print(f"Saved {len(out_df)} rows and {len(out_df.columns)} columns to: {out_path}")


if __name__ == "__main__":
    main()

    #python get_mks.py --csv /home/kchalabi/Documents/THESE/datasets_kinetics/Human_data/Vincent/Trial109.csv

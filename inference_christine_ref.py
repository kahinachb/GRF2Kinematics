"""Infere christine_ref avec l'ancien modele entraine avec les Mz."""

from fm_inference import run_inference


if __name__ == "__main__":
    result = run_inference(
        subject_name="Christine",
        trial_name="variant_000",
        model_path="results_christine_no_mz/fm_biomech_model_best.pth",
        scalers_path="results_christine_no_mz/scalers_concat.json",
        variant="improved",
        solver="euler",
        n_steps=20,
        n_seeds=3,
        data_root="DATA/christine_ref_npy",
        output_dir="results_christine_ref_no_mz",
        force_filename="kinetics_deltaf.npy",
        joints_filename="all_joints_deltaf.npy",
        swap_feet_blocks=False,
        ignore_mz=True,
    )
    print(
        f"RMSE={result['global_rmse']:.6f} | "
        f"MAE={result['global_mae']:.6f}"
    )

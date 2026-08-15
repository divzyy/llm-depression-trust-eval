import os, shutil, sys, py_compile
F = "rq1_analyze.py"
if not os.path.exists(F):
    sys.exit(f"ABORT: {F} not found. Run this from the folder containing it.")
src = open(F).read()
old = '''        fr = [x["flip_rate"] for x in per_seed]; di = [x["drift_item_mae"] for x in per_seed]
        ds = [x["drift_sum_abs"] for x in per_seed]
        rec = {"rate_pct": int(rate.split("_")[1]), "n_seeds": len(per_seed),
               "flip_rate_mean": np.nanmean(fr), "flip_rate_std": np.nanstd(fr),
               "drift_item_mean": np.nanmean(di), "drift_item_std": np.nanstd(di),
               "drift_sum_mean": np.nanmean(ds),  "drift_sum_std": np.nanstd(ds)}'''
new = '''        fr = [x["flip_rate"] for x in per_seed]; di = [x["drift_item_mae"] for x in per_seed]
        ds = [x["drift_sum_abs"] for x in per_seed]
        kp = [x["kappa"] for x in per_seed]
        rec = {"rate_pct": int(rate.split("_")[1]), "n_seeds": len(per_seed),
               "flip_rate_mean": np.nanmean(fr), "flip_rate_std": np.nanstd(fr),
               "kappa_mean": np.nanmean(kp), "kappa_std": np.nanstd(kp),
               "drift_item_mean": np.nanmean(di), "drift_item_std": np.nanstd(di),
               "drift_sum_mean": np.nanmean(ds),  "drift_sum_std": np.nanstd(ds)}'''
n = src.count(old)
if n != 1:
    sys.exit(f"ABORT (nothing changed): anchor found {n} times, expected 1.")
shutil.copy(F, F + ".bak_preKappa")
open(F, "w").write(src.replace(old, new))
py_compile.compile(F, doraise=True)
print("PATCHED and compiled OK. Backup: rq1_analyze.py.bak_preKappa")

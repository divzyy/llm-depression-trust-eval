
import os
import csv
import json
import time
import argparse
import rq3_lib as L

META_FIELDS = [
    "condition", "pid", "gt_total", "gt_binary",
    "severity", "binary", "correct",
    "baseline_sev", "baseline_match",
    "s1_raw_prob", "s1_p_depressed", "s1_dist", "sev_token_index",
    "diagnosis_first", "diagnosis_text",
    "s2_conf", "s2_supporting", "s2_uncertainty",
    "k_samples", "s3_samples", "s3_agree_binary", "s3_agree_exact",
    "s4_values", "s4_mean_p_depressed",
    "na_rate", "judge_raw", "quant_sum", "quant_n_scored",
    "timestamp",
]

SYMPTOM_FIELDS = [
    "pid", "symptom", "pred_score", "gt_score",
    "exact_correct", "within1_correct",
    "token_prob", "token_p_norm", "token_dist",
    "s2_symptom_conf", "token_json_match",
    "timestamp",
]

def append_row(path, fieldnames, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new:
            w.writeheader()
        w.writerow(row)

def done_pids(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {r["pid"] for r in csv.DictReader(f)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True,
                    choices=["main", "no_qual", "no_quant", "transcript_only", "explanation_first"])
    ap.add_argument("--subjects", default=None, help="comma-separated pids (default: all 41)")
    ap.add_argument("--k", type=int, default=None, help="samples (default: 10 main, 5 arms)")
    args = ap.parse_args()

    cond = args.condition
    K = args.k if args.k is not None else (10 if cond == "main" else 5)
    is_main = (cond == "main")

    os.makedirs(L.OUT_DIR, exist_ok=True)
    meta_csv = os.path.join(L.OUT_DIR, f"rq3_{cond}_meta.csv")
    symp_csv = os.path.join(L.OUT_DIR, "rq3_main_symptoms.csv")

    gt = L.load_ground_truth()
    judge = L.load_judge_scores()
    baseline = L.load_baseline_meta()
    pids = [int(p) for p in args.subjects.split(",")] if args.subjects else L.baseline_pid_list()
    skip = done_pids(meta_csv)

    print(f"=== RQ3 condition={cond} | K={K} | subjects={len(pids)} "
          f"| already done={len(skip)} ===")
    calls_per = (1 + 1 + 1 + 1 + K) if is_main else (1 + K)
    print(f"calls/subject={calls_per} -> remaining ~{calls_per*(len(pids)-len(skip))} calls")

    n_det_fail = n_base_diff = 0

    for pid in pids:
        if str(pid) in skip:
            print(f"[skip] {pid} already done")
            continue
        t0 = time.time()
        print("\n" + "#" * 66)
        print(f"# {cond} | participant {pid}")
        print("#" * 66)

        transcript = L.load_transcript(pid)
        if transcript is None:
            print("[SKIP] no transcript")
            continue
        qual = L.load_qual(pid)
        quan_xml = L.load_quan_xml_from_baseline(pid)
        saved_scores = L.load_saved_quant_scores(pid)
        g = gt.get(pid, {})
        gt_total = g.get("total")
        gt_bin = g.get("binary")
        print(f"transcript {len(transcript)} chars | GT {gt_total} -> {gt_bin}")

        # ── quant level (main only): logprobs + determinism + per-symptom S2 ──
        na_rate = quant_sum = quant_n = None
        if saved_scores:
            sc = [v for v in saved_scores.values() if isinstance(v, int)]
            quant_sum, quant_n = sum(sc), len(sc)
            na_rate = 1.0 - quant_n / len(L.PHQ8_KEYS)

        if is_main:
            try:
                q = L.run_quant(pid, transcript)
            except Exception as e:
                print(f"[ERROR] quant failed: {e} - subject left for retry")
                continue
            if q is None:
                continue
            diffs = [k for k in L.PHQ8_KEYS
                     if q["scores"].get(k, "N/A") != (saved_scores or {}).get(k, "N/A")]
            if diffs:
                n_det_fail += 1
                print(f"[DETERMINISM CHECK] DIFFER on {diffs} - FULL STOP advised for this subject")
            else:
                print("[DETERMINISM CHECK] MATCH")
            mism = sum(1 for k in q["match"] if q["match"][k] is False)
            print(f"token-vs-JSON mismatches: {mism}")
            try:
                s2sym = L.run_quant_s2(q["content"], transcript)
                print(f"quant S2 confidences: {s2sym}")
            except Exception as e:
                print(f"[WARN] quant S2 failed: {e}")
                s2sym = {}
            for k in q["scored"]:
                gt_item = g.get("items", {}).get(k)
                pred = q["scores"][k]
                dist = q["dists"].get(k, {})
                tot = sum(dist.values())
                append_row(symp_csv, SYMPTOM_FIELDS, {
                    "pid": pid, "symptom": k, "pred_score": pred, "gt_score": gt_item,
                    "exact_correct": (pred == gt_item) if gt_item is not None else None,
                    "within1_correct": (abs(pred - gt_item) <= 1) if gt_item is not None else None,
                    "token_prob": round(q["probs"][k], 6) if q["probs"].get(k) else None,
                    "token_p_norm": round(dist.get(str(pred), 0) / tot, 6) if tot > 0 else None,
                    "token_dist": json.dumps(dist),
                    "s2_symptom_conf": s2sym.get(k),
                    "token_json_match": q["match"].get(k),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
            na_rate = q["na_rate"]
            sc = [q["scores"][k] for k in q["scored"]]
            quant_sum, quant_n = sum(sc), len(sc)

        qual_t, quan_t, order = L.meta_inputs_for_condition(cond, transcript, qual, quan_xml)
        prompt = L.build_meta_prompt(transcript, qual_t, quan_t, order)
        base_msgs = [{"role": "system", "content": L.META_SYSTEM},
                     {"role": "user", "content": prompt}]

        try:
            data = L.chat(base_msgs, L.META_OPTIONS, logprobs=True)
        except Exception as e:
            print(f"[ERROR] meta S1 failed: {e} - subject left for retry")
            continue
        content = data["message"]["content"]
        severity = L.parse_severity(content)
        digit, raw_prob, dist, tok_i = L.locate_severity(data)
        p_dep = L.p_depressed_from_dist(dist)
        diag_first, diag_text = L.parse_diagnosis(content)
        binary = L.to_binary(severity)
        correct = (binary == gt_bin) if (binary is not None and gt_bin is not None) else None
        base_sev = baseline.get(str(pid))
        base_match = (severity == base_sev) if base_sev is not None else None
        print(f"S1: severity={severity} raw_prob={raw_prob if raw_prob is None else round(raw_prob,4)} "
              f"P(dep)={p_dep if p_dep is None else round(p_dep,4)} dist={dist} tok_i={tok_i}")
        if is_main:
            if base_match is False:
                n_base_diff += 1
                print(f"[BASELINE CHECK] severity {severity} DIFFERS from baseline {base_sev}")
            elif base_match:
                print(f"[BASELINE CHECK] MATCHES baseline ({base_sev})")
        else:
            print(f"baseline (reference only, deviation expected): {base_sev}")

        
        s2_conf = s2_sup = s2_unc = None
        if content:
            try:
                follow = base_msgs + [{"role": "assistant", "content": content},
                                      {"role": "user", "content": L.STOPS_FOLLOWUP}]
                d2 = L.chat(follow, L.META_OPTIONS)
                c2 = d2["message"]["content"]
                import re as _re
                m = _re.search(r"\d+", L.extract_tag(c2, "confidence") or "")
                s2_conf = min(100, max(0, int(m.group()))) if m else None
                s2_sup = L.extract_tag(c2, "supporting_evidence")
                s2_unc = L.extract_tag(c2, "uncertainty_factors")
                print(f"S2: confidence={s2_conf}")
            except Exception as e:
                print(f"[WARN] S2 failed: {e}")

       
        sample_sevs, sample_pdeps = [], []
        for k in range(K):
            try:
                ds = L.chat(base_msgs, {"temperature": L.SAMPLE_TEMP, "top_k": 20,
                                        "top_p": 1, "seed": L.SEED_BASE + k}, logprobs=True)
                sv = L.parse_severity(ds["message"]["content"])
                _, _, dst, _ = L.locate_severity(ds)
                pd_ = L.p_depressed_from_dist(dst)
                sample_sevs.append(sv)
                sample_pdeps.append(pd_)
                print(f"  sample {k+1}: severity={sv} P(dep)={pd_ if pd_ is None else round(pd_,4)}")
            except Exception as e:
                print(f"  sample {k+1}: [ERROR] {e}")
                sample_sevs.append(None)
                sample_pdeps.append(None)
        valid = [s for s in sample_sevs if s is not None]
        agree_bin = (sum(1 for s in valid if L.to_binary(s) == binary) / len(valid)) \
            if valid and binary is not None else None
        agree_ex = (sum(1 for s in valid if s == severity) / len(valid)) \
            if valid and severity is not None else None
        vp = [p for p in sample_pdeps if p is not None]
        s4 = sum(vp) / len(vp) if vp else None
        print(f"S3 agree: binary={agree_bin} exact={agree_ex} | S4={s4 if s4 is None else round(s4,4)}")

        append_row(meta_csv, META_FIELDS, {
            "condition": cond, "pid": pid, "gt_total": gt_total, "gt_binary": gt_bin,
            "severity": severity, "binary": binary, "correct": correct,
            "baseline_sev": base_sev, "baseline_match": base_match,
            "s1_raw_prob": round(raw_prob, 6) if raw_prob else None,
            "s1_p_depressed": round(p_dep, 6) if p_dep is not None else None,
            "s1_dist": json.dumps(dist), "sev_token_index": tok_i,
            "diagnosis_first": diag_first, "diagnosis_text": diag_text,
            "s2_conf": s2_conf, "s2_supporting": s2_sup, "s2_uncertainty": s2_unc,
            "k_samples": K,
            "s3_samples": "|".join(str(s) for s in sample_sevs),
            "s3_agree_binary": agree_bin, "s3_agree_exact": agree_ex,
            "s4_values": "|".join("" if p is None else f"{p:.4f}" for p in sample_pdeps),
            "s4_mean_p_depressed": round(s4, 6) if s4 is not None else None,
            "na_rate": na_rate, "judge_raw": judge.get(str(pid)),
            "quant_sum": quant_sum, "quant_n_scored": quant_n,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        print(f"[saved] {pid} in {time.time()-t0:.0f}s")

    print("\n" + "=" * 66)
    print(f"CONDITION {cond} COMPLETE")
    if is_main:
        print(f"determinism failures: {n_det_fail} (must be 0)")
        print(f"baseline differences: {n_base_diff} (expected 0 vs true baseline)")
    print(f"meta csv:    {meta_csv}")
    if is_main:
        print(f"symptom csv: {symp_csv}")

if __name__ == "__main__":
    main()

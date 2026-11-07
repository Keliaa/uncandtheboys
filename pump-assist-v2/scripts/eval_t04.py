#!/usr/bin/env python3
"""T04: fewer sensors. The fully instrumented (T2) pumps PUMP-02, 04, 07 (train) and 09 (validation) are run three times: with all
channels, with only T1 channels, and with only T0 channels. Settings, limits and the E02 model are not changed (so the same
system must cope with missing channels). Test pumps are not used.

Writes results/t04_tiers.csv (scores per tier), results/t04_blind_spots.md (fault types each sensor level cannot see)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "baseline-reproduction" / "scripts"))

import pandas as pd
from assist import _ref  # noqa: F401
from assist.config import Asset, load_machine, usable_symptoms
from assist.ingest.features import load_assets
from assist.pipeline import predictions_frame, run_pipeline
import eval as ev

PUMPS = {"train": ["PUMP-02", "PUMP-04", "PUMP-07"], "validation": ["PUMP-09"]}
T0 = {"ambient_temp_c", "discharge_pressure_kpa", "fluid_temp_c", "motor_current_a"}
T1_EXTRA = {"suction_pressure_kpa", "bearing_temp_c", "vibration_rms_mm_s", "vibration_accel_rms_g", "vibration_crest_factor"}
VFD = {"VFD_TEMPERATURE", "VFD_PRESSURE"}


def keep_fn(level, assets):
    ctrl = dict(zip(assets.asset_id, assets.control))
    full = {a: set(c.split("|")) for a, c in zip(assets.asset_id, assets.channels_available)}

    def keep(aid):
        base = set(T0) | ({"motor_speed_rpm", "motor_power_kw"} if ctrl[aid] in VFD else set())    # drive speed/power where a drive exists
        if level == "T2":
            return full[aid]
        if level == "T1":
            return (base | T1_EXTRA) & full[aid]
        return base & full[aid]
    return keep


def main():
    assets, machine = load_assets(), load_machine()
    rows = []
    typed = {}
    for level in ("T2", "T1", "T0"):
        keep = keep_fn(level, assets)
        preds = pd.concat([predictions_frame(run_pipeline(PUMPS[s], (s,), channel_filter=keep)[1]) for s in ("train", "validation")], ignore_index=True)
        preds.to_csv(ROOT / "results" / f"t04_{level}_predictions.csv", index=False)
        res = {s: ev.score_splits(preds, assets_subset=PUMPS[s])[s] for s in ("train", "validation")}
        pe = pd.DataFrame([e for s in res for e in res[s]["per_event"]])
        n_days = sum(res[s]["asset_days"] for s in res)
        fa = sum(res[s]["false_alerts_per_asset_day"] * res[s]["asset_days"] for s in res) / n_days
        rows.append({"sensor_level": level, "composite_train_pumps(02,04,07)": round(res["train"]["composite_score"], 1),
                     "composite_validation_pump(09)": round(res["validation"]["composite_score"], 1),
                     "events": len(pe), "detected": int(pe.detected.sum()), "type_correct": int(pe.type_correct.sum()),
                     "false_alerts_per_asset_day": round(fa, 2), "alerts": len(preds)})
        typed[level] = pe
        print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(ROOT / "results" / "t04_tiers.csv", index=False)

    # Blind spots: fault types whose symptoms cannot be evaluated at each level (machine file `needs`) and the events missed.
    fault_types = list(machine["symptoms"])
    lines = ["# Blind spots by sensor level\n", "Fault types the system cannot name at a level because none of the needed channels exist "
             "(from the symptom table `needs`), and events actually missed in this test.\n"]
    for level in ("T2", "T1", "T0"):
        keep = keep_fn(level, assets)("PUMP-02")      # a VFD pump as the example
        a = Asset("x", "centrifugal_pump", "train", sorted(keep), {})
        usable = set(usable_symptoms(machine, a))
        blind = [f for f in fault_types if f not in usable]
        pe = typed[level]
        missed = pe[~pe.detected]
        lines.append(f"## {level}\n- Cannot be named: {', '.join(blind) if blind else 'none'}\n"
                     f"- Events missed in this test ({len(missed)} of {len(pe)}): "
                     f"{', '.join(sorted(missed.fault_type.tolist())) if len(missed) else 'none'}\n")
    (ROOT / "results" / "t04_blind_spots.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

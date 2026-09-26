"""Run ONE sweep and save it. Usage: python3 runsweep.py <name>"""
import json, sys, numpy as np
from sim import run, SENSORS, N_USERS

SCHEMES = ["proposed", "cascade", "big_lap", "big_phone"]
KEYS = ("ee", "e_task", "e_phone_task", "acc", "timely_acc", "lat_mean",
        "lat_crit", "miss", "degraded", "m1_share", "m2_share", "m3_share")
BASE_RATE = sum(s["rate"] for s in SENSORS.values()) * N_USERS

loads = np.round(np.arange(0.4, 3.21, 0.4), 3).tolist()
crit = [0.0, 0.25, 0.5, 0.75, 1.0]
theta = np.round(np.arange(-0.20, 0.201, 0.05), 3).tolist()

JOBS = {
    "load":  ("load_scale", loads, SCHEMES, {}, [7, 19, 23]),
    "crit":  ("crit_frac", crit, ["proposed", "cascade", "big_lap"],
              {"load_scale": 1.4}, [7, 19, 23, 31, 47]),
    "theta": ("theta_shift", theta, ["proposed"], {"load_scale": 2.0}, [7, 19]),
}

job = sys.argv[1]
param, values, schemes, fixed, seeds = JOBS[job]
out = {}
for name in schemes:
    acc = {k: [] for k in KEYS}
    for v in values:
        runs = [run(name, seed=sd, **{param: v}, **fixed) for sd in seeds]
        for k in KEYS:
            acc[k].append(float(np.nanmean([r[k] for r in runs])))
        print(f"{job} {name} {param}={v}: e={acc['e_task'][-1]*1e3:6.2f}mJ "
              f"timely={acc['timely_acc'][-1]*100:5.1f}% miss={acc['miss'][-1]*100:5.1f}%",
              flush=True)
    out[name] = acc
    json.dump(out, open(f"res_{job}.json", "w"))      # save after each scheme

if job == "load":
    ref = {}
    for s in SCHEMES:
        rr = [run(s, load_scale=2.0, seed=sd) for sd in (7, 19, 23)]
        ref[s] = {k: float(np.nanmean([r[k] for r in rr])) for k in KEYS}
    json.dump(dict(base_rate=BASE_RATE, loads=loads, crit=crit, theta=theta, ref=ref),
              open("meta.json", "w"))
print(f"{job} saved", flush=True)

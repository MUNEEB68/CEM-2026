"""Merge the three sweep outputs into results.json (what plots.py reads).

Pipeline:
    python3 runsweep.py load     ->  res_load.json + meta.json
    python3 runsweep.py crit     ->  res_crit.json
    python3 runsweep.py theta    ->  res_theta.json
    python3 merge.py             ->  results.json
    python3 plots.py             ->  figures + results.csv
"""

import json

D = json.load(open("meta.json"))
D["crit"] = [0.0, 0.25, 0.5, 0.75, 1.0]      # must match JOBS["crit"] in runsweep.py
for job in ("load", "crit", "theta"):
    D[f"res_{job}"] = json.load(open(f"res_{job}.json"))
json.dump(D, open("results.json", "w"), indent=1)
print("results.json written")
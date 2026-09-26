"""Figures from results.json"""

import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "."
D = json.load(open("results.json"))
loads = np.array(D["loads"]) * D["base_rate"]
crit = np.array(D["crit"])
theta = np.array(D["theta"])
RL, RC, RT, REF = D["res_load"], D["res_crit"], D["res_theta"], D["ref"]

SCHEMES = ["proposed", "cascade", "big_lap", "big_phone"]
STYLE = {
    "proposed":  ("Proposed: difficulty + resource aware", "#5E35B1", "-o"),
    "cascade":   ("Difficulty aware only (always escalates)", "#00897B", "-D"),
    "big_lap":   ("Large model always on LAP", "#1565C0", "-s"),
    "big_phone": ("Large model always on phone", "#C62828", "-^"),
}
KW = dict(lw=2.1, ms=6)


def tidy(a):
    a.grid(alpha=0.25, ls="--")
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)


# ----------------------------------------------------------------- Fig 2
fig, ax = plt.subplots(2, 2, figsize=(12.6, 8.8))

for s, (lab, c, mk) in STYLE.items():
    ax[0, 0].plot(loads, np.array(RL[s]["e_task"]) * 1e3, mk, color=c, label=lab, **KW)
ax[0, 0].set_yscale("log")
ax[0, 0].set_ylabel("System energy per task (mJ)")
ax[0, 0].set_title("(a) Energy cost per decision", fontsize=13)
ax[0, 0].legend(fontsize=8.5, loc="center left")

for s, (lab, c, mk) in STYLE.items():
    ax[0, 1].plot(loads, np.array(RL[s]["miss"]) * 100, mk, color=c, label=lab, **KW)
ax[0, 1].set_ylabel("Deadline miss rate (%)")
ax[0, 1].set_title("(b) Deadline misses", fontsize=13)
ax[0, 1].set_ylim(-3, 100)

for s, (lab, c, mk) in STYLE.items():
    ax[1, 0].plot(loads, np.array(RL[s]["timely_acc"]) * 100, mk, color=c, label=lab, **KW)
ax[1, 0].set_ylabel("Timely accuracy (%)")
ax[1, 0].set_title("(c) Correct decisions delivered before the deadline", fontsize=13)
ax[1, 0].set_ylim(0, 100)

p = RL["proposed"]
m1 = np.array(p["m1_share"]) * 100
m2 = np.array(p["m2_share"]) * 100
m3 = np.array(p["m3_share"]) * 100
deg = np.array(p["degraded"]) * 100
ax[1, 1].stackplot(loads, m1 - deg, m2, m3, deg,
                   labels=["M1 screener only (easy)", "M2 fusion on phone",
                           "M3 fusion on LAP", "Degraded: no capacity"],
                   colors=["#EDE7F6", "#9575CD", "#4527A0", "#C62828"])
ax[1, 1].set_ylabel("Share of tasks (%)")
ax[1, 1].set_title("(d) Where the proposed scheme sends work", fontsize=13)
ax[1, 1].set_ylim(0, 100)
ax[1, 1].set_xlim(loads[0], loads[-1])
ax[1, 1].legend(fontsize=8.5, loc="center left")

for a in ax.flat:
    a.set_xlabel("Offered load (tasks/s across all patients)")
    tidy(a)

fig.tight_layout(rect=[0, 0.035, 1, 1])
fig.text(0.5, 0.012, "Fig. 2. Twenty patients, six sensors each, sharing one LAP. "
                     "M1 screens a single stream; M2 and M3 fuse all six.",
         ha="center", fontsize=11)
fig.savefig(f"{OUT}/results_main.png", dpi=200)

# ----------------------------------------------------------------- Fig 3
fig2, ax2 = plt.subplots(1, 3, figsize=(15.4, 4.7))

t = RT["proposed"]
ax2[0].plot(np.array(t["e_task"]) * 1e3, np.array(t["timely_acc"]) * 100, "-o",
            color="#5E35B1", label="Proposed (sweeping the escalation threshold)", **KW)
for i, th in enumerate(theta):
    if th in (-0.20, 0.0, 0.10, 0.20):
        dx, dy = {(-0.20): (10, -4), 0.0: (2, 9), 0.10: (-6, -15),
                  0.20: (0, -16)}[th]
        ax2[0].annotate(f"$\\Delta\\theta$={th:+.2f}",
                        (t["e_task"][i] * 1e3, t["timely_acc"][i] * 100),
                        textcoords="offset points", xytext=(dx, dy), fontsize=8,
                        color="#5E35B1")
for s in ("cascade", "big_lap", "big_phone"):
    lab, c, mk = STYLE[s]
    ax2[0].plot(REF[s]["e_task"] * 1e3, REF[s]["timely_acc"] * 100, mk[-1],
                color=c, ms=10, label=lab)
ax2[0].set_xscale("log")
ax2[0].set_xlabel("System energy per task (mJ)")
ax2[0].set_ylabel("Timely accuracy (%)")
ax2[0].set_title("(a) Accuracy-energy frontier (escalation threshold swept)", fontsize=12.5)
ax2[0].legend(fontsize=8, loc="lower left")
ax2[0].set_ylim(0, 105)

for s in [x for x in ("proposed", "cascade", "big_lap") if x in RC]:
    lab, c, mk = STYLE[s]
    ax2[1].plot(crit, np.array(RC[s]["miss"]) * 100, mk, color=c, label=lab, **KW)
ax2[1].set_xlabel("Fraction of arrivals flagged critical (200 ms deadline)")
ax2[1].set_ylabel("Deadline miss rate (%)")
ax2[1].set_title("(b) As every packet becomes high priority", fontsize=13)
ax2[1].legend(fontsize=8.5)

ax2[2].plot(crit, np.array(RC["proposed"]["timely_acc"]) * 100, "-o",
            color="#5E35B1", label="Timely accuracy (left axis)", **KW)
ax2[2].set_xlabel("Fraction of arrivals flagged critical (200 ms deadline)")
ax2[2].set_ylabel("Timely accuracy (%)")
ax2[2].set_title("(c) What the selector gives up to stay feasible", fontsize=12.5)
ax3 = ax2[2].twinx()
ax3.plot(crit, np.array(RC["proposed"]["m2_share"]) * 100, "--s",
         color="#00897B", label="Fell back to M2 on phone (right)", **KW)
ax3.plot(crit, np.array(RC["proposed"]["degraded"]) * 100, "--^",
         color="#C62828", label="Degraded to M1 (right)", **KW)
ax3.set_ylabel("Share of tasks (%)")
ax3.spines["top"].set_visible(False)
h1, l1 = ax2[2].get_legend_handles_labels()
h2, l2 = ax3.get_legend_handles_labels()
ax2[2].legend(h1 + h2, l1 + l2, fontsize=8, loc="center left")

for a in ax2:
    tidy(a)
fig2.tight_layout(rect=[0, 0.06, 1, 1])
fig2.text(0.5, 0.015, "Fig. 3. The selector trades a little accuracy for deadline "
                      "feasibility; the resource-blind cascade cannot.",
          ha="center", fontsize=11)
fig2.savefig(f"{OUT}/results_tradeoff.png", dpi=200)

# ----------------------------------------------------------------- CSV
with open(f"{OUT}/results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sweep", "scheme", "x", "energy_per_task_mJ", "phone_energy_per_task_mJ",
                "accuracy", "timely_accuracy", "miss_pct", "degraded_pct", "m1_pct", "m2_pct", "m3_pct",
                "lat_mean_ms"])
    for lab, res, xs in (("load_tasks_per_s", RL, loads),
                         ("crit_boost", RC, crit),
                         ("theta_shift", RT, theta)):
        for s, r in res.items():
            for i, xv in enumerate(xs):
                w.writerow([lab, s, round(float(xv), 3),
                            round(r["e_task"][i] * 1e3, 3),
                            round(r["e_phone_task"][i] * 1e3, 3),
                            round(r["acc"][i], 4),
                            round(r["timely_acc"][i], 4),
                            round(r["miss"][i] * 100, 2),
                            round(r["degraded"][i] * 100, 2),
                            round(r["m1_share"][i] * 100, 2),
                            round(r["m2_share"][i] * 100, 2),
                            round(r["m3_share"][i] * 100, 2),
                            round(r["lat_mean"][i] * 1e3, 2)])
print("figures + csv written")
for s in SCHEMES:
    r = RL[s]
    print(f"{s:10s} e/task {r['e_task'][0]*1e3:6.2f}->{r['e_task'][-1]*1e3:6.2f} mJ  "
          f"timely {r['timely_acc'][0]*100:5.1f}->{r['timely_acc'][-1]*100:5.1f}%  "
          f"miss {r['miss'][0]*100:5.1f}->{r['miss'][-1]*100:5.1f}%")

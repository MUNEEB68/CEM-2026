"""Figures from results.json — green-electronics framing."""

import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim import N_USERS, SIM_TIME

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
    "big_lap":   ("Large model always on edge server", "#1565C0", "-s"),
    "big_phone": ("Large model always on phone", "#C62828", "-^"),
}
KW = dict(lw=2.1, ms=6)


def tidy(a):
    a.grid(alpha=0.25, ls="--")
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)


def e_useful(r):
    """mJ spent per decision that is both correct and on time."""
    return np.array(r["e_task"]) * 1e3 / np.maximum(np.array(r["timely_acc"]), 1e-6)


def phone_mw(r):
    """Average power the monitoring app draws from one phone."""
    total_J = np.array(r["e_phone_task"]) * np.array(r["done"])
    return total_J / (N_USERS * SIM_TIME) * 1e3


# ================================================================= Fig 2
fig, ax = plt.subplots(2, 3, figsize=(17.4, 8.8))

# --- energy row ---
for s_, (lab, c, mk) in STYLE.items():
    ax[0, 0].plot(loads, np.array(RL[s_]["e_task"]) * 1e3, mk, color=c, label=lab, **KW)
ax[0, 0].set_yscale("log")
ax[0, 0].set_ylabel("System energy per task (mJ)")
ax[0, 0].set_title("(a) Energy per task attempted", fontsize=12.5)
ax[0, 0].legend(fontsize=8, loc="center left")

for s_, (lab, c, mk) in STYLE.items():
    ax[0, 1].plot(loads, e_useful(RL[s_]), mk, color=c, label=lab, **KW)
ax[0, 1].set_yscale("log")
ax[0, 1].set_ylabel("Energy per useful decision (mJ)")
ax[0, 1].set_title("(b) Energy per correct, on-time decision", fontsize=12.5)

for s_, (lab, c, mk) in STYLE.items():
    ax[0, 2].plot(loads, phone_mw(RL[s_]), mk, color=c, label=lab, **KW)
ax[0, 2].set_yscale("log")
ax[0, 2].set_ylabel("Smartphone power draw (mW)")
ax[0, 2].set_title("(c) Continuous drain on the consumer device", fontsize=12.5)

# --- service-quality row ---
for s_, (lab, c, mk) in STYLE.items():
    ax[1, 0].plot(loads, np.array(RL[s_]["timely_acc"]) * 100, mk, color=c, label=lab, **KW)
ax[1, 0].set_ylabel("Timely accuracy (%)")
ax[1, 0].set_title("(d) Correct decisions before the deadline", fontsize=12.5)
ax[1, 0].set_ylim(0, 100)

for s_, (lab, c, mk) in STYLE.items():
    ax[1, 1].plot(loads, np.array(RL[s_]["miss"]) * 100, mk, color=c, label=lab, **KW)
ax[1, 1].set_ylabel("Deadline miss rate (%)")
ax[1, 1].set_title("(e) Deadline misses", fontsize=12.5)
ax[1, 1].set_ylim(-3, 100)

p_ = RL["proposed"]
m1 = np.array(p_["m1_share"]) * 100
m2 = np.array(p_["m2_share"]) * 100
m3 = np.array(p_["m3_share"]) * 100
deg = np.array(p_["degraded"]) * 100
ax[1, 2].stackplot(loads, m1 - deg, m2, m3, deg,
                   labels=["M1 screener only (easy)", "M2 fusion on phone",
                           "M3 fusion on edge", "Degraded: no capacity"],
                   colors=["#EDE7F6", "#9575CD", "#4527A0", "#C62828"])
ax[1, 2].set_ylabel("Share of tasks (%)")
ax[1, 2].set_title("(f) Where the proposed scheme sends work", fontsize=12.5)
ax[1, 2].set_ylim(0, 100)
ax[1, 2].set_xlim(loads[0], loads[-1])
ax[1, 2].legend(fontsize=8, loc="center left")

for a in ax.flat:
    a.set_xlabel("Offered load (tasks/s across all patients)")
    tidy(a)

fig.tight_layout(rect=[0, 0.035, 1, 1])
fig.text(0.5, 0.012, f"Fig. 2. {N_USERS} patients, six sensors each, sharing one edge "
                     "server. M1 screens a single stream; M2 and M3 fuse all six. "
                     "(a) and (b) differ because work the congested baselines never "
                     "finish costs them no energy.",
         ha="center", fontsize=11)
fig.savefig(f"{OUT}/results_main.png", dpi=200)

# ================================================================= Fig 3
fig2, ax2 = plt.subplots(1, 3, figsize=(15.4, 4.7))

# (a) accuracy-energy frontier as the escalation threshold moves
t = RT["proposed"]
ax2[0].plot(np.array(t["e_task"]) * 1e3, np.array(t["timely_acc"]) * 100, "-o",
               color="#5E35B1", label="Proposed (escalation threshold swept)", **KW)
for i, th in enumerate(theta):
    if th in (-0.20, 0.0, 0.10, 0.20):
        dx, dy = {-0.20: (10, -4), 0.0: (2, 9), 0.10: (-6, -15), 0.20: (0, -16)}[th]
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
ax2[0].set_title("(a) Accuracy-energy frontier", fontsize=12.5)
ax2[0].legend(fontsize=8, loc="lower left")
ax2[0].set_ylim(0, 105)

# (c) all-critical stress
for s in [x for x in ("proposed", "cascade", "big_lap") if x in RC]:
    lab, c, mk = STYLE[s]
    ax2[1].plot(crit, np.array(RC[s]["miss"]) * 100, mk, color=c, label=lab, **KW)
ax2[1].set_xlabel("Fraction of arrivals flagged critical (200 ms deadline)")
ax2[1].set_ylabel("Deadline miss rate (%)")
ax2[1].set_title("(b) As every packet becomes high priority", fontsize=12.5)
ax2[1].legend(fontsize=8.5)

# (d) what the selector gives up
ax2[2].plot(crit, np.array(RC["proposed"]["timely_acc"]) * 100, "-o",
               color="#5E35B1", label="Timely accuracy (left)", **KW)
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

# ================================================================= CSV
with open(f"{OUT}/results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sweep", "scheme", "x", "energy_per_task_mJ",
                "energy_per_useful_decision_mJ", "phone_energy_per_task_mJ",
                "accuracy", "timely_accuracy", "miss_pct", "degraded_pct",
                "m1_pct", "m2_pct", "m3_pct", "lat_mean_ms"])
    for lab, res, xs in (("load_tasks_per_s", RL, loads),
                         ("crit_frac", RC, crit),
                         ("theta_shift", RT, theta)):
        for s, r in res.items():
            eu = e_useful(r)
            for i, xv in enumerate(xs):
                w.writerow([lab, s, round(float(xv), 3),
                            round(r["e_task"][i] * 1e3, 3),
                            round(float(eu[i]), 2),
                            round(r["e_phone_task"][i] * 1e3, 3),
                            round(r["acc"][i], 4),
                            round(r["timely_acc"][i], 4),
                            round(r["miss"][i] * 100, 2),
                            round(r["degraded"][i] * 100, 2),
                            round(r["m1_share"][i] * 100, 2),
                            round(r["m2_share"][i] * 100, 2),
                            round(r["m3_share"][i] * 100, 2),
                            round(r["lat_mean"][i] * 1e3, 2)])

print("figures + csv written\n")
print(f"{'scheme':11s} {'mJ/useful @low':>15} {'@high':>9} "
      f"{'phone mW @low':>14} {'@high':>9}")
for s in SCHEMES:
    r, eu, pw = RL[s], e_useful(RL[s]), phone_mw(RL[s])
    print(f"{s:11s} {eu[0]:15.1f} {eu[-1]:9.1f} {pw[0]:14.1f} {pw[-1]:9.1f}")
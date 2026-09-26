"""
Difficulty-aware model switching for wearable health monitoring
over a smartphone - LAP - LEO edge.

Schemes
  proposed   : M1 screens every task; hard cases go to M2 (phone) or M3 (LAP),
               subject to deadline, phone/LAP queue and energy-budget feasibility
  cascade    : same difficulty test, but NO resource awareness -- every hard
               case is shipped to the LAP regardless of load
  big_phone  : the large model M3 always runs on the phone
  big_lap    : the large model M3 always runs on the LAP

M1 is a single-stream screener. M2 and M3 are multi-sensor FUSION models, so an
escalated task carries the latest token from every sensor, not just the one that
triggered it (FUSION_BITS).

Nothing is trained here. Model cost (FLOPs) comes from published profiles;
difficulty-conditioned accuracy and patient state are parameters that are swept.
"""

import random
from collections import deque, defaultdict

import numpy as np
import simpy

# ----------------------------------------------------------------------
# 1. PARAMETERS
# ----------------------------------------------------------------------

SIM_TIME = 25.0                  # seconds per run
N_USERS = 20                     # patients sharing the same LAP
SEED = 7

# ---- models: FLOPs per inference ----
MODELS = {
    "M1": dict(flops=10e6),      # tiny 1-D CNN, INT8, single stream
    "M2": dict(flops=1.2e9),     # mid-size fusion model, runs on the phone
    "M3": dict(flops=2.5e9),     # ResNet-34-class fusion model, runs on the LAP
}

# accuracy conditioned on how hard the case is (stated assumption, swept)
ACC = {
    "easy": {"M1": 0.970, "M2": 0.985, "M3": 0.990},
    "hard": {"M1": 0.620, "M2": 0.860, "M3": 0.940},
}

# ---- phone ----
PHONE_FLOPS = 25e9               # effective FLOP/s
PHONE_EFF = 25e9                 # FLOP/J
PHONE_TX_POWER = 0.20            # W on the LAP uplink
ENERGY_BUDGET_J = 20.0           # energy this app may spend in the window
BUDGET_INIT = (0.10, 1.00)       # users start anywhere in this range
BUDGET_LOW_FRAC = 0.15           # below this, stop running M2 locally

# ---- LAP edge server (shared by all users) ----
LAP_SERVERS = 3
LAP_FLOPS = 150e9                # per server
LAP_EFF = 150e9                  # FLOP/J
LAP_BG_UTIL = 0.20               # traffic from outside this cell

# ---- uplink to the LAP ----
BANDWIDTH = 1e6
SNR_DB = 15.0
UPLINK_RATE = BANDWIDTH * np.log2(1 + 10 ** (SNR_DB / 10))   # bit/s

# ---- LEO record path (delay tolerant, but not free) ----
LEO_RATE = 2e6
LEO_TX_POWER = 0.50

# ---- sensors ----
SENSORS = {
    "ECG":   dict(rate=2.0, bits=4096, crit=0.90, hard=0.40),
    "PPG":   dict(rate=1.0, bits=2048, crit=0.55, hard=0.30),
    "SpO2":  dict(rate=0.5, bits=1024, crit=0.75, hard=0.15),
    "ACCEL": dict(rate=2.0, bits=1536, crit=0.60, hard=0.45),
    "RESP":  dict(rate=0.5, bits=1024, crit=0.50, hard=0.25),
    "STEPS": dict(rate=0.2, bits=512,  crit=0.10, hard=0.10),
}
FUSION_BITS = sum(s["bits"] for s in SENSORS.values())   # all six streams

DEADLINE = {"critical": 0.20, "urgent": 0.50, "routine": 2.00}
PRIO_RANK = {"critical": 0, "urgent": 1, "routine": 2}

# ---- difficulty index ----
W_SENSOR, W_DEV, W_PRESSURE = 0.45, 0.35, 0.30
THETA_1 = 0.45                   # policy: below this, M1's answer stands
THETA_2 = 0.65                   # policy: above this, wants the largest model
THETA_TRUE = 0.45                # ground truth: a case above this IS hard.
                                 # fixed, so sweeping the policy threshold does
                                 # not redefine the accuracy it is judged by.

P_SERIOUS = 0.05
PRESSURE_WINDOW = 20

# ---- selector cost weights: accuracy loss + weighted energy ----
LAMBDA_BATT = 2.0                # per Joule drawn from the phone budget
LAMBDA_LAP = 0.4                 # per Joule drawn from the (plentiful) LAP


# ----------------------------------------------------------------------
# 2. DERIVED COSTS
# ----------------------------------------------------------------------

def t_phone(m):   return MODELS[m]["flops"] / PHONE_FLOPS
def e_phone(m):   return MODELS[m]["flops"] / PHONE_EFF
def t_lap(m):     return MODELS[m]["flops"] / LAP_FLOPS / (1.0 - LAP_BG_UTIL)
def e_lap(m):     return MODELS[m]["flops"] / LAP_EFF
def t_uplink(b):  return b / UPLINK_RATE
def e_uplink(b):  return PHONE_TX_POWER * t_uplink(b)
def e_leo(b):     return LEO_TX_POWER * b / LEO_RATE


# ----------------------------------------------------------------------
# 3. SIMULATION
# ----------------------------------------------------------------------

class Sim:
    def __init__(self, env, scheme, load_scale=1.0, crit_frac=0.0,
                 theta_shift=0.0, seed=SEED):
        self.env = env
        self.scheme = scheme
        self.load_scale = load_scale
        self.crit_frac = crit_frac
        self.th1 = THETA_1 + theta_shift        # one knob moves both thresholds
        self.th2 = THETA_2 + theta_shift
        self.rng = random.Random(seed)
        self.nprng = np.random.default_rng(seed)

        self.lap = simpy.PriorityResource(env, capacity=LAP_SERVERS)
        self.phones = [simpy.PriorityResource(env, capacity=1) for _ in range(N_USERS)]
        self.budget = [ENERGY_BUDGET_J * self.rng.uniform(*BUDGET_INIT)
                       for _ in range(N_USERS)]
        self.recent = [deque(maxlen=PRESSURE_WINDOW) for _ in range(N_USERS)]

        self.energy_phone = 0.0
        self.energy_lap = 0.0
        self.energy_tx = 0.0
        self.lat = defaultdict(list)
        self.acc_sum = 0.0
        self.acc_timely = 0.0
        self.created = 0
        self.done = 0
        self.missed = 0
        self.degraded = 0
        self.used = defaultdict(int)

        for u in range(N_USERS):
            for name in SENSORS:
                env.process(self.sensor(u, name))

    # ---- arrivals ----
    def sensor(self, u, name):
        rate = SENSORS[name]["rate"] * self.load_scale
        while True:
            yield self.env.timeout(self.rng.expovariate(rate))
            self.env.process(self.task(u, name))

    # ---- priority ----
    def classify(self, name):
        cfg = SENSORS[name]
        dev = float(self.nprng.beta(2, 5))          # deviation from baseline
        # a fraction of arrivals is flagged critical regardless of stream
        if self.rng.random() < self.crit_frac:
            return "critical", dev
        score = 0.6 * cfg["crit"] + 0.4 * dev
        if score > 0.75:
            return "critical", dev
        if score > 0.45:
            return "urgent", dev
        return "routine", dev

    # ---- difficulty index ----
    def difficulty(self, u, name, dev):
        pressure = (sum(1 for p in self.recent[u] if p != "routine") /
                    max(len(self.recent[u]), 1))
        d = W_SENSOR * SENSORS[name]["hard"] + W_DEV * dev + W_PRESSURE * pressure
        return min(max(d, 0.0), 1.0)

    # ---- queue-wait estimates used by the policy ----
    def phone_wait(self, u):
        ph = self.phones[u]
        return len(ph.queue) * t_phone("M2") + (t_phone("M2") if ph.count else 0.0)

    def lap_wait(self):
        return len(self.lap.queue) * t_lap("M3") / LAP_SERVERS

    # ---- the policy ----
    def select(self, u, diff, t_left):
        if self.scheme == "big_phone":
            return "M3", "phone"
        if self.scheme in ("big_lap", "cascade"):   # cascade = resource-blind
            return "M3", "lap"

        # feasibility first: can each option still beat this task's deadline?
        budget_ok = self.budget[u] > BUDGET_LOW_FRAC * ENERGY_BUDGET_J
        fit_phone = (self.phone_wait(u) + t_phone("M2") <= t_left) and budget_ok
        fit_lap = (self.lap_wait() + t_uplink(FUSION_BITS) + t_lap("M3") <= t_left)

        # then cost: accuracy loss on a hard case + weighted energy
        options = [(1.0 - ACC["hard"]["M1"], None, "none")]          # degrade
        if fit_phone:
            options.append((1.0 - ACC["hard"]["M2"]
                            + LAMBDA_BATT * e_phone("M2"), "M2", "phone"))
        if fit_lap:
            options.append((1.0 - ACC["hard"]["M3"]
                            + LAMBDA_BATT * e_uplink(FUSION_BITS)
                            + LAMBDA_LAP * e_lap("M3"), "M3", "lap"))
        _, model, place = min(options, key=lambda o: o[0])
        return model, place

    # ---- one task = one packet from one sensor ----
    def task(self, u, name):
        self.created += 1
        t0 = self.env.now
        prio, dev = self.classify(name)
        self.recent[u].append(prio)
        deadline = DEADLINE[prio]
        diff = self.difficulty(u, name, dev)
        band = "easy" if diff < THETA_TRUE else "hard"
        serious = self.rng.random() < P_SERIOUS
        rank = PRIO_RANK[prio]
        phone = self.phones[u]
        final = None

        # ---- stage 1: M1 screens the triggering stream ----
        if self.scheme in ("proposed", "cascade"):
            with phone.request(priority=rank) as req:
                yield req
                yield self.env.timeout(t_phone("M1"))
            self.spend(u, e_phone("M1"))
            self.used["M1"] += 1
            final = "M1"
            if diff < self.th1:
                yield self.env.process(
                    self.deliver(u, name, prio, t0, serious, band, final))
                return

        # ---- stage 2: escalate. fusion models need all six streams ----
        t_left = deadline - (self.env.now - t0)
        model, place = self.select(u, diff, t_left)

        if place == "phone":
            with phone.request(priority=rank) as req:
                yield req
                yield self.env.timeout(t_phone(model))
            self.spend(u, e_phone(model))
            self.used[model] += 1
            final = model

        elif place == "lap":
            bits = FUSION_BITS if self.scheme in ("proposed", "cascade") \
                else SENSORS[name]["bits"]
            yield self.env.timeout(t_uplink(bits))
            self.spend(u, e_uplink(bits), tx=True)
            with self.lap.request(priority=rank) as req:
                yield req
                yield self.env.timeout(t_lap(model))
            self.energy_lap += e_lap(model)
            self.used[model] += 1
            final = model

        else:
            self.degraded += 1                       # keep M1's answer

        yield self.env.process(
            self.deliver(u, name, prio, t0, serious, band, final))

    # ---- result delivery ----
    def deliver(self, u, name, prio, t0, serious, band, final):
        if serious:                                  # alarm, straight out
            yield self.env.timeout(t_uplink(256))
            self.spend(u, e_uplink(256), tx=True)
        else:                                        # record, next LEO window
            self.spend(u, e_leo(SENSORS[name]["bits"]), tx=True)
            yield self.env.timeout(0.0)              # not deadline-bound
        lat = self.env.now - t0
        self.lat[prio].append(lat)
        a = ACC[band][final or "M1"]
        self.acc_sum += a
        self.done += 1
        if lat > DEADLINE[prio]:
            self.missed += 1
        else:
            self.acc_timely += a          # correct AND delivered in time

    def spend(self, u, j, tx=False):
        self.budget[u] -= j
        if tx:
            self.energy_tx += j
        else:
            self.energy_phone += j


def run(scheme, load_scale=1.0, crit_frac=0.0, theta_shift=0.0, seed=SEED):
    env = simpy.Environment()
    sim = Sim(env, scheme, load_scale, crit_frac, theta_shift, seed)
    env.run(until=SIM_TIME)

    total_e = sim.energy_phone + sim.energy_tx + sim.energy_lap
    all_lat = [x for v in sim.lat.values() for x in v]
    unfinished = max(sim.created - sim.done, 0)
    miss_rate = (sim.missed + unfinished) / sim.created if sim.created else float("nan")
    created = max(sim.created, 1)

    return dict(
        scheme=scheme, load=load_scale, crit_frac=crit_frac, theta=theta_shift,
        created=sim.created, done=sim.done,
        ee=sim.done / total_e if total_e > 0 else 0.0,
        e_task=total_e / sim.done if sim.done else float("nan"),
        e_phone_task=(sim.energy_phone + sim.energy_tx) / sim.done if sim.done else float("nan"),
        acc=sim.acc_sum / sim.done if sim.done else float("nan"),
        # tasks that never finished, or finished late, count as no decision
        timely_acc=sim.acc_timely / created,
        lat_mean=float(np.mean(all_lat)) if all_lat else float("nan"),
        lat_crit=float(np.mean(sim.lat["critical"])) if sim.lat["critical"] else float("nan"),
        miss=miss_rate,
        degraded=sim.degraded / sim.created if sim.created else 0.0,
        m1_share=(sim.created - sim.used["M2"] - sim.used["M3"]) / created,
        m2_share=sim.used["M2"] / created, m3_share=sim.used["M3"] / created,
        m1=sim.used["M1"], m2=sim.used["M2"], m3=sim.used["M3"],
    )


if __name__ == "__main__":
    for s in ("proposed", "cascade", "big_phone", "big_lap"):
        for L in (0.6, 1.4, 2.4):
            r = run(s, L)
            print(f"{s:10s} L={L:3.1f} EE={r['ee']:7.1f} e/task={r['e_task']*1e3:7.2f}mJ "
                  f"acc={r['acc']:.3f} miss={r['miss']*100:5.1f}% "
                  f"M2={r['m2_share']*100:4.1f}% M3={r['m3_share']*100:4.1f}% "
                  f"degr={r['degraded']*100:4.1f}%")

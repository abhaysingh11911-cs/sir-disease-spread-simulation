"""
Disease Spread Simulation Using the SIR Model
Course : Solving Problems Using Modelling and Simulation (SPMS) - NEC00091
Authors: Abhay Singh Chauhan (BTTC25O1003), Mohd Shafa At Khan (BTTC25O1040)

Run:  python sir_simulation.py
Output: PNG graphs in ./graphs and results.json

Model (population N = S + I + R, constant):
    dS/dt = -beta * S * I / N
    dI/dt =  beta * S * I / N - gamma * I
    dR/dt =  gamma * I
    R0 = beta / gamma
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------- parameters
N = 1000          # total population
I0 = 1            # initially infected
R0_INIT = 0       # initially recovered
BETA = 0.30       # transmission rate per day
GAMMA = 0.10      # recovery rate per day (infectious period = 10 days)
DAYS = 160
OUT = "graphs"
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- SIR core
def sir_rhs(t, y, beta, gamma, n):
    s, i, r = y
    new_inf = beta * s * i / n
    rec = gamma * i
    return [-new_inf, new_inf - rec, rec]


def solve_sir(beta=BETA, gamma=GAMMA, n=N, i0=I0, r0=R0_INIT, days=DAYS):
    """Deterministic solution using scipy's adaptive Runge-Kutta (RK45)."""
    t_eval = np.linspace(0, days, days * 4 + 1)
    y0 = [n - i0 - r0, i0, r0]
    sol = solve_ivp(sir_rhs, (0, days), y0, t_eval=t_eval,
                    args=(beta, gamma, n), rtol=1e-8, atol=1e-8)
    return sol.t, sol.y


def rk4_sir(beta=BETA, gamma=GAMMA, n=N, i0=I0, r0=R0_INIT, days=DAYS, dt=0.25):
    """Hand-written 4th-order Runge-Kutta, used to verify the library solver."""
    steps = int(days / dt)
    y = np.array([n - i0 - r0, i0, r0], dtype=float)
    ts, ys = [0.0], [y.copy()]
    f = lambda yy: np.array(sir_rhs(0, yy, beta, gamma, n))
    for k in range(steps):
        k1 = f(y)
        k2 = f(y + dt * k1 / 2)
        k3 = f(y + dt * k2 / 2)
        k4 = f(y + dt * k3)
        y = y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        ts.append((k + 1) * dt)
        ys.append(y.copy())
    return np.array(ts), np.array(ys).T


def stochastic_sir(beta=BETA, gamma=GAMMA, n=N, i0=I0, days=DAYS, seed=0):
    """Chain-binomial (discrete-time, daily) stochastic SIR simulation."""
    rng = np.random.default_rng(seed)
    s, i, r = n - i0, i0, 0
    out = [(s, i, r)]
    for _ in range(days):
        p_inf = 1 - np.exp(-beta * i / n)
        p_rec = 1 - np.exp(-gamma)
        new_i = rng.binomial(s, p_inf)
        new_r = rng.binomial(i, p_rec)
        s, i, r = s - new_i, i + new_i - new_r, r + new_r
        out.append((s, i, r))
    return np.arange(days + 1), np.array(out).T


def final_size(r0):
    """Solve the final-size relation  s_inf = exp(-R0 (1 - s_inf))  by iteration."""
    s = 0.5
    for _ in range(2000):
        s = np.exp(-r0 * (1 - s))
    return 1 - s


# ---------------------------------------------------------------- experiments
def main():
    plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.alpha": 0.3})
    results = {"params": {"N": N, "I0": I0, "beta": BETA, "gamma": GAMMA,
                          "R0": BETA / GAMMA, "days": DAYS}}

    # 1) Baseline deterministic run
    t, (S, I, R) = solve_sir()
    peak_idx = int(np.argmax(I))
    results["baseline"] = {
        "peak_infected": float(I[peak_idx]),
        "peak_day": float(t[peak_idx]),
        "final_recovered": float(R[-1]),
        "attack_rate_pct": float(R[-1] / N * 100),
        "theory_attack_rate_pct": float(final_size(BETA / GAMMA) * 100),
        "days": [float(x) for x in t[::20]],
        "S": [float(x) for x in S[::20]],
        "I": [float(x) for x in I[::20]],
        "R": [float(x) for x in R[::20]],
    }
    plt.figure(figsize=(9, 5))
    plt.plot(t, S, label="Susceptible", color="#2E86AB", lw=2.5)
    plt.plot(t, I, label="Infected", color="#D7263D", lw=2.5)
    plt.plot(t, R, label="Recovered", color="#3BB273", lw=2.5)
    plt.axvline(t[peak_idx], ls="--", color="gray")
    plt.title(f"SIR Model  (beta={BETA}, gamma={GAMMA}, R0={BETA/GAMMA:.1f})")
    plt.xlabel("Days"); plt.ylabel("People"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/01_baseline_sir.png", dpi=150); plt.close()

    # 2) RK4 vs scipy
    t4, y4 = rk4_sir()
    err = float(np.max(np.abs(np.interp(t, t4, y4[1]) - I)))
    results["rk4_max_abs_error_infected"] = err

    # 3) Effect of R0 (vary beta)
    sweep = {}
    plt.figure(figsize=(9, 5))
    for beta in (0.15, 0.20, 0.30, 0.50):
        tt, (s, i, r) = solve_sir(beta=beta, days=400)
        plt.plot(tt, i, lw=2.2, label=f"R0 = {beta/GAMMA:.1f}")
        k = int(np.argmax(i))
        sweep[f"{beta/GAMMA:.1f}"] = {"peak": float(i[k]), "peak_day": float(tt[k]),
                                      "attack_rate_pct": float(r[-1] / N * 100)}
    plt.xlim(0, 250); plt.title("Infected curve for different R0 values")
    plt.xlabel("Days"); plt.ylabel("Infected"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/02_r0_comparison.png", dpi=150); plt.close()
    results["r0_sweep"] = sweep

    # 4) Vaccination / herd immunity
    herd = 1 - GAMMA / BETA
    results["herd_immunity_threshold_pct"] = float(herd * 100)
    vacc = {}
    plt.figure(figsize=(9, 5))
    for cov in (0.0, 0.30, 0.50, 0.70):
        tt, (s, i, r) = solve_sir(r0=int(cov * N), days=400)
        plt.plot(tt, i, lw=2.2, label=f"{int(cov*100)}% vaccinated")
        k = int(np.argmax(i))
        vacc[f"{int(cov*100)}"] = {"peak": float(i[k]),
                                   "total_infected": float(r[-1] - cov * N)}
    plt.xlim(0, 250); plt.title("Effect of vaccination (herd immunity threshold = 66.7%)")
    plt.xlabel("Days"); plt.ylabel("Infected"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/03_vaccination.png", dpi=150); plt.close()
    results["vaccination"] = vacc

    # 5) Stochastic runs vs deterministic
    plt.figure(figsize=(9, 5))
    for seed in range(8):
        ts, (_, si, _) = stochastic_sir(i0=5, seed=seed)
        plt.plot(ts, si, color="#D7263D", alpha=0.35)
    td, (_, idet, _) = solve_sir(i0=5)
    plt.plot(td, idet, color="black", lw=2.5, label="Deterministic")
    plt.plot([], [], color="#D7263D", alpha=0.6, label="Stochastic runs")
    plt.title("Stochastic vs deterministic SIR (I0 = 5)")
    plt.xlabel("Days"); plt.ylabel("Infected"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/04_stochastic.png", dpi=150); plt.close()

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    b = results["baseline"]
    print(f"Peak infected: {b['peak_infected']:.0f} on day {b['peak_day']:.0f}")
    print(f"Attack rate: {b['attack_rate_pct']:.1f}%  (theory {b['theory_attack_rate_pct']:.1f}%)")
    print(f"RK4 max error: {err:.2e}")
    print("Herd immunity threshold: %.1f%%" % results["herd_immunity_threshold_pct"])


if __name__ == "__main__":
    main()
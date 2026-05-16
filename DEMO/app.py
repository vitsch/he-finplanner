"""
app.py — Entropy-Driven HE Planning Demo  [DEMO Steps D2–D7  /  complete]

Four-tab Streamlit prototype for the entropy-driven POMDP framework described in:
  Schetinin V. (2026). Enrollment Management under Deep Uncertainty.
  SSRN 6308238.  https://doi.org/10.2139/ssrn.6308238

  Tab 1  Year-by-Year        — step through 12 years manually (D3)
  Tab 2  12-Year Simulation  — shared-path comparison of 6 strategies (D4)
  Tab 3  Strategy Comparison — Monte Carlo Table 4 reproduction (D5)
  Tab 4  Calibrate Your Data — CSV upload, T/Ω re-estimation (D6)

Deployment:
  Local:           streamlit run DEMO/app.py  (from project root)
  Community Cloud: entry point = DEMO/app.py
                   requirements = DEMO/requirements.txt
                   params cache = DEMO/paper_params.npz  (committed, 2 KB)
"""

import sys
import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import streamlit as st

# ── path setup ────────────────────────────────────────────────────────────────
_DEMO_DIR  = os.path.dirname(os.path.abspath(__file__))
_PAPER_DIR = os.path.normpath(os.path.join(_DEMO_DIR, ".."))
sys.path.insert(0, _DEMO_DIR)
sys.path.insert(0, _PAPER_DIR)

from pomdp_core import (
    EntropyPOMDP, load_paper_params,
    STATE_NAMES, ACTION_NAMES, OBS_NAMES,
    N_STATES, N_ACTIONS, N_OBS,
    LOW, STABLE, GROWTH,
    CONSERVE, MAINTAIN, RECRUIT, INVEST, RATIONALISE,
    PAPER_DEFAULTS,
)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Entropy-Driven HE Planning",
    page_icon="\U0001f393",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── colour palette ────────────────────────────────────────────────────────────
ACTION_COLORS = {
    CONSERVE:    "#d62728",
    MAINTAIN:    "#1f77b4",
    RECRUIT:     "#2ca02c",
    INVEST:      "#9467bd",
    RATIONALISE: "#ff7f0e",
}
STATE_COLORS = ["#d62728", "#1f77b4", "#2ca02c"]   # Low / Stable / Growth
_H_MAX       = math.log(N_STATES)                   # ln 3 ≈ 1.099

# ── Simulation tab constants (D4) ─────────────────────────────────────────────
_SIM_STRATEGIES = [
    ("Unified Framework",  "unified"),
    ("BDT Baseline",       "baseline"),
    ("Myopic POMDP",       "myopic"),
    ("Robust Satisficing", "robust_satisficing"),
    ("DAPP",               "dapp"),
    ("RDM-Conservative",   "rdm"),
]

_SIM_LINE_STYLES = {
    "Unified Framework":  dict(lw=2.5, ls="-",  color="#1f77b4"),
    "BDT Baseline":       dict(lw=1.2, ls="--", color="#ff7f0e"),
    "Myopic POMDP":       dict(lw=1.2, ls="--", color="#2ca02c"),
    "Robust Satisficing": dict(lw=1.2, ls="--", color="#d62728"),
    "DAPP":               dict(lw=1.2, ls="--", color="#9467bd"),
    "RDM-Conservative":   dict(lw=1.2, ls="--", color="#8c564b"),
}

_PRESET_OBS = {
    "Post-2022 Shock":  [0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 2],
    "Shock & Recovery": [0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 3],
    "Sustained Growth": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    "Volatile":         [0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3],
}

_OBS_COLORS    = ["#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]  # Poor/Weak/Moderate/Strong
_PER_STEP_CRISIS = -8   # same as rg_pomdp_calibration.PER_STEP_CRISIS

# MC comparison constants (D5)
_MC_ORDER = [                           # matches run_mc internal strategy order
    "BDT Baseline", "Myopic POMDP", "Robust Satisficing",
    "DAPP", "RDM-Conservative", "Unified Framework",
]
_PAPER_TABLE4 = {                       # paper §6.1 Table 4 targets (n=500, seed=42)
    "Unified Framework":  dict(reward=349.2, distress=1.1),
    "Robust Satisficing": dict(reward=348.4, distress=2.0),
}


# ── HESA calibration cache (expensive; runs once per server session) ──────────
@st.cache_resource(show_spinner="Loading HESA-calibrated POMDP parameters …")
def _get_paper_params():
    # Fast path: pre-computed cache committed alongside app.py (for Community Cloud)
    cache = os.path.join(_DEMO_DIR, "paper_params.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return {k: (float(d[k]) if d[k].ndim == 0 else d[k]) for k in d.files}
    # Slow path: run the full HESA calibration pipeline (may use synthetic fallback)
    return load_paper_params(verbose=False)


def _build_model(params, sl):
    """Cheap: construct EntropyPOMDP from cached params + current slider values.

    When calib6_active is set, replaces T and Ω with the user-uploaded calibration.
    R_net is always from the HESA paper (reward cannot be estimated from
    enrolment counts alone).  T_true is set equal to T for custom calibrations
    (no separate model-uncertainty draw).
    """
    p = params
    if st.session_state.get("calib6_active", False):
        T      = st.session_state["calib6_T"]
        Omega  = st.session_state["calib6_Omega"]
        T_true = T
    else:
        T, Omega, T_true = p["T"], p["Omega"], p["T_true"]

    return EntropyPOMDP(
        T, Omega, p["R_net"],
        sl["b0"], T_true,
        gamma    = sl["gamma"],
        delta    = sl["delta"],
        u_min    = sl["u_min"],
        lambda_v = sl["lambda_v"],
        mu_ov    = sl["mu_ov"],
        n_sat    = sl["n_sat"],
    )


# ── session state ─────────────────────────────────────────────────────────────
_SS_DEFAULTS = dict(
    belief          = None,   # initialised from b0 on first render
    year            = 0,
    obs_history     = [],
    action_history  = [],
    belief_history  = [],
    entropy_history = [],
    episode_done    = False,
)


def _init_session(b0):
    for k, v in _SS_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if st.session_state.belief is None:
        st.session_state.belief = b0.copy()
        st.session_state.belief_history = [b0.copy()]


def _reset_session(b0):
    st.session_state.belief          = b0.copy()
    st.session_state.year            = 0
    st.session_state.obs_history     = []
    st.session_state.action_history  = []
    st.session_state.belief_history  = [b0.copy()]
    st.session_state.entropy_history = []
    st.session_state.episode_done    = False


# ── sidebar ───────────────────────────────────────────────────────────────────
def _render_sidebar():
    st.sidebar.title("Framework Parameters")
    st.sidebar.caption("All defaults match the paper (Schetinin 2026, §5).")
    st.sidebar.markdown("---")

    st.sidebar.subheader("Core")
    gamma = st.sidebar.slider(
        "γ — discount factor", 0.80, 0.99,
        float(PAPER_DEFAULTS["gamma"]), 0.01,
        help="Per-cycle future reward discount (paper: 0.95)")
    delta = st.sidebar.slider(
        "δ — SR3 risk tolerance", 0.01, 0.30,
        float(PAPER_DEFAULTS["delta"]), 0.01,
        help="SR3 gate: actions with P_sat < 1−δ eliminated (paper: 0.10)")
    u_min = st.sidebar.slider(
        "U_min — satisficing floor", -50.0, -5.0,
        float(PAPER_DEFAULTS["u_min"]), 1.0,
        help="Minimum acceptable cumulative reward over 12 years (paper: −25)")

    st.sidebar.subheader("SR4 / SR5 weights")
    lambda_v = st.sidebar.slider(
        "λ_VoI — SR4 weight", 0.5, 3.0,
        float(PAPER_DEFAULTS["lambda_v"]), 0.1,
        help="Bonus added to Q[Recruit] proportional to myopic VoI (paper: 1.5)")
    mu_ov = st.sidebar.slider(
        "μ_OV — SR5 weight", 0.5, 2.5,
        float(PAPER_DEFAULTS["mu_ov"]), 0.1,
        help="Penalty subtracted from Q[Invest/Rationalise] for option value (paper: 1.2)")

    st.sidebar.subheader("SR3 precision")
    n_sat = st.sidebar.slider(
        "n_sat — rollouts", 10, 60,
        int(PAPER_DEFAULTS["n_sat"]), 5,
        help="MC rollouts for satisficing probability (paper: 30; fewer = faster UI)")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Initial belief  b₀")
    st.sidebar.caption("Auto-normalised.")
    b0_low    = st.sidebar.slider("b₀(Low)",    0.01, 0.98, 0.159, 0.001)
    b0_stable = st.sidebar.slider("b₀(Stable)", 0.01, 0.98, 0.730, 0.001)
    b0_growth = st.sidebar.slider("b₀(Growth)", 0.01, 0.98, 0.111, 0.001)
    b0_raw    = np.array([b0_low, b0_stable, b0_growth])
    b0        = b0_raw / b0_raw.sum()
    st.sidebar.caption(
        f"→ Low {b0[0]:.3f} · Stable {b0[1]:.3f} · Growth {b0[2]:.3f}")

    st.sidebar.markdown("---")

    with st.sidebar.expander("About", expanded=False):
        st.markdown(
            "**Entropy-Driven HE Financial Planning**  \n"
            "POMDP + Bayesian Decision Theory framework  \n\n"
            "**Paper**  \n"
            "Schetinin V. (2026). *Enrollment Management under Deep "
            "Uncertainty: An Entropy-Indexed POMDP Framework for UK "
            "Higher Education Finance.*  \n"
            "[SSRN 6308238](https://doi.org/10.2139/ssrn.6308238)  \n\n"
            "**Data**  \n"
            "HESA enrolments 2014/15–2024/25 · ONS demographic projections "
            "· Home Office visa data  \n\n"
            "**Validated result**  \n"
            "Unified Framework: **349.2 reward · 1.1 % distress**  \n"
            "(n = 500 MC paths, Table 4)  \n\n"
            "**Author**  \n"
            "Vitaly Schetinin  \n"
            "University of Bedfordshire  \n\n"
            "*All processing runs in your local session — no data is "
            "stored or transmitted.*")

    return dict(gamma=gamma, delta=delta, u_min=u_min,
                lambda_v=lambda_v, mu_ov=mu_ov, n_sat=n_sat, b0=b0)


# ── helper plots ──────────────────────────────────────────────────────────────
def _fig_belief(b, title="Belief state  b_t"):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    bars = ax.bar(STATE_NAMES, b, color=STATE_COLORS, alpha=0.85,
                  edgecolor="k", linewidth=0.5)
    for bar, v in zip(bars, b):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                f"{v:.1%}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Probability", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def _fig_entropy_gauge(H):
    frac = H / _H_MAX
    if frac < 0.33:
        color, label = "#2ca02c", "Low uncertainty"
    elif frac < 0.70:
        color, label = "#ff7f0e", "Moderate uncertainty"
    else:
        color, label = "#d62728", "High uncertainty"
    fig, ax = plt.subplots(figsize=(3.4, 1.1))
    ax.barh(0, 1.0,  color="#e8e8e8", height=0.5)
    ax.barh(0, frac, color=color,     height=0.5, alpha=0.85)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.33, 0.70, 1.0])
    ax.set_xticklabels(["0", "0.36", "0.77", "ln 3"], fontsize=8)
    ax.set_title(f"H(b) = {H:.3f}   {label}", fontsize=10,
                 fontweight="bold", color=color)
    ax.spines[["top", "right", "left"]].set_visible(False)
    plt.tight_layout()
    return fig


def _fig_entropy_trajectory(e_hist):
    fig, ax = plt.subplots(figsize=(7, 2.4))
    xs = list(range(1, len(e_hist) + 1))
    ax.plot(xs, e_hist, "o-", color="#1f77b4", lw=1.8, ms=5)
    ax.axhline(_H_MAX * 0.70, color="#d62728", lw=0.8, ls="--",
               alpha=0.6, label="High-uncertainty zone")
    ax.axhline(_H_MAX * 0.33, color="#2ca02c", lw=0.8, ls="--",
               alpha=0.6, label="Low-uncertainty zone")
    ax.fill_between(xs, _H_MAX * 0.70, _H_MAX,
                    color="#d62728", alpha=0.05)
    ax.fill_between(xs, 0, _H_MAX * 0.33,
                    color="#2ca02c", alpha=0.05)
    ax.set_xlim(0.5, max(12.5, len(e_hist) + 0.5))
    ax.set_ylim(0, _H_MAX * 1.08)
    ax.set_xlabel("Planning year", fontsize=9)
    ax.set_ylabel("H(b_t)", fontsize=9)
    ax.set_title("Entropy trajectory — belief certainty over time", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


# ── Q-value table ─────────────────────────────────────────────────────────────
def _q_table(model, b, action, Q, gated):
    Q_base   = model.R_net @ b
    voi_val  = model.voi(b)
    ov_inv   = max(0.0, model.option_value(INVEST,      b))
    ov_rat   = max(0.0, model.option_value(RATIONALISE, b))

    rows = []
    for a, aname in enumerate(ACTION_NAMES):
        adj_str = "—"
        if a == RECRUIT:
            bonus = model.lambda_v * voi_val
            adj_str = f"+{bonus:.2f}  (SR4 VoI)"
        elif a == INVEST and ov_inv > 0:
            adj_str = f"−{model.mu_ov * ov_inv:.2f}  (SR5 OV)"
        elif a == RATIONALISE and ov_rat > 0:
            adj_str = f"−{model.mu_ov * ov_rat:.2f}  (SR5 OV)"

        rows.append({
            "Action":        ("▶ " if a == action else "  ") + aname,
            "Q base (R·b)":  f"{Q_base[a]:+.2f}",
            "Adjustment":    adj_str,
            "Q final":       f"{float(Q[a]):+.2f}",
            "SR3":           "✗ gated" if a in gated else "✓",
        })
    return pd.DataFrame(rows).set_index("Action")


# ── Tab 1: Year-by-Year (D3) ──────────────────────────────────────────────────
_OBS_DESC = {
    "Poor":     "Actual enrolment decline  (Δ% < 0)",
    "Weak":     "Slow positive growth, below regime threshold",
    "Moderate": "On-trend growth  (coincident with Stable regime)",
    "Strong":   "Above-trend growth  (coincident with Growth regime)",
}


def _tab_year_by_year(model, sliders):
    if st.session_state.get("calib6_active", False):
        st.info(
            "🔧 **Custom calibration active** — T and Ω replaced by your "
            "uploaded data (Tab 4).  R_net remains HESA paper values.")
    b0 = sliders["b0"]
    _init_session(b0)

    st.header("Year-by-Year Planning")
    st.caption(
        "The framework recommends a strategic posture at the **start** of each "
        "planning year based on the current belief state.  "
        "Record the actual enrolment outcome at year-end to update the belief "
        "for the following cycle.")

    # Reset button
    if st.button("↺  New episode", type="secondary"):
        _reset_session(b0)
        st.rerun()

    ss = st.session_state

    if ss.episode_done:
        st.success(
            f"Episode complete — {model.t_horizon} planning years simulated.  "
            "Press **↺ New episode** to restart.")
    else:
        t = ss.year
        b = ss.belief

        # ── Compute recommendation (start of year) ────────────────────────────
        rng    = np.random.default_rng(42 + t * 997)
        action, Q, gated, H, sat_prob = model.decide(
            b, t=t, rng=rng,
            use_sr2=False, use_sr3=True, use_sr4=True, use_sr5=True)

        # ── Year header ───────────────────────────────────────────────────────
        st.subheader(f"Year {t + 1} of {model.t_horizon}")

        # ── Belief + entropy panels ───────────────────────────────────────────
        c_belief, c_gauge, c_stats = st.columns([2, 2, 3])
        with c_belief:
            st.pyplot(_fig_belief(b), use_container_width=True)
        with c_gauge:
            st.pyplot(_fig_entropy_gauge(H), use_container_width=True)
        with c_stats:
            st.markdown("**Regime probabilities**")
            for i, name in enumerate(STATE_NAMES):
                st.progress(float(b[i]),
                            text=f"{name}: {b[i]:.1%}")
            st.markdown(
                f"**H(b)** = `{H:.4f}`  "
                f"(max = ln 3 ≈ 1.099)  \n"
                f"**Year:** {t + 1} / {model.t_horizon}")

        # ── Recommendation badge ──────────────────────────────────────────────
        st.markdown("---")
        col_rec, col_sr3 = st.columns([2, 3])
        with col_rec:
            color = ACTION_COLORS[action]
            st.markdown(
                f"<div style='background:{color};color:white;"
                f"padding:14px 22px;border-radius:8px;"
                f"font-size:1.25em;font-weight:bold;display:inline-block'>"
                f"Recommended posture: {ACTION_NAMES[action]}</div>",
                unsafe_allow_html=True)
        with col_sr3:
            if sat_prob is not None:
                if gated:
                    st.error(
                        f"🔒 **SR3 gate triggered** — P_sat = {sat_prob:.2f} "
                        f"< {1 - model.delta:.2f}.  "
                        f"Only **Conserve** meets the viability threshold.")
                else:
                    st.success(
                        f"✅ **SR3 passed** — P_sat = {sat_prob:.2f} "
                        f"≥ {1 - model.delta:.2f}.  "
                        f"All viable actions evaluated.")

        # ── Q-value breakdown ─────────────────────────────────────────────────
        with st.expander("Q-value breakdown (SR4 / SR5 adjustments)", expanded=False):
            st.dataframe(_q_table(model, b, action, Q, gated),
                         use_container_width=True)
            st.caption(
                "**Q base** = R_net · b (expected one-period reward).  "
                "**SR4** adds a VoI bonus to Recruit.  "
                "**SR5** subtracts an option-value penalty from Invest and Rationalise.")

        # ── Observation input (end of year) ───────────────────────────────────
        st.markdown("---")
        st.subheader("Record this year's enrolment outcome")
        st.caption(
            "Select the signal that best describes what the institution observed "
            "at the end of this planning year.  "
            "This updates the belief for Year " + str(t + 2) + ".")

        oc1, oc2 = st.columns([1, 2])
        with oc1:
            obs_label = st.radio(
                "Enrolment signal",
                options=OBS_NAMES, index=2,
                label_visibility="collapsed")
        with oc2:
            st.info(f"**{obs_label}:** {_OBS_DESC[obs_label]}")

        if st.button(
                f"Record '{obs_label}' and advance to Year {t + 2}",
                type="primary"):
            obs_idx = OBS_NAMES.index(obs_label)
            b_new   = model.belief_update(b, obs_idx)

            ss.action_history.append(action)
            ss.obs_history.append(obs_idx)
            ss.entropy_history.append(H)
            ss.belief_history.append(b_new.copy())
            ss.belief = b_new
            ss.year  += 1

            if ss.year >= model.t_horizon:
                ss.episode_done = True
            st.rerun()

    # ── SR7 alert ─────────────────────────────────────────────────────────────
    e = ss.entropy_history
    if len(e) >= 3 and e[-3] < e[-2] < e[-1]:
        st.warning(
            "⚠️ **SR7 signal:** Belief entropy has increased for 3 consecutive "
            "years — this may indicate a structural regime shift.  "
            "Consider activating Dirichlet transition adaptation (SR7).")

    # ── History panels ────────────────────────────────────────────────────────
    if ss.action_history:
        st.markdown("---")
        st.subheader("Planning history")

        # Action / observation timeline
        cols = st.columns(len(ss.action_history))
        for i, (a, o) in enumerate(zip(ss.action_history, ss.obs_history)):
            with cols[i]:
                st.markdown(
                    f"<div style='background:{ACTION_COLORS[a]};color:white;"
                    f"padding:5px 4px;border-radius:5px;"
                    f"font-size:0.72em;text-align:center;line-height:1.4'>"
                    f"<b>Yr {i + 1}</b><br>{ACTION_NAMES[a]}<br>"
                    f"<span style='opacity:0.85'>{OBS_NAMES[o]}</span></div>",
                    unsafe_allow_html=True)

        # Entropy trajectory
        if len(e) >= 2:
            st.pyplot(_fig_entropy_trajectory(e), use_container_width=True)

        # Belief trajectory table
        with st.expander("Belief state history", expanded=False):
            rows = []
            for yr, bel in enumerate(ss.belief_history):
                rows.append({
                    "Year": yr,
                    "P(Low)":    f"{bel[LOW]:.3f}",
                    "P(Stable)": f"{bel[STABLE]:.3f}",
                    "P(Growth)": f"{bel[GROWTH]:.3f}",
                    "H(b)":      f"{model.entropy(bel):.4f}",
                })
            st.dataframe(pd.DataFrame(rows).set_index("Year"),
                         use_container_width=True)


# ── Simulation helpers (D4) ───────────────────────────────────────────────────

def _generate_path(model, seed):
    """Sample one T_horizon-year (states, observations) trajectory from T_true and Omega."""
    rng = np.random.default_rng(seed)
    s   = rng.choice(N_STATES, p=model.b0)
    states, observations = [], []
    for _ in range(model.t_horizon):
        states.append(int(s))
        o = rng.choice(N_OBS, p=model.Omega[s])
        observations.append(int(o))
        s = rng.choice(N_STATES, p=model.T_true[s])
    return states, observations


def _run_all_on_path(model, states, observations, dec_seed):
    """Run all 6 strategies on a fixed (states, observations) path.

    Belief updates are identical across strategies (same obs sequence).
    SR3 decision RNGs are seeded separately per strategy.
    """
    out = {}
    for idx, (label, key) in enumerate(_SIM_STRATEGIES):
        rng   = np.random.default_rng(dec_seed * 100 + idx * 10000)
        b     = model.b0.copy()
        actions, beliefs, entropies, raw_r = [], [], [], []
        beliefs.append(b.copy())
        entropies.append(model.entropy(b))
        obs_hist_local = []

        for t, (true_s, obs) in enumerate(zip(states, observations)):
            if key == "unified":
                a, _Q, _gated, _H, _sp = model.decide(
                    b, t=t, rng=rng,
                    use_sr2=False, use_sr3=True, use_sr4=True, use_sr5=True)
            else:
                a = model._action_comparator(key, b, obs_hist_local, t, rng)

            actions.append(int(a))
            raw_r.append(float(model.R_net[a, true_s]))
            obs_hist_local.append(obs)
            b = model.belief_update(b, obs)
            beliefs.append(b.copy())
            entropies.append(model.entropy(b))

        disc = [model.gamma ** t * r for t, r in enumerate(raw_r)]
        cum  = list(np.cumsum(disc).tolist())
        out[label] = dict(
            actions=actions,
            beliefs=beliefs,
            entropies=entropies,
            raw_rewards=raw_r,
            cumulative_rewards=cum,
            total_reward=cum[-1] if cum else 0.0,
            distress_count=sum(1 for r in raw_r if r < _PER_STEP_CRISIS),
        )
    return out


def _fig_state_obs_strip(states, observations):
    """2-row colour strip: true regime (top) + observed signal (bottom)."""
    T     = len(states)
    years = list(range(1, T + 1))

    fig, axes = plt.subplots(2, 1, figsize=(9, 1.8), sharex=True,
                             gridspec_kw={"hspace": 0.05})
    for ax, colors, labels, ylabel in [
        (axes[0],
         [STATE_COLORS[s] for s in states],
         [STATE_NAMES[s][:3] for s in states],
         "Regime"),
        (axes[1],
         [_OBS_COLORS[o] for o in observations],
         [OBS_NAMES[o][:3] for o in observations],
         "Signal"),
    ]:
        ax.bar(years, [1] * T, color=colors, alpha=0.85, width=0.90,
               edgecolor="white", linewidth=0.5)
        for i, lbl in enumerate(labels):
            ax.text(years[i], 0.5, lbl, ha="center", va="center",
                    fontsize=7.5, fontweight="bold", color="white")
        ax.set_yticks([])
        ax.set_ylabel(ylabel, fontsize=8, rotation=0, labelpad=45, va="center")
        ax.set_ylim(0, 1)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    axes[1].set_xticks(years)
    axes[1].set_xticklabels(years, fontsize=8)
    axes[1].set_xlabel("Planning year", fontsize=9)
    plt.tight_layout()
    return fig


def _fig_action_heatmap(results):
    """6-strategy × 12-year action colour heatmap."""
    labels = [lbl for lbl, _ in _SIM_STRATEGIES]
    T      = len(results[labels[0]]["actions"])

    rgb = np.ones((len(labels), T, 3))
    for i, label in enumerate(labels):
        for j, a in enumerate(results[label]["actions"]):
            rgb[i, j] = mcolors.to_rgb(ACTION_COLORS[a])

    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.imshow(rgb, aspect="auto",
              extent=[0.5, T + 0.5, len(labels) - 0.5, -0.5])
    for i, label in enumerate(labels):
        for j, a in enumerate(results[label]["actions"]):
            ax.text(j + 1, i, ACTION_NAMES[a][:3],
                    ha="center", va="center", fontsize=6.5,
                    fontweight="bold", color="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks(range(1, T + 1))
    ax.set_xticklabels(range(1, T + 1), fontsize=8)
    ax.set_xlabel("Planning year", fontsize=9)
    ax.set_title("Strategy × Year — recommended posture",
                 fontsize=10, fontweight="bold")
    patches = [mpatches.Patch(color=ACTION_COLORS[a], label=ACTION_NAMES[a])
               for a in range(N_ACTIONS)]
    ax.legend(handles=patches, loc="lower right", fontsize=7, ncol=5,
              framealpha=0.9)
    plt.tight_layout()
    return fig


def _fig_cumulative_rewards_sim(results, gamma):
    """Cumulative discounted reward curves for all 6 strategies."""
    labels = [lbl for lbl, _ in _SIM_STRATEGIES]
    T      = len(results[labels[0]]["cumulative_rewards"])
    years  = list(range(1, T + 1))

    fig, ax = plt.subplots(figsize=(9, 3.4))
    for label in labels:
        style = _SIM_LINE_STYLES[label]
        cum   = results[label]["cumulative_rewards"]
        ax.plot(years[:len(cum)], cum, label=label, **style)
    ax.set_xlabel("Planning year", fontsize=9)
    ax.set_ylabel("Cumulative discounted reward", fontsize=9)
    ax.set_title(f"Reward trajectories  (γ = {gamma:.2f})",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def _fig_entropy_traj_sim(results):
    """Entropy trajectory (shared across all strategies on a fixed obs path)."""
    entropies = results["Unified Framework"]["entropies"]
    xs = list(range(len(entropies)))

    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.plot(xs, entropies, "o-", color="#1f77b4", lw=1.8, ms=5)
    ax.axhline(_H_MAX * 0.70, color="#d62728", lw=0.8, ls="--", alpha=0.6,
               label="High-uncertainty zone")
    ax.axhline(_H_MAX * 0.33, color="#2ca02c", lw=0.8, ls="--", alpha=0.6,
               label="Low-uncertainty zone")
    ax.fill_between(xs, _H_MAX * 0.70, _H_MAX,
                    color="#d62728", alpha=0.05)
    ax.fill_between(xs, 0, _H_MAX * 0.33,
                    color="#2ca02c", alpha=0.05)
    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel("H(b_t)", fontsize=9)
    ax.set_title("Belief entropy H(b_t) — identical for all strategies on this path",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


# ── Tab 2: 12-Year Simulation (D4) ───────────────────────────────────────────
def _tab_simulation(model, sliders):
    if st.session_state.get("calib6_active", False):
        st.info(
            "🔧 **Custom calibration active** — T and Ω replaced by your "
            "uploaded data (Tab 4).  R_net remains HESA paper values.")
    st.header("12-Year Simulation")
    st.caption(
        "All 6 strategies run on a **shared observation path** (common random numbers) "
        "so reward differences reflect strategy quality, not path luck.  "
        "True regime is sampled from the calibrated T_true; observations are "
        "either model-consistent, pre-defined, or custom.")

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([3, 2, 2])
    with ctrl1:
        scenario = st.selectbox(
            "Scenario",
            ["Random (model-consistent)"] + list(_PRESET_OBS.keys()) + ["Custom"],
            index=0, key="sim4_scenario_sel")
    with ctrl2:
        path_seed = int(st.number_input(
            "Path seed", min_value=0, max_value=9999, value=42, step=1,
            key="sim4_seed_input"))
    with ctrl3:
        run_btn = st.button("▶  Run Simulation", type="primary",
                            use_container_width=True, key="sim4_run")

    # ── Custom observation editor ─────────────────────────────────────────────
    custom_obs = None
    if scenario == "Custom":
        st.markdown("**Select the observation for each of the 12 planning years:**")
        cols6 = st.columns(6)
        custom_obs = []
        for yr in range(model.t_horizon):
            with cols6[yr % 6]:
                lbl = st.selectbox(
                    f"Yr {yr + 1}", OBS_NAMES, index=2,
                    key=f"sim4_cobs_{yr}")
                custom_obs.append(OBS_NAMES.index(lbl))

    # ── Session state initialisation ──────────────────────────────────────────
    for k in ("sim4_results", "sim4_states", "sim4_observations"):
        if k not in st.session_state:
            st.session_state[k] = None

    # ── Execute on button press ───────────────────────────────────────────────
    if run_btn:
        with st.spinner("Running 6 strategies on shared path …"):
            states, obs_random = _generate_path(model, path_seed)

            if scenario == "Random (model-consistent)":
                observations = obs_random
            elif scenario == "Custom":
                observations = list(custom_obs)
            else:
                observations = list(_PRESET_OBS[scenario])

            results = _run_all_on_path(model, states, observations, path_seed)
            st.session_state.sim4_results      = results
            st.session_state.sim4_states       = states
            st.session_state.sim4_observations = observations

    # ── Display results ───────────────────────────────────────────────────────
    if st.session_state.sim4_results is not None:
        results      = st.session_state.sim4_results
        states       = st.session_state.sim4_states
        observations = st.session_state.sim4_observations

        st.markdown("---")
        st.subheader("Simulated path")
        st.caption(
            "Top row: true hidden regime (seed-fixed sample from T_true).  "
            "Bottom row: enrolment signal that drives belief updates.")
        st.pyplot(_fig_state_obs_strip(states, observations),
                  use_container_width=True)

        st.markdown("---")
        st.subheader("Strategy comparison")
        best_r = max(results[l]["total_reward"]   for l in results)
        best_d = min(results[l]["distress_count"] for l in results)
        rows = []
        for label, _ in _SIM_STRATEGIES:
            r   = results[label]
            dom = ACTION_NAMES[max(range(N_ACTIONS),
                                   key=lambda a, acts=r["actions"]: acts.count(a))]
            flag_r = " ★" if abs(r["total_reward"]   - best_r) < 0.1 else ""
            flag_d = " ★" if r["distress_count"] == best_d       else ""
            rows.append({
                "Strategy":        label,
                "Total reward":    f"{r['total_reward']:.1f}{flag_r}",
                "Distress years":  f"{r['distress_count']}{flag_d}",
                "Dominant action": dom,
            })
        st.dataframe(pd.DataFrame(rows).set_index("Strategy"),
                     use_container_width=True)
        st.caption("★ = best in column on this path.")

        st.markdown("---")
        st.subheader("Reward trajectories")
        st.pyplot(_fig_cumulative_rewards_sim(results, model.gamma),
                  use_container_width=True)

        st.subheader("Action map")
        st.pyplot(_fig_action_heatmap(results), use_container_width=True)

        st.subheader("Belief entropy")
        st.pyplot(_fig_entropy_traj_sim(results), use_container_width=True)


# ── MC comparison plot functions (D5) ────────────────────────────────────────

def _fig_mc_rewards(results, n_paths):
    """Horizontal bar chart: mean reward ± 95% CI for all 6 strategies."""
    labels = [lbl for lbl, _ in _SIM_STRATEGIES]
    means  = [results[l]["mean_reward"]  for l in labels]
    err_lo = [results[l]["mean_reward"] - results[l]["ci_lo"] for l in labels]
    err_hi = [results[l]["ci_hi"] - results[l]["mean_reward"] for l in labels]
    colors = [_SIM_LINE_STYLES[l]["color"] for l in labels]

    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ys = list(range(len(labels)))
    ax.barh(ys, means, color=colors, alpha=0.82, height=0.60,
            xerr=[err_lo, err_hi],
            error_kw=dict(lw=1.2, capsize=4, ecolor="#555"))
    for i, m in enumerate(means):
        ax.text(m + 0.8, i, f"{m:.1f}", va="center", fontsize=8.5)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()   # Unified Framework at the top
    ax.set_xlabel("Mean discounted reward (12-yr horizon)", fontsize=9)
    ax.set_title(f"Mean reward ± 95% CI  (n = {n_paths} paths per strategy)",
                 fontsize=10, fontweight="bold")
    ax.axvline(349.2, color="#1f77b4", ls=":", lw=1.0, alpha=0.7,
               label="Paper target — Unified (n=500)")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def _fig_mc_distress(results, n_paths):
    """Horizontal bar chart: crisis probability per strategy."""
    labels = [lbl for lbl, _ in _SIM_STRATEGIES]
    dists  = [results[l]["distress_pct"] for l in labels]
    colors = [_SIM_LINE_STYLES[l]["color"] for l in labels]

    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ys = list(range(len(labels)))
    ax.barh(ys, dists, color=colors, alpha=0.75, height=0.60)
    for i, d in enumerate(dists):
        ax.text(d + 0.04, i, f"{d:.1f}%", va="center", fontsize=8.5)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Crisis probability  (% years with R < −8)", fontsize=9)
    ax.set_title(f"Crisis probability  (n = {n_paths} paths per strategy)",
                 fontsize=10, fontweight="bold")
    ax.axvline(1.1, color="#1f77b4", ls=":", lw=1.0, alpha=0.7,
               label="Paper target — Unified (n=500)")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def _fig_mc_pareto(results):
    """Pareto frontier scatter: reward vs. crisis probability."""
    labels     = [lbl for lbl, _ in _SIM_STRATEGIES]
    means      = [results[l]["mean_reward"]  for l in labels]
    distresses = [results[l]["distress_pct"] for l in labels]
    colors     = [_SIM_LINE_STYLES[l]["color"] for l in labels]

    # Non-dominated: no other point has both higher reward and lower distress
    is_pareto = []
    for i in range(len(labels)):
        dominated = any(
            means[j] >= means[i] and distresses[j] <= distresses[i]
            and (means[j] > means[i] or distresses[j] < distresses[i])
            for j in range(len(labels)) if j != i
        )
        is_pareto.append(not dominated)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for i, label in enumerate(labels):
        ax.scatter(distresses[i], means[i],
                   color=colors[i],
                   s=(14 ** 2 if is_pareto[i] else 9 ** 2),
                   marker=("*" if is_pareto[i] else "o"),
                   zorder=3, edgecolors="k", linewidths=0.6)
        ax.annotate(label, (distresses[i], means[i]),
                    textcoords="offset points", xytext=(7, 3), fontsize=8)

    # Paper target crosshair
    ax.axhline(349.2, color="#aaa", ls=":", lw=0.8, alpha=0.6)
    ax.axvline(1.1,   color="#aaa", ls=":", lw=0.8, alpha=0.6)
    ax.scatter([1.1], [349.2], marker="x", s=100, color="k", zorder=5,
               linewidths=2.0, label="Paper Table 4 target (n=500)")

    ax.set_xlabel("Crisis probability (%)", fontsize=9)
    ax.set_ylabel("Mean discounted reward", fontsize=9)
    ax.set_title(
        "Pareto frontier — reward vs. crisis probability\n"
        "★ = non-dominated strategy",
        fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


# ── Tab 3: Strategy Comparison — MC (D5) ─────────────────────────────────────
def _tab_comparison(model, sliders):
    if st.session_state.get("calib6_active", False):
        st.info(
            "🔧 **Custom calibration active** — T and Ω replaced by your "
            "uploaded data (Tab 4).  R_net remains HESA paper values.")
    st.header("Strategy Comparison  (Monte Carlo)")
    st.caption(
        "Compare all 6 strategies over many independent paths and reproduce "
        "the Pareto dominance result from Table 4 of the paper.  "
        "Use n = 500, seed = 42 to match the paper exactly.")

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([3, 2, 2])
    with ctrl1:
        n_paths = st.slider(
            "n_paths — paths per strategy", 50, 500, 100, 50,
            help="500 reproduces Table 4; lower values run faster",
            key="sim5_n_paths")
    with ctrl2:
        mc_seed = int(st.number_input(
            "Seed", min_value=0, max_value=9999, value=42, step=1,
            key="sim5_seed"))
    with ctrl3:
        run_btn = st.button("▶  Run MC", type="primary",
                            use_container_width=True, key="sim5_run")

    st.caption(
        "Lower n_paths = faster run.  "
        "Paper targets (n=500, seed=42): "
        "Unified Framework = **349.2 reward · 1.1% distress**; "
        "Robust Satisficing = **348.4 · 2.0%**.")

    # ── Session state ─────────────────────────────────────────────────────────
    for k in ("sim5_results", "sim5_n_paths_used", "sim5_seed_used"):
        if k not in st.session_state:
            st.session_state[k] = None

    # ── Run MC ────────────────────────────────────────────────────────────────
    if run_btn:
        prog_bar = st.progress(0, text="Initialising …")

        def _progress(label, done, total):
            if done % max(1, total // 20) == 0 or done == total:
                try:
                    si = _MC_ORDER.index(label)
                except ValueError:
                    si = 0
                frac = (si * total + done) / (6 * total)
                prog_bar.progress(min(frac, 1.0),
                                  text=f"{label}  {done}/{total} paths …")

        results = model.run_mc(n_paths=n_paths, seed=mc_seed,
                               progress_fn=_progress)
        prog_bar.empty()

        st.session_state.sim5_results      = results
        st.session_state.sim5_n_paths_used = n_paths
        st.session_state.sim5_seed_used    = mc_seed

    # ── Display ───────────────────────────────────────────────────────────────
    if st.session_state.sim5_results is not None:
        results   = st.session_state.sim5_results
        n_used    = st.session_state.sim5_n_paths_used
        seed_used = st.session_state.sim5_seed_used

        st.markdown("---")
        st.subheader(f"Table 4 reproduction  (n = {n_used}, seed = {seed_used})")

        rows = []
        for label, _ in _SIM_STRATEGIES:
            r     = results[label]
            paper = _PAPER_TABLE4.get(label, {})
            r_str = (f"{r['mean_reward']:.1f}  (paper: {paper['reward']:.1f})"
                     if paper else f"{r['mean_reward']:.1f}")
            d_str = (f"{r['distress_pct']:.1f}%  (paper: {paper['distress']:.1f}%)"
                     if paper else f"{r['distress_pct']:.1f}%")
            rows.append({
                "Strategy":    label,
                "Mean reward": r_str,
                "95% CI":      f"[{r['ci_lo']:.1f}, {r['ci_hi']:.1f}]",
                "Distress %":  d_str,
            })
        st.dataframe(pd.DataFrame(rows).set_index("Strategy"),
                     use_container_width=True)
        if n_used < 500:
            st.caption(
                f"MC variance at n={n_used} is normal — run with n=500 to "
                "reproduce Table 4 exactly.")

        st.markdown("---")
        st.subheader("Reward distribution")
        st.pyplot(_fig_mc_rewards(results, n_used), use_container_width=True)

        st.subheader("Crisis probability")
        st.pyplot(_fig_mc_distress(results, n_used), use_container_width=True)

        st.subheader("Pareto frontier")
        st.caption(
            "★ = non-dominated (no other strategy achieves both higher "
            "reward and lower distress).  "
            "✕ = paper Table 4 target (n=500, seed=42).")
        st.pyplot(_fig_mc_pareto(results), use_container_width=True)


# ── Calibration helpers (D6) ──────────────────────────────────────────────────

def _calibrate_enrollment_csv(df, use_paper_thresholds, low_t_paper, high_t_paper):
    """Estimate T, Omega, b0 from a single-institution enrollment DataFrame.

    Mirrors the paper pipeline (§4): threshold-based regime assignment +
    Laplace-1 smoothed transition and emission counting.

    Returns (T, Omega, b0, states, pct, low_t, high_t).
    Raises ValueError on invalid input.
    """
    df  = df.sort_values("year").reset_index(drop=True)
    enr = df["enrollment_total"].values.astype(float)

    if len(enr) < 4:
        raise ValueError(
            "Need at least 4 data rows (gives 3 annual % changes "
            "and 2 regime transitions).")
    if np.any(enr <= 0):
        raise ValueError("All enrollment_total values must be positive.")

    pct = np.diff(enr) / enr[:-1] * 100   # length N-1

    if use_paper_thresholds:
        low_t, high_t = low_t_paper, high_t_paper
    else:
        low_t  = float(np.percentile(pct, 33))
        high_t = float(np.percentile(pct, 67))
        if high_t <= low_t:
            high_t = low_t + 0.01   # guard against flat data

    # Regime assignment (same logic as assign_states_threshold)
    states = np.where(pct < low_t, LOW,
              np.where(pct >= high_t, GROWTH, STABLE))

    # Transition matrix (Laplace-1 smoothing)
    cnt_T = np.ones((N_STATES, N_STATES))
    for i in range(len(states) - 1):
        cnt_T[states[i], states[i + 1]] += 1
    T = cnt_T / cnt_T.sum(axis=1, keepdims=True)

    # Emission matrix (Laplace-1 smoothing, same observation thresholds)
    raw_O = np.zeros((N_STATES, N_OBS), dtype=int)
    for delta, s in zip(pct, states):
        if   delta < 0.0:    o = POOR
        elif delta < low_t:  o = WEAK
        elif delta < high_t: o = MODERATE
        else:                o = STRONG
        raw_O[s, o] += 1
    cnt_O = raw_O + 1
    Omega = cnt_O / cnt_O.sum(axis=1, keepdims=True)

    # b0: empirical state frequency
    b0 = np.zeros(N_STATES)
    for s in states:
        b0[s] += 1
    b0 /= b0.sum()

    return T, Omega, b0, states, pct, low_t, high_t


def _fig_regime_timeline(years_mid, pct, states, low_t, high_t):
    """2-row figure: regime colour strip (top) + annual % change bars (bottom)."""
    n = len(states)
    fw = max(7.0, min(14.0, n * 0.85))
    xs = list(range(n))

    fig, axes = plt.subplots(2, 1, figsize=(fw, 2.8),
                             sharex=True,
                             gridspec_kw={"hspace": 0.05,
                                          "height_ratios": [1, 2.5]})

    # Row 0: colour strip
    ax0 = axes[0]
    for i, s in enumerate(states):
        ax0.bar(i, 1, color=STATE_COLORS[s], alpha=0.85, width=0.92,
                edgecolor="white", linewidth=0.5)
        ax0.text(i, 0.5, STATE_NAMES[s][:3], ha="center", va="center",
                 fontsize=7.5, fontweight="bold", color="white")
    ax0.set_yticks([])
    ax0.set_ylabel("Regime", fontsize=8, rotation=0, labelpad=42, va="center")
    ax0.set_ylim(0, 1)
    ax0.spines[["top", "right", "left", "bottom"]].set_visible(False)

    # Row 1: % change bars
    ax1 = axes[1]
    ax1.bar(xs, pct, color=[STATE_COLORS[s] for s in states],
            alpha=0.70, width=0.80)
    ax1.axhline(low_t,  color="#d62728", ls="--", lw=1.0, alpha=0.8,
                label=f"Low threshold  ({low_t:.2f}%)")
    ax1.axhline(high_t, color="#2ca02c", ls="--", lw=1.0, alpha=0.8,
                label=f"Growth threshold  ({high_t:.2f}%)")
    ax1.axhline(0, color="k", lw=0.5, alpha=0.35)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([str(y) for y in years_mid],
                        fontsize=8, rotation=45, ha="right")
    ax1.set_ylabel("Enrolment Δ%", fontsize=8)
    ax1.legend(fontsize=7, loc="upper right")
    ax1.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Regime classification from uploaded data",
                 fontsize=10, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def _fig_matrix_heatmap(M, row_labels, col_labels, title, cmap="Blues"):
    """Annotated probability heatmap for small matrices (T or Omega)."""
    nr, nc = M.shape
    fig, ax = plt.subplots(figsize=(max(3.2, nc * 1.1), max(2.2, nr * 0.9)))
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(nr):
        for j in range(nc):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if v > 0.60 else "black")
    ax.set_xticks(range(nc))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(nr))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title, fontsize=9, fontweight="bold")
    plt.tight_layout()
    return fig


# ── Tab 4: Calibrate Your Data (D6) ──────────────────────────────────────────
def _tab_calibrate(model, sliders):
    st.header("Calibrate Your Data")
    st.caption(
        "Upload your institution's annual enrolment history to re-estimate "
        "T and Ω using the same Laplace-smoothed pipeline as the paper.  "
        "Once activated, all other tabs use your matrices instead of the "
        "HESA Russell Group parameters.  "
        "All processing runs in your local session — no data is stored or transmitted.")

    # ── CSV format guide ──────────────────────────────────────────────────────
    with st.expander("CSV format and example", expanded=False):
        st.markdown(
            "Required columns:\n\n"
            "| year | enrollment_total |\n"
            "|------|------------------|\n"
            "| 2014 | 10 000 |\n"
            "| 2015 | 10 250 |\n"
            "| …    | …      |\n"
            "| 2024 | 11 800 |\n\n"
            "- **year** — calendar year, ascending, no gaps\n"
            "- **enrollment_total** — FTE headcount, positive integer\n"
            "- Minimum **4 rows** (gives 3 annual % changes and 2 transitions)")
        example = "year,enrollment_total\n" + "\n".join(
            f"{y},{int(10000 * (1 + 0.02 * (y - 2014)))}"
            for y in range(2014, 2025))
        st.download_button(
            "Download example CSV",
            data=example,
            file_name="example_enrollment.csv",
            mime="text/csv",
            key="calib6_dl_example")

    # ── File upload ───────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload enrolment CSV",
        type=["csv"],
        help="Required columns: year, enrollment_total",
        key="calib6_upload")

    # ── Session state init ────────────────────────────────────────────────────
    for k, default in [("calib6_T",     None),
                       ("calib6_Omega", None),
                       ("calib6_b0",    None),
                       ("calib6_active", False)]:
        if k not in st.session_state:
            st.session_state[k] = default

    if uploaded is None:
        if st.session_state.calib6_active:
            st.success(
                "Custom calibration is **active** from a previous upload.  "
                "Upload a new file to replace it, or press **Revert** below.")
            if st.button("Revert to HESA paper parameters",
                         key="calib6_deactivate_nofile"):
                st.session_state.calib6_active = False
                st.rerun()
        else:
            st.info(
                "Upload a CSV to begin.  Once calibrated, activate the custom "
                "T and Ω in all tabs with one button press.")
        return

    # ── Parse CSV ─────────────────────────────────────────────────────────────
    try:
        df      = pd.read_csv(uploaded)
        missing = {"year", "enrollment_total"} - set(df.columns)
        if missing:
            st.error(f"Missing column(s): {', '.join(sorted(missing))}.  "
                     "Expected: `year`, `enrollment_total`.")
            return
        df = df[["year", "enrollment_total"]].dropna()
        df["year"]             = df["year"].astype(int)
        df["enrollment_total"] = df["enrollment_total"].astype(float)
    except Exception as exc:
        st.error(f"Could not parse CSV: {exc}")
        return

    # ── Threshold selector ────────────────────────────────────────────────────
    params   = _get_paper_params()
    low_t_p  = float(params["low_t"])
    high_t_p = float(params["high_t"])

    thresh_choice = st.radio(
        "Regime thresholds",
        [f"HESA paper  ({low_t_p:.2f}% / {high_t_p:.2f}%)  — recommended",
         "Data-derived  (33rd / 67th percentile of uploaded data)"],
        index=0, horizontal=True, key="calib6_thresh")
    use_paper_t = thresh_choice.startswith("HESA")

    # ── Run calibration ───────────────────────────────────────────────────────
    try:
        T_c, Omega_c, b0_c, states_c, pct_c, low_t, high_t = \
            _calibrate_enrollment_csv(df, use_paper_t, low_t_p, high_t_p)
    except ValueError as exc:
        st.error(str(exc))
        return

    years_all = df["year"].values
    years_mid = years_all[1:]   # pct change labelled by destination year

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Calibration summary")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Data span",      f"{years_all[0]}–{years_all[-1]}")
    mc2.metric("Annual changes", len(pct_c))
    mc3.metric("Transitions",    len(states_c) - 1)
    mc4.metric("Thresholds",     f"{low_t:.2f}% / {high_t:.2f}%")

    regime_counts = {STATE_NAMES[s]: int((states_c == s).sum())
                     for s in range(N_STATES)}
    st.caption(
        "Regime year counts — "
        + "  ·  ".join(f"**{k}:** {v}" for k, v in regime_counts.items()))

    # ── Regime timeline ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Regime timeline")
    st.pyplot(_fig_regime_timeline(years_mid, pct_c, states_c, low_t, high_t),
              use_container_width=True)

    # ── T and Omega heatmaps side-by-side ─────────────────────────────────────
    st.markdown("---")
    st.subheader("Estimated matrices")
    col_T, col_O = st.columns(2)
    with col_T:
        st.pyplot(
            _fig_matrix_heatmap(
                T_c, STATE_NAMES, STATE_NAMES,
                "Transition matrix T  (Laplace-1 smoothed)"),
            use_container_width=True)
        st.caption("Row = current regime → Column = next regime.")
    with col_O:
        st.pyplot(
            _fig_matrix_heatmap(
                Omega_c, STATE_NAMES, OBS_NAMES,
                "Emission matrix Ω  (Laplace-1 smoothed)",
                cmap="Greens"),
            use_container_width=True)
        st.caption("Row = true regime → Column = observed enrolment signal.")

    # ── Initial belief from data ──────────────────────────────────────────────
    st.subheader("Empirical b₀ from uploaded data")
    st.caption(
        "Derived from regime frequency in the uploaded series.  "
        "The sidebar b₀ sliders override the initial belief in the "
        "Year-by-Year tab.")
    c_b0, _ = st.columns([1, 2])
    with c_b0:
        st.pyplot(_fig_belief(b0_c, "b₀  (regime frequencies)"),
                  use_container_width=True)

    # ── Paper T comparison ────────────────────────────────────────────────────
    with st.expander("Compare with HESA paper T matrix", expanded=False):
        col_p, col_c = st.columns(2)
        with col_p:
            st.pyplot(
                _fig_matrix_heatmap(
                    params["T"], STATE_NAMES, STATE_NAMES,
                    "Paper T  (Russell Group panel)"),
                use_container_width=True)
        with col_c:
            st.pyplot(
                _fig_matrix_heatmap(
                    T_c, STATE_NAMES, STATE_NAMES,
                    "Your T  (uploaded data)"),
                use_container_width=True)
        diff = T_c - params["T"]
        max_diff = float(np.abs(diff).max())
        st.caption(
            f"Max absolute difference: **{max_diff:.3f}**  "
            "(0 = identical to paper; 1 = completely different row).")

    # ── Activate / deactivate ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Apply to all tabs")

    if st.session_state.calib6_active:
        st.success(
            "Custom calibration is **active** — Tabs 1–3 are using your "
            "T and Ω.  R_net remains the HESA paper reward matrix.")
        if st.button("Revert to HESA paper parameters", key="calib6_deactivate"):
            st.session_state.calib6_active = False
            st.rerun()
    else:
        st.info(
            "Press **Activate** to replace the paper T and Ω with the "
            "matrices calibrated above in all tabs.  "
            "R_net is unchanged (reward estimation requires financial data "
            "not available from enrolment counts alone).")
        if st.button("Activate custom calibration",
                     type="primary", key="calib6_activate"):
            st.session_state.calib6_T     = T_c
            st.session_state.calib6_Omega = Omega_c
            st.session_state.calib6_b0    = b0_c
            st.session_state.calib6_active = True
            _reset_session(sliders["b0"])   # re-init Tab 1 with current b0
            st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    sliders = _render_sidebar()
    params  = _get_paper_params()
    model   = _build_model(params, sliders)

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.title("Smarter Financial Planning for UK Universities")
    st.markdown(
        "An **entropy-driven POMDP decision framework** that translates annual "
        "HESA enrolment observations into financially justified strategic postures.  \n"
        "Calibrated on Russell Group panel data "
        "(Warwick · Exeter · Leeds · Bristol, 2014/15–2024/25).  \n"
        "Based on: Schetinin (2026) — "
        "[SSRN 6308238](https://doi.org/10.2139/ssrn.6308238)")

    # Custom calibration indicator
    if st.session_state.get("calib6_active", False):
        st.warning(
            "🔧 **Custom calibration active** — T and Ω from uploaded data "
            "(Tab 4).  Deactivate in Tab 4 to revert to HESA paper parameters.")

    # Key result metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Unified reward",    "349.2",
              help="Mean 12-yr discounted reward, Unified Framework (Table 4, n=500)")
    m2.metric("Crisis probability", "1.1 %",
              help="% planning years with R < −8 (financial distress threshold)")
    m3.metric("Planning horizon",  "12 yr",
              help="T_plan = 12 annual cycles (§5)")
    m4.metric("Strategies compared", "6",
              help="Unified · Robust Satisficing · Myopic · DAPP · RDM · Baseline")
    m5.metric("HESA panel",        "4 RG inst.",
              help="Warwick · Exeter · Leeds · Bristol, 2014/15–2024/25")

    # Getting-started guide
    with st.expander("Getting started", expanded=False):
        st.markdown(
            "**Choose a tab to explore the framework:**\n\n"
            "| Tab | What it does |\n"
            "|-----|--------------|\n"
            "| 📅 Year-by-Year | Step through a 12-year horizon one year at a time — "
            "record your institution's enrolment outcome and watch the belief update |\n"
            "| 📈 12-Year Simulation | Run all 6 strategies on a shared scenario in "
            "one click and compare reward trajectories and action maps |\n"
            "| ⚖️ Strategy Comparison | Reproduce Table 4 via Monte Carlo — confirm "
            "Unified Framework Pareto-dominance over all comparators |\n"
            "| 📂 Calibrate Your Data | Upload a CSV of your institution's enrolment "
            "history to replace the Russell Group parameters |\n\n"
            "**Sidebar sliders** adjust the framework hyperparameters in real time.  "
            "All defaults match the paper (§5, Table 3).  "
            "Hover any slider for the paper reference.")

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "\U0001f4c5  Year-by-Year",
        "\U0001f4c8  12-Year Simulation",
        "⚖️  Strategy Comparison",
        "\U0001f4c2  Calibrate Your Data",
    ])
    with tab1: _tab_year_by_year(model, sliders)
    with tab2: _tab_simulation(model, sliders)
    with tab3: _tab_comparison(model, sliders)
    with tab4: _tab_calibrate(model, sliders)


if __name__ == "__main__":
    main()

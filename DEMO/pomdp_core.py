"""
pomdp_core.py — Entropy-Driven POMDP Core Module  [DEMO Step D1]

Wraps the paper's calibration pipeline (rg_pomdp_calibration.py) in a clean
class API ready for the Streamlit app.  No new mathematics — all SR1–SR7
arithmetic delegates back to the original functions.

Default parameters reproduce Table 4 (paper §6.1):
  Unified Framework  ≈ 349.2 reward,  1.1 % distress
  Robust Satisficing ≈ 348.4 reward,  2.0 % distress
"""

import sys
import os
import io
import contextlib
import warnings
import numpy as np

# ── locate calibration module ─────────────────────────────────────────────────
_PAPER_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PAPER_DIR)
import rg_pomdp_calibration as _rg  # noqa: E402

# ── index constants (re-exported for convenience) ─────────────────────────────
N_STATES  = _rg.N_STATES   # 3
N_ACTIONS = _rg.N_ACTIONS  # 5
N_OBS     = _rg.N_OBS      # 4

LOW,      STABLE,   GROWTH      = 0, 1, 2
CONSERVE, MAINTAIN, RECRUIT, INVEST, RATIONALISE = 0, 1, 2, 3, 4
POOR,     WEAK,     MODERATE,   STRONG = 0, 1, 2, 3

STATE_NAMES  = _rg.STATE_NAMES    # ["Low", "Stable", "Growth"]
ACTION_NAMES = _rg.ACTION_NAMES   # ["Conserve", "Maintain", "Recruit", "Invest", "Rationalise"]
OBS_NAMES    = ["Poor", "Weak", "Moderate", "Strong"]
STRATEGY_LABELS = {
    "baseline":           "BDT Baseline",
    "myopic":             "Myopic POMDP",
    "robust_satisficing": "Robust Satisficing",
    "dapp":               "DAPP",
    "rdm":                "RDM-Conservative",
    "unified":            "Unified Framework",
}

# ── paper hyperparameter defaults ─────────────────────────────────────────────
PAPER_DEFAULTS = dict(
    gamma     = _rg.GAMMA,      # 0.95
    t_horizon = _rg.T_HORIZON,  # 12
    delta     = _rg.DELTA,      # 0.10
    lambda_v  = _rg.LAMBDA_V,   # 1.5
    mu_ov     = _rg.MU_OV,      # 1.2
    u_min     = _rg.U_MIN,      # -25.0
    n_sat     = _rg.N_SAT,      # 30 rollouts for SR3
    alpha_min = 0.0,
    alpha_max = 0.5,
)

# ── calibration cache ─────────────────────────────────────────────────────────
_CALIBRATED = None


def load_paper_params(verbose=False):
    """Run the HESA calibration pipeline and return POMDP parameters.

    Caches the result; subsequent calls are free.  Falls back to synthetic data
    automatically when HESA data files are absent (rg_pomdp_calibration
    handles that internally).

    Returns
    -------
    dict with keys: T, Omega, R_net, b0, T_true, low_t, high_t
    """
    global _CALIBRATED
    if _CALIBRATED is not None:
        return _CALIBRATED

    _orig_dir = os.getcwd()
    try:
        os.chdir(_PAPER_DIR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if verbose:
                _CALIBRATED = _run_calibration()
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    _CALIBRATED = _run_calibration()
    finally:
        os.chdir(_orig_dir)

    return _CALIBRATED


def _run_calibration():
    enrolments     = _rg.load_hesa_enrolments()
    pct_changes    = _rg.compute_pct_change(enrolments)
    low_t, high_t  = _rg.compute_empirical_thresholds(pct_changes)
    state_seqs     = _rg.assign_states_threshold(pct_changes, low_t, high_t)
    T              = _rg.compute_T_empirical(state_seqs)
    Omega          = _rg.build_omega_hesa(pct_changes, state_seqs, low_t, high_t)
    fin_data       = _rg.load_hesa_finance()
    R_net, _, _    = _rg.build_reward_matrix(fin_data, state_seqs)
    demo_prior     = _rg.load_demographic_prior()
    visa_adj       = _rg.load_visa_adjustment()
    b0_per_inst, _ = _rg.build_initial_priors(demo_prior, visa_adj)
    b0             = np.mean(np.stack(list(b0_per_inst.values())), axis=0)
    b0            /= b0.sum()
    # Perturbed T for MC model-uncertainty check (seed fixed for reproducibility)
    T_true = np.clip(T + np.random.default_rng(99).normal(0, 0.015, T.shape), 0.01, 1.0)
    T_true /= T_true.sum(axis=1, keepdims=True)
    return dict(T=T, Omega=Omega, R_net=R_net, b0=b0, T_true=T_true,
                low_t=low_t, high_t=high_t)


# ─────────────────────────────────────────────────────────────────────────────
#  EntropyPOMDP
# ─────────────────────────────────────────────────────────────────────────────

class EntropyPOMDP:
    """Entropy-Driven Bayesian Decision Tree POMDP for UK HE enrollment management.

    Parameters
    ----------
    T       : (3,3) transition matrix
    Omega   : (3,4) emission matrix
    R_net   : (5,3) net reward matrix (actions × states)
    b0      : (3,) initial belief
    T_true  : (3,3) ground-truth transition used in simulation (defaults to T)
    **kw    : override any key in PAPER_DEFAULTS
    """

    def __init__(self, T, Omega, R_net, b0, T_true=None, **kw):
        self.T      = np.asarray(T,     dtype=float)
        self.Omega  = np.asarray(Omega, dtype=float)
        self.R_net  = np.asarray(R_net, dtype=float)
        self.b0     = np.asarray(b0,    dtype=float)
        self.T_true = np.asarray(T_true, dtype=float) if T_true is not None else self.T.copy()

        p = {**PAPER_DEFAULTS, **kw}
        self.gamma     = float(p["gamma"])
        self.t_horizon = int(  p["t_horizon"])
        self.delta     = float(p["delta"])
        self.lambda_v  = float(p["lambda_v"])
        self.mu_ov     = float(p["mu_ov"])
        self.u_min     = float(p["u_min"])
        self.n_sat     = int(  p["n_sat"])
        self.alpha_min = float(p["alpha_min"])
        self.alpha_max = float(p["alpha_max"])

    @classmethod
    def from_paper(cls, verbose=False, **kw):
        """Construct from HESA-calibrated paper parameters (Table 3 / §5)."""
        p = load_paper_params(verbose=verbose)
        return cls(p["T"], p["Omega"], p["R_net"], p["b0"], p["T_true"], **kw)

    # ── Shannon entropy ───────────────────────────────────────────────────────

    def entropy(self, b):
        """H(b) = -Σ b(s) ln b(s).  Returns 0 at deterministic beliefs."""
        b = np.clip(b, 1e-12, 1.0)
        return float(-np.dot(b, np.log(b)))

    # ── SR1: Bayesian belief filter ───────────────────────────────────────────

    def belief_update(self, b, obs):
        """SR1: update belief given observation index (0=Poor … 3=Strong).

        Two-stage POMDP filter: predict via T, then correct via Omega[:,obs].
        """
        return _rg.sr1_belief_update(b, obs, self.T, self.Omega)

    # ── SR2: Entropy-driven belief blending ──────────────────────────────────

    def blend_belief(self, b, b_prior=None):
        """SR2: blend posterior b toward prior proportional to H(b) / ln N.

        α_t = α_min + (α_max − α_min) · H(b) / ln N
        At high entropy → lean on prior; at low entropy → trust posterior.
        """
        if b_prior is None:
            b_prior = self.b0
        alpha   = self.alpha_min + (self.alpha_max - self.alpha_min) * (
            self.entropy(b) / np.log(N_STATES))
        blended = (1.0 - alpha) * np.asarray(b) + alpha * np.asarray(b_prior)
        return blended / blended.sum()

    # ── SR3: Robust satisficing gate ─────────────────────────────────────────

    def satisficing_prob(self, b, horizon=None, n_rollouts=None, rng=None):
        """SR3: Monte Carlo probability that greedy rollout satisfies U_min.

        Returns p_sat ∈ [0,1].  Action is gated (→ Conserve) when p_sat < 1-δ.
        """
        if rng is None:
            rng = np.random.default_rng()
        h = min(horizon if horizon is not None else self.t_horizon, 6)
        n = n_rollouts if n_rollouts is not None else self.n_sat
        return _rg.sr3_satisficing_prob(
            b, self.T, self.Omega, self.R_net,
            self.u_min, self.gamma, h, n, rng)

    # ── SR4: Myopic Value of Information ─────────────────────────────────────

    def voi(self, b):
        """SR4: expected EVSI from one additional observation.

        Added to Q[Recruit] to reward information-gathering.
        """
        return _rg.sr4_myopic_voi(b, self.T, self.Omega, self.R_net)

    # ── SR5: Option value of delay ────────────────────────────────────────────

    def option_value(self, action, b):
        """SR5: cost of committing to irreversible action now vs. delaying.

        Subtracted from Q[Invest] and Q[Rationalise] when positive.
        """
        return _rg.sr5_option_value(b, action, self.T, self.Omega,
                                    self.R_net, self.gamma)

    # ── SR6: Centered reward normalisation ───────────────────────────────────

    def normalise_reward(self, r, mean_r=0.0, std_r=1.0):
        """SR6: centre and scale reward (used in display / comparison tables)."""
        return (r - mean_r) / max(float(std_r), 1e-8)

    # ── SR7: Dirichlet transition adaptation ─────────────────────────────────

    def adapt_T(self, T_current, b_history, lr=0.05):
        """SR7: soften T toward uniform when entropy increases for 3+ steps.

        Detects sustained H(b_t) increases as a regime-shift signal and
        pulls T toward the non-informative prior at learning rate lr.
        """
        if len(b_history) < 3:
            return T_current
        recent_H = [self.entropy(b) for b in b_history[-3:]]
        if all(h2 > h1 for h1, h2 in zip(recent_H, recent_H[1:])):
            T_new = (1.0 - lr) * T_current + lr * np.ones((N_STATES, N_STATES)) / N_STATES
            T_new /= T_new.sum(axis=1, keepdims=True)
            return T_new
        return T_current

    # ── Unified decision loop (Algorithm 1) ──────────────────────────────────

    def decide(self, b, t=0, rng=None,
               use_sr2=False, use_sr3=True, use_sr4=True, use_sr5=True):
        """Full entropy-driven decision loop for one annual planning cycle.

        Parameters
        ----------
        b        : current belief (3,)
        t        : current year index (0-based)
        use_sr*  : enable/disable individual components (for ablation)

        Returns
        -------
        action   : int   — recommended action index
        Q        : (5,)  — Q-values after all SR4/SR5 adjustments
        gated    : set   — actions eliminated by SR3 gate (empty if gate not triggered)
        H        : float — H(b) before decision
        sat_prob : float or None — SR3 satisficing probability
        """
        if rng is None:
            rng = np.random.default_rng()

        H   = self.entropy(b)
        b_w = self.blend_belief(b) if use_sr2 else b
        rem = max(1, self.t_horizon - t)

        sat_prob = None
        gated    = set()

        if use_sr3:
            sat_prob = self.satisficing_prob(b_w, rem, rng=rng)
            if sat_prob < (1.0 - self.delta):
                Q = self.R_net @ b_w
                gated = set(range(N_ACTIONS)) - {CONSERVE}
                return CONSERVE, Q, gated, H, sat_prob

        Q = (self.R_net @ b_w).copy()

        if use_sr4:
            Q[RECRUIT] += self.lambda_v * self.voi(b_w)

        if use_sr5:
            for a_irrev in (INVEST, RATIONALISE):
                ov = self.option_value(a_irrev, b_w)
                if ov > 0:
                    Q[a_irrev] -= self.mu_ov * ov

        return int(np.argmax(Q)), Q, gated, H, sat_prob

    # ── Comparator action selector ────────────────────────────────────────────

    def _action_comparator(self, strategy, b, obs_hist, t, rng):
        """Select action for a named comparator strategy (non-unified)."""
        if strategy == "baseline":
            return MAINTAIN

        if strategy == "myopic":
            return int(np.argmax(self.R_net @ b))

        if strategy == "robust_satisficing":
            rem = max(1, self.t_horizon - t)
            sp  = self.satisficing_prob(b, rem, rng=rng)
            return CONSERVE if sp < (1.0 - self.delta) else int(np.argmax(self.R_net @ b))

        if strategy == "dapp":
            if len(obs_hist) >= 3 and all(o <= POOR for o in obs_hist[-3:]):
                return RATIONALISE
            if len(obs_hist) >= 2 and all(o <= WEAK for o in obs_hist[-2:]):
                return CONSERVE
            return MAINTAIN

        if strategy == "rdm":
            if b[LOW]    > 0.55: return RATIONALISE
            if b[LOW]    > 0.35: return CONSERVE
            if b[GROWTH] > 0.55: return RECRUIT
            return MAINTAIN

        raise ValueError(f"Unknown strategy: {strategy!r}")

    # ── Episode simulator ─────────────────────────────────────────────────────

    def simulate_episode(self, b0=None, obs_sequence=None, rng=None,
                         strategy="unified",
                         use_sr2=False, use_sr3=True, use_sr4=True, use_sr5=True):
        """Run one T_horizon-year planning episode.

        Parameters
        ----------
        b0           : initial belief — defaults to self.b0
        obs_sequence : fixed list of observation indices; if None, sampled stochastically
        rng          : numpy Generator
        strategy     : 'unified' | 'baseline' | 'myopic' | 'robust_satisficing'
                       | 'dapp' | 'rdm'
        use_sr*      : active only when strategy='unified'

        Returns
        -------
        dict
            total_reward   : float
            distress_count : int   (years where R < PER_STEP_CRISIS)
            actions        : list[int]  length T
            beliefs        : list[ndarray] length T+1 (includes post-final update)
            observations   : list[int]  length T
            Q_values       : list[ndarray] length T  (full Q vector per step)
            entropies      : list[float] length T+1
            states         : list[int]  length T  (true hidden state per step)
            sat_probs      : list[float|None] length T
        """
        if rng is None:
            rng = np.random.default_rng()
        b = np.array(b0 if b0 is not None else self.b0, dtype=float)
        s = rng.choice(N_STATES, p=b)

        total_r, distress = 0.0, 0
        actions, beliefs, obs_list = [], [], []
        Q_vals, entropies, states, sat_probs = [], [], [], []
        obs_hist = []

        for t in range(self.t_horizon):
            beliefs.append(b.copy())
            entropies.append(self.entropy(b))
            states.append(s)

            if strategy == "unified":
                a, Q, _, H, sp = self.decide(b, t=t, rng=rng,
                                              use_sr2=use_sr2, use_sr3=use_sr3,
                                              use_sr4=use_sr4, use_sr5=use_sr5)
                Q_vals.append(Q.copy())
                sat_probs.append(sp)
            else:
                a = self._action_comparator(strategy, b, obs_hist, t, rng)
                Q_vals.append((self.R_net @ b).copy())
                sat_probs.append(None)

            actions.append(a)
            r = self.R_net[a, s]
            total_r += (self.gamma ** t) * r
            if r < _rg.PER_STEP_CRISIS:
                distress += 1

            # True state transition
            s = rng.choice(N_STATES, p=self.T_true[s])

            # Observation: fixed sequence or stochastic
            if obs_sequence is not None and t < len(obs_sequence):
                o = int(obs_sequence[t])
            else:
                o = rng.choice(N_OBS, p=self.Omega[s])
            obs_hist.append(o)
            obs_list.append(o)

            b = self.belief_update(b, o)

        beliefs.append(b.copy())
        entropies.append(self.entropy(b))

        return dict(
            total_reward   = total_r,
            distress_count = distress,
            actions        = actions,
            beliefs        = beliefs,
            observations   = obs_list,
            Q_values       = Q_vals,
            entropies      = entropies,
            states         = states,
            sat_probs      = sat_probs,
        )

    # ── Monte Carlo comparison ────────────────────────────────────────────────

    def run_mc(self, n_paths=500, seed=42, progress_fn=None):
        """Compare all 6 strategies via Monte Carlo (reproduces Table 4).

        Parameters
        ----------
        n_paths     : paths per strategy
        seed        : master RNG seed
        progress_fn : optional callback(strategy_label, done_count, total)

        Returns
        -------
        dict {label: {mean_reward, distress_pct, ci_lo, ci_hi, rewards_raw}}
        """
        strategy_order = [
            ("BDT Baseline",       "baseline"),
            ("Myopic POMDP",       "myopic"),
            ("Robust Satisficing", "robust_satisficing"),
            ("DAPP",               "dapp"),
            ("RDM-Conservative",   "rdm"),
            ("Unified Framework",  "unified"),
        ]
        master  = np.random.default_rng(seed)
        results = {}

        for label, strategy in strategy_order:
            strat_rng = np.random.default_rng(master.integers(10 ** 9))
            rewards, distress_counts = [], []

            for i in range(n_paths):
                ep_rng = np.random.default_rng(strat_rng.integers(10 ** 9))
                ep     = self.simulate_episode(rng=ep_rng, strategy=strategy)
                rewards.append(ep["total_reward"])
                distress_counts.append(ep["distress_count"])
                if progress_fn:
                    progress_fn(label, i + 1, n_paths)

            r_arr = np.array(rewards)
            results[label] = dict(
                mean_reward  = float(np.mean(r_arr)),
                distress_pct = float(np.mean(distress_counts) / self.t_horizon * 100),
                ci_lo        = float(np.percentile(r_arr, 2.5)),
                ci_hi        = float(np.percentile(r_arr, 97.5)),
                rewards_raw  = r_arr,
            )

        return results


# ─────────────────────────────────────────────────────────────────────────────
#  Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_paper_result(n_paths=500, seed=42, tol_reward=10.0, tol_distress=1.5):
    """Reproduce Table 4 and check Unified Framework against paper targets.

    Targets: reward ≈ 349.2 (±tol_reward),  distress ≈ 1.1 % (±tol_distress).

    Returns
    -------
    results : dict   — full MC results table
    passed  : bool   — True when both metrics within tolerance
    """
    print("Loading calibrated parameters …")
    model   = EntropyPOMDP.from_paper()
    print(f"Running MC comparison  (n={n_paths}, seed={seed}) …")
    results = model.run_mc(n_paths=n_paths, seed=seed)

    uf      = results["Unified Framework"]
    ok_r    = abs(uf["mean_reward"]  - 349.2) < tol_reward
    ok_d    = abs(uf["distress_pct"] - 1.1)   < tol_distress
    passed  = ok_r and ok_d

    print(f"\n{'Strategy':<22}  {'Reward':>8}  {'Distress%':>10}  {'95% CI':>22}")
    print("-" * 70)
    for name, r in results.items():
        marker = " ← target" if name == "Unified Framework" else ""
        print(f"{name:<22}  {r['mean_reward']:>8.1f}  {r['distress_pct']:>9.1f}%"
              f"  [{r['ci_lo']:>7.1f}, {r['ci_hi']:>7.1f}]{marker}")

    status = "PASS" if passed else "WARN (check calibration)"
    print(f"\n[{status}]  Unified: reward={uf['mean_reward']:.1f} (target ~349.2)"
          f"  distress={uf['distress_pct']:.1f}% (target ~1.1%)")
    print(f"          Robust Satisficing: reward={results['Robust Satisficing']['mean_reward']:.1f}"
          f"  distress={results['Robust Satisficing']['distress_pct']:.1f}%  (target ~348.4 / ~2.0%)")
    return results, passed


if __name__ == "__main__":
    validate_paper_result()

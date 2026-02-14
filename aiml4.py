#!/usr/bin/env python3
"""
rl_gm_sim_agent.py

Agent-based GM mosquito simulator + reinforcement learning (tabular Q-learning).
Run with your Python (no RL libraries required).

Save, run, and experiment.

Author: ChatGPT (adapted for user's project)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Text
from tkinter import simpledialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib
import os
import json
from datetime import datetime

# ---------------------------
# Configuration / Hyperparams
# ---------------------------
ACTIONS = [0, 10, 20, 30, 40, 50, 60, 70, 80]  # GM% options
N_ACTIONS = len(ACTIONS)

# Discretization bins for state
MOSQ_BINS = [0, 50, 100, 200, 400, 800, 1600, 4000]  # mosquito_count buckets
PAST_CASES_BINS = [0, 5, 15, 30, 60, 120]
MONTH_BINS = list(range(1, 13))  # use month as separate state dimension

# RL hyperparams
ALPHA = 0.1       # learning rate
GAMMA = 0.95      # discount
EPS_START = 0.9   # starting epsilon for epsilon-greedy
EPS_END = 0.05
EPS_DECAY = 0.995

# Environment params
DAYS_PER_EPISODE = 180
INITIAL_MOSQ = 800.0
ECOLOGICAL_THRESHOLD_DEFAULT = 100.0

# Reward weights
WEIGHT_DISEASE = -1.0    # negative disease reduces reward
WEIGHT_GM_COST = -0.02   # cost per percent GM release (small)
PENALTY_ECO = -200.0     # heavy penalty if min mosquito falls below threshold

# Random seed
RNG = np.random.RandomState(42)

# ---------------------------
# Simple agent-based environment
# ---------------------------
class MosquitoEnv:
    """
    Simple day-by-day environment.
    State includes mosquito_count, month, past_cases.
    Dynamics:
      mosquito_next = mosquito_current * (1 + growth_rate - natural_mortality) - removed_by_GM + rainfall_effect + noise
    Disease model:
      disease_cases = baseline + alpha * mosquito_count + beta * past_cases + noise
    """
    def __init__(self, n_days=DAYS_PER_EPISODE, init_mosq=INITIAL_MOSQ, ecological_threshold=ECOLOGICAL_THRESHOLD_DEFAULT):
        self.n_days = n_days
        self.init_mosq = float(init_mosq)
        self.ecological_threshold = float(ecological_threshold)
        self.reset()

    def reset(self, start_day_index=0):
        self.day = 0
        self.index = start_day_index
        self.mosquito_count = float(self.init_mosq)
        self.past_cases = float(RNG.poisson(20))  # initial past cases sample
        self.history = {'day': [], 'mosquito': [], 'gm_pct': [], 'disease': [], 'past_cases': []}
        self.min_mosquito_seen = self.mosquito_count
        # seasonal seed
        return self._get_obs()

    def _seasonality(self, day_index):
        # simple seasonal factor: sin wave over 365 days
        t = (day_index % 365) / 365.0
        # peak around day ~150
        seasonal = 0.2 * np.sin(2 * np.pi * (t - 0.2))  # amplitude 0.2
        rainfall = 50 + 30 * max(0, np.sin(2 * np.pi * (t - 0.25)))
        return seasonal, rainfall

    def _get_obs(self):
        # return raw observations: mosquito_count, month (1-12), past_cases
        month = ((self.index % 365) // 30) + 1
        return {'mosquito': self.mosquito_count, 'month': int(month), 'past_cases': self.past_cases}

    def step(self, action_pct):
        """
        action_pct: integer e.g., 0..80 (must be in ACTIONS)
        returns: obs, reward, done, info
        """
        # apply GM removal
        removed = self.mosquito_count * (action_pct / 100.0)
        # natural reproduction / seasonal effect
        seasonal, rainfall = self._seasonality(self.index + self.day)
        # growth rate baseline
        growth = 0.02 + seasonal  # small baseline growth + season
        # rainfall effect: increases breeding - proportional factor
        rainfall_effect = 0.001 * (rainfall - 50)  # small
        noise = RNG.normal(0, 10.0)

        # update mosquito population
        next_mosq = self.mosquito_count * (1 + growth + rainfall_effect) - removed + noise
        next_mosq = max(1.0, next_mosq)  # keep >= 1
        self.mosquito_count = next_mosq
        self.min_mosquito_seen = min(self.min_mosquito_seen, self.mosquito_count)

        # disease model (simple linear)
        baseline = 5.0
        alpha = 0.03
        beta = 0.4
        disease_mean = baseline + alpha * self.mosquito_count + beta * self.past_cases
        disease_noise = max(0, RNG.normal(0, 3.0))
        disease_cases = max(0.0, disease_mean + disease_noise)

        # update past_cases for next day (simple moving)
        self.past_cases = 0.7 * self.past_cases + 0.3 * disease_cases

        # record history
        self.history['day'].append(self.index + self.day)
        self.history['mosquito'].append(self.mosquito_count)
        self.history['gm_pct'].append(action_pct)
        self.history['disease'].append(disease_cases)
        self.history['past_cases'].append(self.past_cases)

        # compute reward
        # negative disease (we want to minimize disease), small cost for GM action, penalty if ecological threshold violated
        reward = WEIGHT_DISEASE * disease_cases + WEIGHT_GM_COST * action_pct
        if self.min_mosquito_seen < self.ecological_threshold:
            reward += PENALTY_ECO  # heavy penalty

        done = (self.day >= self.n_days - 1)
        self.day += 1
        obs = self._get_obs()
        info = {'min_mosquito_seen': self.min_mosquito_seen}
        return obs, reward, done, info

    def get_history_df(self):
        return pd.DataFrame(self.history)

# ---------------------------
# Discretization utilities
# ---------------------------
def discretize_mosquito(mosq_value):
    return int(np.digitize([mosq_value], MOSQ_BINS)[0])  # bin index

def discretize_past_cases(pc_value):
    return int(np.digitize([pc_value], PAST_CASES_BINS)[0])

def make_state_tuple(obs):
    """
    Input obs: dict with 'mosquito', 'month', 'past_cases'
    Output: tuple of discrete indices
    """
    m_bin = discretize_mosquito(obs['mosquito'])
    p_bin = discretize_past_cases(obs['past_cases'])
    month = int(obs['month'])  # 1..12
    return (m_bin, month, p_bin)

# ---------------------------
# Q-Learning Agent (tabular)
# ---------------------------
class QAgent:
    def __init__(self, actions=ACTIONS):
        self.actions = actions
        # Q-table: nested dict keyed by state tuples -> np.array of action values
        self.Q = {}
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = EPS_START

    def get_q(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(len(self.actions), dtype=float)
        return self.Q[state]

    def choose_action(self, state, greedy=False):
        q = self.get_q(state)
        if not greedy and RNG.rand() < self.epsilon:
            return RNG.randint(len(self.actions))
        else:
            # tie-breaking random choice among maxima
            maxv = q.max()
            choices = np.where(np.isclose(q, maxv))[0]
            return int(RNG.choice(choices))

    def learn(self, state, action_index, reward, next_state, done):
        q = self.get_q(state)
        q_next = self.get_q(next_state)
        target = reward + (0.0 if done else self.gamma * q_next.max())
        q[action_index] += self.alpha * (target - q[action_index])

    def decay_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({'alpha': self.alpha, 'gamma': self.gamma, 'epsilon': self.epsilon,
                       'Q': {str(k): list(v.tolist()) for k, v in self.Q.items()}}, f)

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        self.alpha = data.get('alpha', self.alpha)
        self.gamma = data.get('gamma', self.gamma)
        self.epsilon = data.get('epsilon', self.epsilon)
        qdict = data.get('Q', {})
        self.Q = {eval(k): np.array(v, dtype=float) for k, v in qdict.items()}

# ---------------------------
# GUI: Tkinter app
# ---------------------------
class AppGUI:
    def __init__(self, root):
        self.root = root
        root.title("RL GM Mosquito Agent Simulator")
        root.geometry("1100x720")
        self.env = MosquitoEnv()
        self.agent = QAgent()
        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self.root); top.pack(fill='x', padx=8, pady=8)
        ttk.Button(top, text="Generate Demo Env", command=self.on_generate_demo).pack(side='left', padx=6)
        ttk.Button(top, text="Reset Env", command=self.on_reset_env).pack(side='left', padx=6)
        ttk.Button(top, text="Run Manual Simulation", command=self.on_run_manual).pack(side='left', padx=6)
        ttk.Button(top, text="Train Agent (Q-learning)", command=self.on_train_agent).pack(side='left', padx=6)
        ttk.Button(top, text="Run Agent Policy", command=self.on_run_policy).pack(side='left', padx=6)
        ttk.Button(top, text="Export last sim CSV", command=self.on_export_sim).pack(side='left', padx=6)
        ttk.Button(top, text="Save Agent", command=self.on_save_agent).pack(side='left', padx=6)
        ttk.Button(top, text="Load Agent", command=self.on_load_agent).pack(side='left', padx=6)

        # Parameters frame
        params = ttk.Frame(self.root); params.pack(fill='x', padx=8, pady=4)
        ttk.Label(params, text="Episodes:").pack(side='left')
        self.episodes_var = tk.IntVar(value=200)
        ttk.Entry(params, textvariable=self.episodes_var, width=6).pack(side='left', padx=4)

        ttk.Label(params, text="Days per episode:").pack(side='left', padx=8)
        self.days_var = tk.IntVar(value=DAYS_PER_EPISODE)
        ttk.Entry(params, textvariable=self.days_var, width=6).pack(side='left', padx=4)

        ttk.Label(params, text="Ecological threshold:").pack(side='left', padx=8)
        self.threshold_var = tk.DoubleVar(value=ECOLOGICAL_THRESHOLD_DEFAULT)
        ttk.Entry(params, textvariable=self.threshold_var, width=8).pack(side='left', padx=4)

        ttk.Label(params, text="Learning rate (alpha):").pack(side='left', padx=8)
        self.alpha_var = tk.DoubleVar(value=ALPHA)
        ttk.Entry(params, textvariable=self.alpha_var, width=6).pack(side='left', padx=4)

        # Canvas frames for plots and info
        main = ttk.Frame(self.root); main.pack(fill='both', expand=True, padx=8, pady=8)
        left = ttk.Frame(main); left.pack(side='left', fill='both', expand=True)
        right = ttk.Frame(main, width=300); right.pack(side='right', fill='y')

        # Left: matplotlib plot
        self.fig, self.axs = plt.subplots(2, 1, figsize=(8,6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Right: text logs and controls
        ttk.Label(right, text="Logs / Info").pack(anchor='w')
        self.log_text = Text(right, height=20, width=40)
        self.log_text.pack(fill='y', padx=6, pady=4)
        ttk.Label(right, text="Policy (state -> action index)").pack(anchor='w')
        self.policy_text = Text(right, height=12, width=40)
        self.policy_text.pack(fill='y', padx=6, pady=4)

        # status bar
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status_var).pack(fill='x', padx=8, pady=4)

        # storage for last sim
        self.last_sim_df = None

    def log(self, *args):
        s = " ".join(str(a) for a in args) + "\n"
        self.log_text.insert(tk.END, s)
        self.log_text.see(tk.END)

    def on_generate_demo(self):
        # re-create environment with default params
        days = int(self.days_var.get())
        threshold = float(self.threshold_var.get())
        self.env = MosquitoEnv(n_days=days, init_mosq=INITIAL_MOSQ, ecological_threshold=threshold)
        self.log("Generated demo environment:", days, "days, threshold:", threshold)
        self.status_var.set("Demo environment ready")

    def on_reset_env(self):
        self.env.reset()
        self.log("Environment reset")
        self.status_var.set("Environment reset")

    def _plot_history(self, df, title_suffix=""):
        self.axs[0].clear(); self.axs[1].clear()
        self.axs[0].plot(df['day'], df['mosquito'], label='mosquito_count')
        self.axs[0].set_ylabel('Mosquito count')
        self.axs[0].legend()
        self.axs[1].plot(df['day'], df['disease'], label='disease_cases')
        # show actions as scatter
        self.axs[1].scatter(df['day'], df['gm_pct'], label='gm_pct (action)', marker='x')
        self.axs[1].set_ylabel('Disease / GM%')
        self.axs[1].legend()
        self.fig.suptitle(f"Simulation {title_suffix}")
        self.canvas.draw()

    def on_run_manual(self):
        # run a simulation with a manual GM% schedule (use current slider or constant)
        # ask user for GM% to apply each day (simple constant via dialog)
        try:
            gm = simpledialog.askinteger("Manual GM%", "Enter constant GM% to apply each day (0-80):", minvalue=0, maxvalue=80, initialvalue=50)
            if gm is None:
                return
            days = int(self.days_var.get())
            self.env = MosquitoEnv(n_days=days, init_mosq=INITIAL_MOSQ, ecological_threshold=float(self.threshold_var.get()))
            obs = self.env.reset()
            done = False
            while not done:
                obs, reward, done, info = self.env.step(gm)
            df = self.env.get_history_df()
            self.last_sim_df = df
            self._plot_history(df, title_suffix=f"(manual gm={gm}%)")
            self.log("Manual sim done. Mean disease:", df['disease'].mean(), "Min mosq:", df['mosquito'].min())
            self.status_var.set("Manual sim completed")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_train_agent(self):
        episodes = int(self.episodes_var.get())
        days = int(self.days_var.get())
        threshold = float(self.threshold_var.get())
        self.agent.alpha = float(self.alpha_var.get())
        self.agent.epsilon = EPS_START
        self.log(f"Training agent: episodes={episodes}, days={days}, threshold={threshold}, alpha={self.agent.alpha}")
        best_reward = -1e9
        reward_history = []
        for ep in range(1, episodes + 1):
            env = MosquitoEnv(n_days=days, init_mosq=INITIAL_MOSQ, ecological_threshold=threshold)
            obs = env.reset()
            state = make_state_tuple(obs)
            total_reward = 0.0
            done = False
            step_count = 0
            while not done:
                a_index = self.agent.choose_action(state)
                action_pct = ACTIONS[a_index]
                next_obs, reward, done, info = env.step(action_pct)
                next_state = make_state_tuple(next_obs)
                self.agent.learn(state, a_index, reward, next_state, done)
                state = next_state
                total_reward += reward
                step_count += 1
            self.agent.decay_epsilon()
            reward_history.append(total_reward)
            if total_reward > best_reward:
                best_reward = total_reward
            # occasional logging
            if ep % max(1, episodes // 10) == 0 or ep <= 5:
                self.log(f"Episode {ep}/{episodes}  total_reward={total_reward:.1f}  best={best_reward:.1f}  eps={self.agent.epsilon:.3f}")
        # after training show some policy info
        self.log("Training completed. Best reward:", best_reward)
        self.status_var.set("Training completed")
        self._display_policy()
        # run one simulation with greedy policy to visualize
        self._run_policy_sim(days=days, threshold=threshold, greedy=True)

    def _run_policy_sim(self, days=None, threshold=None, greedy=True):
        days = int(self.days_var.get()) if days is None else int(days)
        threshold = float(self.threshold_var.get()) if threshold is None else float(threshold)
        env = MosquitoEnv(n_days=days, init_mosq=INITIAL_MOSQ, ecological_threshold=threshold)
        obs = env.reset()
        state = make_state_tuple(obs)
        done = False
        while not done:
            a_index = self.agent.choose_action(state, greedy=greedy)
            action_pct = ACTIONS[a_index]
            next_obs, reward, done, info = env.step(action_pct)
            state = make_state_tuple(next_obs)
        df = env.get_history_df()
        self.last_sim_df = df
        self._plot_history(df, title_suffix="(policy run)")
        self.log("Policy run done. Mean disease:", df['disease'].mean(), "Min mosq:", df['mosquito'].min())

    def on_run_policy(self):
        days = int(self.days_var.get())
        threshold = float(self.threshold_var.get())
        self._run_policy_sim(days=days, threshold=threshold, greedy=True)
        self.status_var.set("Policy run completed")

    def on_export_sim(self):
        if self.last_sim_df is None:
            messagebox.showwarning("No sim", "Run a simulation first")
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv")
        if not p:
            return
        try:
            self.last_sim_df.to_csv(p, index=False)
            messagebox.showinfo("Saved", f"Saved simulation to {p}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def on_save_agent(self):
        p = filedialog.asksaveasfilename(defaultextension=".json")
        if not p:
            return
        try:
            self.agent.save(p)
            messagebox.showinfo("Saved", f"Agent Q-table saved to {p}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def on_load_agent(self):
        p = filedialog.askopenfilename(filetypes=[("JSON","*.json"),("All files","*.*")])
        if not p:
            return
        try:
            self.agent.load(p)
            messagebox.showinfo("Loaded", f"Agent loaded from {p}")
            self._display_policy()
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def _display_policy(self):
        # show top action for some representative states
        self.policy_text.delete(1.0, tk.END)
        # iterate over known Q states
        items = list(self.agent.Q.items())[:200]
        for st, qvals in items:
            best_a = int(np.argmax(qvals))
            action_pct = ACTIONS[best_a]
            self.policy_text.insert(tk.END, f"{st} -> {action_pct}% (a{best_a})\n")
        self.policy_text.see(tk.END)

# ---------------------------
# Helpers to make state tuples (outside classes)
# ---------------------------
def make_state_tuple(obs):
    mosq = obs['mosquito']
    month = obs['month']
    past = obs['past_cases']
    return (discretize_mosquito(mosq), int(month), discretize_past_cases(past))

# ---------------------------
# Run the UI
# ---------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = AppGUI(root)
    root.mainloop()
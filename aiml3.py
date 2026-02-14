#!/usr/bin/env python3
"""
aiml_improved.py
Improved GM Mosquito AI GUI (threaded training, consistent encoding, automatic optimal-GM finder).
Save and run with your venv Python.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Text, simpledialog
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import threading
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta
import joblib
import os

# ----------------------------
# App state dataclass
# ----------------------------
@dataclass
class AppState:
    df: Optional[pd.DataFrame] = None
    features: list = field(default_factory=list)
    target: Optional[str] = None
    model: Optional[Any] = None
    X_train: Optional[pd.DataFrame] = None
    X_val: Optional[pd.DataFrame] = None
    y_train: Optional[pd.Series] = None
    y_val: Optional[pd.Series] = None
    predictions: Optional[np.ndarray] = None
    gm_reduction_pct: float = 0.0
    ecological_threshold: float = 100.0
    last_figure: Optional[plt.Figure] = None
    mae: float = 0.0
    r2: float = 0.0
    cat_mappings: Dict[str, Any] = field(default_factory=dict)
    model_path: Optional[str] = None
    last_sim_df: Optional[pd.DataFrame] = None

# ----------------------------
# Utility functions
# ----------------------------
def auto_choose_features_and_target(df: pd.DataFrame):
    numcols = df.select_dtypes(include=[np.number]).columns.tolist()
    target = None
    for name in ['disease_cases', 'cases', 'disease', 'disease_case', 'disease_count']:
        if name in df.columns:
            target = name
            break
    if target is None:
        for c in numcols:
            if 'case' in c.lower():
                target = c
                break
    if target is None and numcols:
        target = numcols[-1]
    if target:
        features = [c for c in numcols if c != target]
    else:
        features = numcols
    if 'mosquito_count' in df.columns and 'mosquito_count' not in features:
        features.insert(0, 'mosquito_count')
    return features, target

def suggest_threshold_from_data(df: pd.DataFrame, percentile=5):
    if df is None or 'mosquito_count' not in df.columns:
        return None
    try:
        arr = pd.to_numeric(df['mosquito_count'], errors='coerce').dropna().values
        if len(arr) == 0:
            return None
        return float(np.percentile(arr, percentile))
    except Exception:
        return None

def simple_recommendation(base_mean, gm_mean, min_mosquito, threshold):
    if min_mosquito < threshold:
        return "High risk — mosquito count drops below safe threshold. Do NOT proceed."
    if gm_mean < base_mean:
        return "Good: GM reduces predicted disease cases. Proceed with caution."
    return "No improvement: GM does not reduce predicted disease cases."

# ----------------------------
# Main GUI class
# ----------------------------
class GMMosquitoAIGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GM Mosquito AI Impact Study (Improved)")
        self.root.geometry("1000x700")
        self.state = AppState()

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        self.create_data_tab()
        self.create_train_tab()
        self.create_simulate_tab()
        self.create_report_tab()

    # ----------------------------
    # DATA Tab
    # ----------------------------
    def create_data_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text='DATA')

        btn_frame = ttk.Frame(tab)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="Load CSV...", command=self.load_csv).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Generate Demo Data", command=self.generate_demo_data).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Load Model...", command=self.load_model).pack(side='left', padx=5)

        select_frame = ttk.Frame(tab)
        select_frame.pack(pady=10, fill='x', padx=20)

        ttk.Label(select_frame, text="Select Target Column:").grid(row=0, column=0, sticky='w', pady=5)
        self.target_combo = ttk.Combobox(select_frame, state='readonly', width=30)
        self.target_combo.grid(row=0, column=1, sticky='w', pady=5, padx=10)

        ttk.Label(select_frame, text="Select Features (multi-select):").grid(row=1, column=0, sticky='nw', pady=5)
        features_frame = ttk.Frame(select_frame)
        features_frame.grid(row=1, column=1, sticky='w', pady=5, padx=10)

        self.features_listbox = tk.Listbox(features_frame, selectmode='multiple', height=6, width=40)
        self.features_listbox.pack(side='left')
        scrollbar = ttk.Scrollbar(features_frame, orient='vertical', command=self.features_listbox.yview)
        scrollbar.pack(side='left', fill='y')
        self.features_listbox.config(yscrollcommand=scrollbar.set)

        ttk.Label(select_frame, text="Test Split (validation):").grid(row=2, column=0, sticky='w', pady=5)
        slider_frame = ttk.Frame(select_frame)
        slider_frame.grid(row=2, column=1, sticky='w', pady=5, padx=10)

        self.test_split_var = tk.DoubleVar(value=0.2)
        self.test_split_slider = ttk.Scale(slider_frame, from_=0.05, to=0.4, orient='horizontal',
                                           variable=self.test_split_var, length=200)
        self.test_split_slider.pack(side='left')
        self.test_split_label = ttk.Label(slider_frame, text="0.20")
        self.test_split_label.pack(side='left', padx=10)
        self.test_split_var.trace_add('write', lambda *args: self.update_split_label())

        ttk.Button(select_frame, text="Apply selection", command=self.apply_selection).grid(row=3, column=1, sticky='w', pady=10, padx=10)

        ttk.Label(tab, text="Data Preview (top 200 rows):").pack(pady=5)
        preview_frame = ttk.Frame(tab)
        preview_frame.pack(fill='both', expand=True, padx=20, pady=5)

        self.preview_text = Text(preview_frame, height=15, width=100)
        self.preview_text.pack(side='left', fill='both', expand=True)
        preview_scroll = ttk.Scrollbar(preview_frame, orient='vertical', command=self.preview_text.yview)
        preview_scroll.pack(side='left', fill='y')
        self.preview_text.config(yscrollcommand=preview_scroll.set)

    def update_split_label(self, *args):
        self.test_split_label.config(text=f"{self.test_split_var.get():.2f}")

    def load_csv(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filename:
            try:
                self.state.df = pd.read_csv(filename)
                self._after_data_load()
                messagebox.showinfo("Success", "CSV loaded successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV: {str(e)}")

    def generate_demo_data(self):
        np.random.seed(42)
        n = 365
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n)]
        day_of_year = np.arange(n)
        seasonal_temp = 20 + 10 * np.sin(2 * np.pi * day_of_year / 365)
        seasonal_rainfall = 50 + 30 * np.sin(2 * np.pi * (day_of_year - 90) / 365)

        avg_temp = seasonal_temp + np.random.normal(0, 3, n)
        rainfall = np.maximum(0, seasonal_rainfall + np.random.normal(0, 15, n))
        humidity = 60 + 15 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 5, n)

        mosquito_count = (500 + 200 * np.sin(2 * np.pi * (day_of_year - 60) / 365) +
                         5 * rainfall + 3 * avg_temp + np.random.normal(0, 50, n))
        mosquito_count = np.maximum(10, mosquito_count)

        past_cases = np.random.poisson(20, n)

        disease_cases = (10 + 0.05 * mosquito_count + 0.3 * past_cases +
                        0.2 * humidity - 0.1 * avg_temp + np.random.normal(0, 5, n))
        disease_cases = np.maximum(0, disease_cases).astype(int)

        self.state.df = pd.DataFrame({
            'date': dates,
            'region': np.random.choice(['North', 'South', 'East', 'West'], n),
            'mosquito_count': mosquito_count,
            'avg_temp': avg_temp,
            'rainfall': rainfall,
            'humidity': humidity,
            'past_cases': past_cases,
            'disease_cases': disease_cases
        })

        self._after_data_load()
        messagebox.showinfo("Success", "Demo data generated successfully")

    def _after_data_load(self):
        if self.state.df is None:
            return
        self.preview_text.delete(1.0, tk.END)
        try:
            self.preview_text.insert(1.0, self.state.df.head(200).to_string())
        except Exception:
            self.preview_text.insert(1.0, repr(self.state.df.head(200)))

        columns = list(self.state.df.columns)
        self.target_combo['values'] = columns

        self.features_listbox.delete(0, tk.END)
        for col in columns:
            self.features_listbox.insert(tk.END, col)

        # auto-select features/target and show a gentle message
        features_auto, target_auto = auto_choose_features_and_target(self.state.df)
        if target_auto:
            self.target_combo.set(target_auto)
        # pre-select features in listbox
        for i, col in enumerate(columns):
            if col in features_auto:
                self.features_listbox.selection_set(i)

    def apply_selection(self):
        if self.state.df is None:
            messagebox.showwarning("Warning", "No data loaded")
            return

        target = self.target_combo.get()
        if not target:
            messagebox.showwarning("Warning", "Please select a target column")
            return

        selected_indices = self.features_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "Please select at least one feature")
            return

        features = [self.features_listbox.get(i) for i in selected_indices]

        if target in features:
            messagebox.showwarning("Warning", "Target cannot be in features")
            return

        self.state.target = target
        self.state.features = features

        # suggest threshold based on data
        suggested = suggest_threshold_from_data(self.state.df)
        if suggested is not None:
            self.state.ecological_threshold = suggested

        messagebox.showinfo("Success", f"Target: {target}\nFeatures: {', '.join(features)}\nSuggested ecological threshold: {self.state.ecological_threshold:.2f}")

    # ----------------------------
    # Encoding helper (consistent)
    # ----------------------------
    def _encode_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        X_enc = X.copy()
        cat_cols = X_enc.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in cat_cols:
            if fit:
                cat = pd.Categorical(X_enc[col])
                categories = list(cat.categories)
                mapping = {cat_val: code for code, cat_val in enumerate(categories)}
                self.state.cat_mappings[col] = {'mapping': mapping, 'categories': categories}
                X_enc[col] = X_enc[col].map(mapping).fillna(-1).astype(int)
            else:
                info = self.state.cat_mappings.get(col)
                if info is not None:
                    mapping = info['mapping']
                    X_enc[col] = X_enc[col].map(mapping).fillna(-1).astype(int)
                else:
                    X_enc[col] = pd.Categorical(X_enc[col]).codes
        return X_enc

    # ----------------------------
    # TRAIN Tab (threaded)
    # ----------------------------
    def create_train_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text='TRAIN')

        ttk.Button(tab, text="Train model", command=self.train_model).pack(pady=20)

        metrics_frame = ttk.LabelFrame(tab, text="Validation Metrics")
        metrics_frame.pack(pady=10, padx=20, fill='x')

        self.mae_label = ttk.Label(metrics_frame, text="MAE: -", font=('Arial', 12))
        self.mae_label.pack(pady=5)
        self.r2_label = ttk.Label(metrics_frame, text="R²: -", font=('Arial', 12))
        self.r2_label.pack(pady=5)

        self.train_plot_frame = ttk.Frame(tab)
        self.train_plot_frame.pack(fill='both', expand=True, padx=20, pady=10)

    def train_model(self):
        if self.state.df is None or not self.state.features or not self.state.target:
            messagebox.showwarning("Warning", "Please load data and apply selection first")
            return
        # run in background thread
        t = threading.Thread(target=self._train_worker, daemon=True)
        t.start()
        messagebox.showinfo("Training", "Training started in background — you'll be notified when it's done.")

    def _train_worker(self):
        try:
            df = self.state.df.copy()
            # basic validation
            if self.state.target not in df.columns:
                raise ValueError("Selected target not in dataset")
            if any(f not in df.columns for f in self.state.features):
                raise ValueError("One or more features missing from dataset")

            # coerce numeric mosquito if present
            if 'mosquito_count' in self.state.features:
                df['mosquito_count'] = pd.to_numeric(df['mosquito_count'], errors='coerce')

            X = df[self.state.features].copy()
            y = df[self.state.target].copy()

            X_enc = self._encode_features(X, fit=True)

            test_size = float(self.test_split_var.get())
            # use sklearn split (shuffled)
            X_train, X_val, y_train, y_val = train_test_split(X_enc, y, test_size=test_size, random_state=42)

            # For speed + interpretability, use RandomForest by default
            model = RandomForestRegressor(n_estimators=200, random_state=42)
            model.fit(X_train, y_train)

            predictions = model.predict(X_val)
            mae = mean_absolute_error(y_val, predictions)
            r2 = r2_score(y_val, predictions)

            # store in state
            self.state.model = model
            self.state.X_train = X_train
            self.state.X_val = X_val
            self.state.y_train = y_train
            self.state.y_val = y_val
            self.state.predictions = predictions
            self.state.mae = mae
            self.state.r2 = r2

            # UI updates must be run on main thread
            self.root.after(0, lambda: self.mae_label.config(text=f"MAE: {mae:.4f}"))
            self.root.after(0, lambda: self.r2_label.config(text=f"R²: {r2:.4f}"))
            self.root.after(0, lambda: self.plot_train_results(y_val, predictions))
            self.root.after(0, lambda: messagebox.showinfo("Success", "Model trained successfully"))

            # show feature importances (if RF)
            try:
                importances = model.feature_importances_
                feat_imp = sorted(zip(self.state.features, importances), key=lambda x: x[1], reverse=True)
                text = "Feature importances:\n" + "\n".join([f"{f}: {imp:.3f}" for f, imp in feat_imp])
                self.root.after(0, lambda: messagebox.showinfo("Feature importances", text))
            except Exception:
                pass

            # optionally prompt to save model (ask on main thread)
            def ask_save():
                model_path = filedialog.asksaveasfilename(title="Save trained model as...", defaultextension=".joblib",
                                                          filetypes=[("Joblib", "*.joblib"), ("All files", "*.*")])
                if model_path:
                    try:
                        joblib.dump({'model': model, 'cat_mappings': self.state.cat_mappings,
                                     'features': self.state.features, 'target': self.state.target}, model_path)
                        self.state.model_path = model_path
                        messagebox.showinfo("Saved", f"Model saved to:\n{model_path}")
                    except Exception as e:
                        messagebox.showerror("Save error", f"Failed to save model: {e}")
            self.root.after(100, ask_save)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Training failed: {e}"))

    def plot_train_results(self, y_true, y_pred):
        for widget in self.train_plot_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_true, y_pred, alpha=0.5)
        try:
            min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
            max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x')
        except Exception:
            pass

        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('True vs Predicted')
        ax.legend()
        ax.grid(True, alpha=0.3)

        self.state.last_figure = fig

        canvas = FigureCanvasTkAgg(fig, self.train_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    # ----------------------------
    # SIMULATION Tab
    # ----------------------------
    def create_simulate_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text='SIMULATE')

        control_frame = ttk.Frame(tab)
        control_frame.pack(pady=10, padx=20, fill='x')

        ttk.Label(control_frame, text="GM Reduction % (0-80):").grid(row=0, column=0, sticky='w', pady=5)
        slider_frame = ttk.Frame(control_frame)
        slider_frame.grid(row=0, column=1, sticky='w', pady=5, padx=10)

        self.gm_reduction_var = tk.DoubleVar(value=50.0)
        self.gm_slider = ttk.Scale(slider_frame, from_=0, to=80, orient='horizontal',
                                   variable=self.gm_reduction_var, length=200)
        self.gm_slider.pack(side='left')
        self.gm_slider_label = ttk.Label(slider_frame, text="50%")
        self.gm_slider_label.pack(side='left', padx=10)
        self.gm_reduction_var.trace_add('write', lambda *args: self.update_gm_label())

        ttk.Label(control_frame, text="Ecological Threshold (min mosquito_count):").grid(row=1, column=0, sticky='w', pady=5)
        self.threshold_entry = ttk.Entry(control_frame, width=15)
        self.threshold_entry.insert(0, f"{self.state.ecological_threshold:.2f}")
        self.threshold_entry.grid(row=1, column=1, sticky='w', pady=5, padx=10)

        # Suggest threshold button
        ttk.Button(control_frame, text="Suggest Threshold", command=self.on_suggest_threshold).grid(row=1, column=2, sticky='w', padx=6)

        ttk.Button(control_frame, text="Run simulation", command=self.run_simulation).grid(row=2, column=1, sticky='w', pady=10, padx=10)
        ttk.Button(control_frame, text="Find Optimal GM %", command=self.on_find_optimal_clicked).grid(row=2, column=2, sticky='w', pady=10, padx=10)
        ttk.Button(control_frame, text="Export simulation CSV", command=self.export_simulation_csv).grid(row=2, column=0, sticky='w', pady=10, padx=10)

        self.sim_plot_frame = ttk.Frame(tab)
        self.sim_plot_frame.pack(fill='both', expand=True, padx=20, pady=10)

        summary_frame = ttk.LabelFrame(tab, text="Simulation Summary")
        summary_frame.pack(pady=10, padx=20, fill='x')

        self.sim_summary_text = Text(summary_frame, height=8, width=80)
        self.sim_summary_text.pack(pady=5, padx=5)

    def update_gm_label(self, *args):
        self.gm_slider_label.config(text=f"{self.gm_reduction_var.get():.0f}%")

    def on_suggest_threshold(self):
        suggested = suggest_threshold_from_data(self.state.df, percentile=5)
        if suggested is None:
            messagebox.showinfo("Suggest threshold", "Could not suggest threshold (missing mosquito_count or data).")
            return
        self.threshold_entry.delete(0, tk.END)
        self.threshold_entry.insert(0, f"{suggested:.2f}")
        messagebox.showinfo("Suggested threshold", f"Suggested (5th percentile): {suggested:.2f}")

    def run_simulation(self):
        if self.state.model is None:
            messagebox.showwarning("Warning", "Please train or load a model first")
            return

        if 'mosquito_count' not in self.state.features:
            messagebox.showwarning("Warning", "mosquito_count must be in features for simulation")
            return

        try:
            gm_pct = float(self.gm_reduction_var.get())
            threshold = float(self.threshold_entry.get())
            self.state.gm_reduction_pct = gm_pct
            self.state.ecological_threshold = threshold

            X_sim_raw = self.state.df[self.state.features].copy()
            X_sim = self._encode_features(X_sim_raw, fit=False)

            base_pred = self.state.model.predict(X_sim)

            X_gm = X_sim.copy()
            X_gm['mosquito_count'] = pd.to_numeric(X_gm['mosquito_count'], errors='coerce').fillna(0)
            X_gm['mosquito_count'] = X_gm['mosquito_count'] * (1 - gm_pct / 100.0)
            X_gm['mosquito_count'] = np.maximum(1, X_gm['mosquito_count'])

            gm_pred = self.state.model.predict(X_gm)

            min_mosquito = float(X_gm['mosquito_count'].min())
            ecological_risk = "HIGH" if min_mosquito < threshold else "OK"

            base_mean = float(np.mean(base_pred))
            gm_mean = float(np.mean(gm_pred))
            diff = gm_mean - base_mean
            diff_pct = (diff / base_mean * 100) if base_mean != 0 else 0.0

            if gm_mean < base_mean and ecological_risk == "OK":
                recommendation = "Proceed: GM mosquitoes reduce disease cases without ecological risk"
            elif gm_mean >= base_mean:
                recommendation = "No benefit: GM mosquitoes do not reduce disease cases"
            else:
                recommendation = "High risk: Ecological threshold violated. Do NOT proceed."

            summary = f"""GM Reduction: {gm_pct:.0f}%
Base Mean Disease Cases: {base_mean:.2f}
GM Mean Disease Cases: {gm_mean:.2f}
Difference: {diff:.2f} ({diff_pct:.2f}%)
Ecological Risk: {ecological_risk}
Minimum Mosquito Count: {min_mosquito:.2f}
Threshold: {threshold:.2f}

Recommendation: {recommendation}
"""

            self.sim_summary_text.delete(1.0, tk.END)
            self.sim_summary_text.insert(1.0, summary)

            sim_df = pd.DataFrame({
                'base_pred': base_pred,
                'gm_pred': gm_pred,
                'diff': gm_pred - base_pred
            })
            self.state.last_sim_df = sim_df

            self.plot_simulation(base_pred, gm_pred)

        except Exception as e:
            messagebox.showerror("Error", f"Simulation failed: {str(e)}")

    def plot_simulation(self, base_pred, gm_pred):
        for widget in self.sim_plot_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(base_pred))
        ax.plot(x, base_pred, label='Without GM', alpha=0.8)
        ax.plot(x, gm_pred, label='With GM', alpha=0.8)

        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Predicted Disease Cases')
        ax.set_title('Disease Cases: With GM vs Without GM')
        ax.legend()
        ax.grid(True, alpha=0.3)

        self.state.last_figure = fig

        canvas = FigureCanvasTkAgg(fig, self.sim_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def export_simulation_csv(self):
        if self.state.last_sim_df is None:
            messagebox.showwarning("Warning", "No simulation available to export. Run a simulation first.")
            return
        fname = filedialog.asksaveasfilename(title="Save simulation CSV", defaultextension=".csv",
                                             filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if not fname:
            return
        try:
            self.state.last_sim_df.to_csv(fname, index=False)
            messagebox.showinfo("Success", f"Simulation exported to {fname}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export simulation CSV: {str(e)}")

    # ----------------------------
    # Find optimal GM % (grid search)
    # ----------------------------
    def find_optimal_gm_release(self, step_pct=1.0, ecological_threshold=None, mode='minimize_disease', target_reduction_pct=20.0, verbose=False):
        if self.state.model is None:
            return {'success': False, 'message': 'No trained model.'}
        if self.state.df is None:
            return {'success': False, 'message': 'No data loaded.'}
        if 'mosquito_count' not in self.state.features:
            return {'success': False, 'message': "Feature 'mosquito_count' must be in selected features for simulation."}

        X_raw = self.state.df[self.state.features].copy()
        # simple encoding (same as training)
        for c in X_raw.columns:
            if X_raw[c].dtype == 'O':
                X_raw[c], _ = pd.factorize(X_raw[c])

        base_pred = self.state.model.predict(X_raw)
        base_mean = float(np.mean(base_pred))

        best = None
        grid = [round(x, 6) for x in np.arange(0.0, 80.0 + 1e-9, step_pct)]

        thr = ecological_threshold if ecological_threshold is not None else self.state.ecological_threshold

        for gm_pct in grid:
            X_gm = X_raw.copy()
            X_gm['mosquito_count'] = pd.to_numeric(X_gm['mosquito_count'], errors='coerce').fillna(0)
            X_gm['mosquito_count'] = X_gm['mosquito_count'] * (1 - gm_pct / 100.0)
            X_gm['mosquito_count'] = np.maximum(1, X_gm['mosquito_count'])
            min_mosquito = float(X_gm['mosquito_count'].min())

            if min_mosquito < thr:
                if verbose:
                    print(f"gm {gm_pct}% -> min_mosquito {min_mosquito:.2f} violates threshold {thr}")
                continue

            try:
                gm_pred = self.state.model.predict(X_gm)
            except Exception as e:
                if verbose:
                    print("Prediction error at gm", gm_pct, e)
                continue

            gm_mean = float(np.mean(gm_pred))
            reduction_pct = ((base_mean - gm_mean) / base_mean * 100.0) if base_mean != 0 else 0.0

            if mode == 'minimize_disease':
                if best is None or gm_mean < best['gm_mean'] - 1e-9 or (abs(gm_mean - best['gm_mean']) < 1e-9 and gm_pct < best['gm_pct']):
                    best = {'gm_pct': gm_pct, 'base_mean': base_mean, 'gm_mean': gm_mean,
                            'min_mosquito': min_mosquito, 'reduction_pct': reduction_pct}
            else:  # min_release_for_target
                if reduction_pct >= target_reduction_pct:
                    if best is None or gm_pct < best['gm_pct']:
                        best = {'gm_pct': gm_pct, 'base_mean': base_mean, 'gm_mean': gm_mean,
                                'min_mosquito': min_mosquito, 'reduction_pct': reduction_pct}

        if best is None:
            return {'success': False, 'message': 'No safe GM percentage found that satisfies constraints.', 'base_mean': base_mean}

        best['success'] = True
        return best

    def on_find_optimal_clicked(self):
        if self.state.model is None:
            messagebox.showwarning("No model", "Train or load a model first")
            return
        # ask user mode
        ans = messagebox.askquestion("Mode", "Choose mode:\nYes = minimize disease (safe)\nNo = find minimal release for a target reduction")
        if ans == 'yes':
            mode = 'minimize_disease'
            tr = None
        else:
            mode = 'min_release_for_target'
            tr = simpledialog.askfloat("Target reduction", "Enter target reduction % (e.g. 20)", minvalue=0.0, maxvalue=100.0, initialvalue=20.0)
            if tr is None:
                return

        try:
            thr = float(self.threshold_entry.get())
        except:
            thr = self.state.ecological_threshold

        # run search (fast)
        res = self.find_optimal_gm_release(step_pct=1.0, ecological_threshold=thr, mode=mode, target_reduction_pct=(tr or 0.0))
        if not res.get('success'):
            messagebox.showwarning("No safe solution", res.get('message', 'No solution found. Try relaxing constraints or check data.'))
            return

        gm = res['gm_pct']; base = res['base_mean']; gm_mean = res['gm_mean']; rp = res['reduction_pct']
        msg = (f"Recommended GM release: {gm:.0f}%\n"
               f"Avg disease before: {base:.2f}\n"
               f"Avg disease after: {gm_mean:.2f}\n"
               f"Expected reduction: {rp:.2f}%\n")
        messagebox.showinfo("Optimal GM %", msg)

        # apply recommended gm and re-run simulation to update UI
        self.gm_reduction_var.set(gm)
        self.threshold_entry.delete(0, tk.END); self.threshold_entry.insert(0, str(thr))
        self.run_simulation()

    # ----------------------------
    # MODEL load/save
    # ----------------------------
    def load_model(self):
        path = filedialog.askopenfilename(title="Load model (joblib)", filetypes=[("Joblib", "*.joblib"), ("All files", "*.*")])
        if not path:
            return
        try:
            data = joblib.load(path)
            if isinstance(data, dict) and 'model' in data:
                self.state.model = data['model']
                self.state.cat_mappings = data.get('cat_mappings', {})
                self.state.features = data.get('features', self.state.features)
                self.state.target = data.get('target', self.state.target)
                self.state.model_path = path
                messagebox.showinfo("Success", f"Model loaded from {path}")
                # update UI selections if possible
                if self.state.df is not None:
                    self._after_data_load()
            else:
                self.state.model = data
                messagebox.showinfo("Success", f"Model loaded from {path} (no metadata found)")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    # ----------------------------
    # REPORT Tab
    # ----------------------------
    def create_report_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text='REPORT')

        ttk.Label(tab, text="Notes:").pack(pady=10)

        self.notes_text = Text(tab, height=15, width=80)
        self.notes_text.pack(padx=20, pady=5)

        ttk.Button(tab, text="Save summary report", command=self.save_report).pack(pady=20)

    def save_report(self):
        if self.state.model is None:
            messagebox.showwarning("Warning", "No model trained or loaded yet")
            return

        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            notes = self.notes_text.get(1.0, tk.END).strip()

            report = f"""GM Mosquito AI Impact Study Report
Generated: {timestamp}

=== DATA CONFIGURATION ===
Target Column: {self.state.target}
Features: {', '.join(self.state.features) if self.state.features else 'N/A'}
Test Split: {self.test_split_var.get():.2f}

=== MODEL VALIDATION ===
MAE: {self.state.mae:.4f}
R²: {self.state.r2:.4f}
Model Path: {self.state.model_path or 'Not saved'}

=== GM SIMULATION ===
GM Reduction: {self.state.gm_reduction_pct:.0f}%
Ecological Threshold: {self.state.ecological_threshold:.2f}

=== NOTES ===
{notes}

"""
            fname = filedialog.asksaveasfilename(title="Save report", defaultextension=".txt",
                                                 filetypes=[("Text file", "*.txt"), ("All files", "*.*")])
            if not fname:
                return

            with open(fname, 'w') as f:
                f.write(report)

            if self.state.last_figure is not None:
                plot_path = os.path.splitext(fname)[0] + "_plot.png"
                try:
                    self.state.last_figure.savefig(plot_path, dpi=150, bbox_inches='tight')
                except Exception:
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                messagebox.showinfo("Success", f"Report saved as {fname}\nPlot saved as {plot_path}")
            else:
                messagebox.showinfo("Success", f"Report saved as {fname}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save report: {str(e)}")


# ----------------------------
# Run the app
# ----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = GMMosquitoAIGUI(root)
    root.mainloop()
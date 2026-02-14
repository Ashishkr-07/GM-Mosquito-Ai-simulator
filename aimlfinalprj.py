#!/usr/bin/env python3
"""
super_simple_gm_gui.py
Very simple GUI for GM mosquito simulation:
1) Load CSV or generate demo data
2) Auto-select sensible features & target
3) Train model with one click
4) Simulate GM reduction with slider and get plain recommendation
Designed for non-technical users.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Text
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import joblib
import os

# ---------- Helper functions ----------
def auto_choose_features_and_target(df):
    """Pick a sensible target and features automatically.
       Prefer 'disease_cases' as target if present, else numeric column with name containing 'case' or last numeric column."""
    numcols = df.select_dtypes(include=[np.number]).columns.tolist()
    target = None
    for name in ['disease_cases', 'cases', 'disease', 'disease_case', 'disease_count']:
        if name in df.columns:
            target = name
            break
    if target is None:
        # choose numeric column with 'case' in name, else last numeric col
        for c in numcols:
            if 'case' in c.lower():
                target = c
                break
    if target is None and numcols:
        target = numcols[-1]
    # features = numeric columns except target (also keep mosquito_count if present)
    if target:
        features = [c for c in numcols if c != target]
    else:
        features = numcols
    # if mosquito_count exists, ensure it's included
    if 'mosquito_count' in df.columns and 'mosquito_count' not in features:
        features.insert(0, 'mosquito_count')
    return features, target

def simple_recommendation(base_mean, gm_mean, min_mosquito, threshold):
    """Plain-language recommendation rule."""
    if min_mosquito < threshold:
        return "High risk — mosquito count would drop below safe threshold. Do NOT proceed."
    if gm_mean < base_mean:
        return "Good: GM reduces predicted disease cases. Proceed with caution."
    return "No improvement: GM does not reduce predicted disease cases."

# ---------- GUI ----------
class SuperSimpleApp:
    def __init__(self, root):
        self.root = root
        root.title("Simple GM Mosquito Simulator")
        root.geometry("880x640")
        self.df = None
        self.features = []
        self.target = None
        self.model = None
        self.last_sim_df = None
        self.last_fig = None

        # Top instructions
        ttk.Label(root, text="Simple GM Mosquito App — Follow the 4 big buttons below", font=("Arial", 14, "bold")).pack(pady=8)

        # Buttons row
        row = ttk.Frame(root); row.pack(fill='x', padx=12)
        ttk.Button(row, text="1) Load CSV", command=self.load_csv, width=18).pack(side='left', padx=6, pady=6)
        ttk.Button(row, text="Or: Generate Demo Data", command=self.generate_demo, width=18).pack(side='left', padx=6, pady=6)
        ttk.Button(row, text="2) Train Model", command=self.train_model, width=18).pack(side='left', padx=6, pady=6)
        ttk.Button(row, text="3) Run Simulation", command=self.open_simulation_window, width=18).pack(side='left', padx=6, pady=6)

        # Preview + status
        status_frame = ttk.Frame(root); status_frame.pack(fill='both', expand=False, padx=12, pady=6)
        ttk.Label(status_frame, text="Data preview (first 10 rows):", font=("Arial", 11)).pack(anchor='w')
        self.preview_text = Text(status_frame, height=10, width=104)
        self.preview_text.pack(side='left', padx=0)
        scroll = ttk.Scrollbar(status_frame, orient='vertical', command=self.preview_text.yview)
        scroll.pack(side='left', fill='y')
        self.preview_text.config(yscrollcommand=scroll.set)

        # Model metrics + save
        metrics_frame = ttk.Frame(root); metrics_frame.pack(fill='x', padx=12, pady=6)
        self.mae_label = ttk.Label(metrics_frame, text="MAE: -", font=("Arial", 11))
        self.mae_label.pack(side='left', padx=6)
        self.r2_label = ttk.Label(metrics_frame, text="R²: -", font=("Arial", 11))
        self.r2_label.pack(side='left', padx=6)
        ttk.Button(metrics_frame, text="Save Model", command=self.save_model).pack(side='right', padx=6)

        # Plot area
        plot_frame = ttk.LabelFrame(root, text="Training: True vs Predicted (validation set)")
        plot_frame.pack(fill='both', expand=True, padx=12, pady=8)
        self.plot_container = ttk.Frame(plot_frame)
        self.plot_container.pack(fill='both', expand=True)

        # Bottom quick help
        ttk.Label(root, text="Quick help: 1) Load or generate data → 2) Train → 3) Run simulation → Export results from simulation window", foreground="gray").pack(pady=6)

    def load_csv(self):
        p = filedialog.askopenfilename(filetypes=[("CSV","*.csv"),("All files","*.*")])
        if not p: return
        try:
            self.df = pd.read_csv(p)
            self._after_data_load()
            messagebox.showinfo("Loaded", f"Loaded {os.path.basename(p)} ({len(self.df)} rows).")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load CSV: {e}")

    def generate_demo(self):
        # simple demo similar to your previous demo
        np.random.seed(1)
        n = 365
        day = np.arange(n)
        mosquito = 500 + 200*np.sin(2*np.pi*(day-60)/365) + np.random.normal(0,40,n)
        mosquito = np.maximum(10, mosquito)
        temp = 20 + 8*np.sin(2*np.pi*day/365) + np.random.normal(0,2,n)
        rain = np.maximum(0, 50 + 30*np.sin(2*np.pi*(day-90)/365) + np.random.normal(0,10,n))
        past = np.random.poisson(20, n)
        disease = (10 + 0.05*mosquito + 0.3*past + 0.2*rain - 0.1*temp + np.random.normal(0,5,n)).astype(int)
        self.df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=n),
            "mosquito_count": mosquito,
            "avg_temp": temp,
            "rainfall": rain,
            "past_cases": past,
            "disease_cases": disease
        })
        self._after_data_load()
        messagebox.showinfo("Demo", "Demo data created (365 rows).")

    def _after_data_load(self):
        # auto choose features and target and show preview
        self.features, self.target = auto_choose_features_and_target(self.df)
        # display top 10 rows
        self.preview_text.delete(1.0, tk.END)
        try:
            self.preview_text.insert(1.0, self.df.head(10).to_string(index=False))
        except Exception:
            self.preview_text.insert(1.0, repr(self.df.head(10)))
        # show chosen features/target plainly to user
        choices = f"Auto-chosen target: {self.target}\nAuto-chosen features: {', '.join(self.features)}"
        messagebox.showinfo("Auto selection", choices)

    def train_model(self):
        if self.df is None:
            messagebox.showwarning("No data", "Load or generate data first (button 1 or 2).")
            return
        if not self.features or not self.target:
            messagebox.showwarning("No selection", "No features/target chosen.")
            return
        try:
            # prepare X,y (convert non-numeric simply)
            X = self.df[self.features].copy()
            for c in X.columns:
                if X[c].dtype == 'O':
                    X[c], _ = pd.factorize(X[c])
            y = self.df[self.target].values
            # simple train/validation split: last 20% as validation
            n = len(X)
            split = int(n * 0.8)
            X_train, X_val = X.iloc[:split], X.iloc[split:]
            y_train, y_val = y[:split], y[split:]
            # train simple RandomForest
            model = RandomForestRegressor(n_estimators=150, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            mae = mean_absolute_error(y_val, preds)
            r2 = r2_score(y_val, preds)
            self.model = model
            self.mae_label.config(text=f"MAE: {mae:.3f}")
            self.r2_label.config(text=f"R²: {r2:.3f}")
            # plot simple true vs pred
            self._plot_true_vs_pred(y_val, preds)
            messagebox.showinfo("Trained", "Model trained successfully. See metrics and plot.")
        except Exception as e:
            messagebox.showerror("Train error", f"Training failed: {e}")

    def _plot_true_vs_pred(self, y_true, y_pred):
        for child in self.plot_container.winfo_children():
            child.destroy()
        fig, ax = plt.subplots(figsize=(7,4))
        ax.scatter(y_true, y_pred, alpha=0.6)
        mn = min(float(np.min(y_true)), float(np.min(y_pred)))
        mx = max(float(np.max(y_true)), float(np.max(y_pred)))
        ax.plot([mn,mx],[mn,mx],'r--',linewidth=1)
        ax.set_xlabel("True disease cases")
        ax.set_ylabel("Predicted disease cases")
        ax.set_title("Validation: True vs Predicted")
        canvas = FigureCanvasTkAgg(fig, master=self.plot_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        self.last_fig = fig

    def open_simulation_window(self):
        if self.model is None:
            messagebox.showwarning("No model", "Train a model first (button 3).")
            return
        # open a small window with slider & Run button
        sim = tk.Toplevel(self.root)
        sim.title("Run Simulation (simple)")
        sim.geometry("520x420")
        ttk.Label(sim, text="Move slider to choose GM reduction (% of wild mosquitos removed)", wraplength=480).pack(pady=8)
        gm_var = tk.DoubleVar(value=50.0)
        slider = ttk.Scale(sim, from_=0, to=80, orient='horizontal', variable=gm_var, length=420)
        slider.pack(pady=6)
        ttk.Label(sim, text="Set ecological threshold (minimum allowed mosquito_count):").pack(pady=6)
        thresh_entry = ttk.Entry(sim); thresh_entry.insert(0, "100"); thresh_entry.pack(pady=6)

        # area for summary and small plot
        result_text = Text(sim, height=8, width=62); result_text.pack(pady=6)
        plot_holder = ttk.Frame(sim); plot_holder.pack(fill='both', expand=True, padx=6, pady=6)

        def run_it():
            try:
                gm_pct = float(gm_var.get())
                threshold = float(thresh_entry.get())
                X_raw = self.df[self.features].copy()
                # same simple encoding as training
                for c in X_raw.columns:
                    if X_raw[c].dtype == 'O':
                        X_raw[c], _ = pd.factorize(X_raw[c])
                # base predictions
                base_pred = self.model.predict(X_raw)
                # simulate reducing mosquito_count (if exists)
                if 'mosquito_count' not in X_raw.columns:
                    messagebox.showwarning("Missing column", "Feature 'mosquito_count' not found in features.")
                    return
                X_gm = X_raw.copy()
                X_gm['mosquito_count'] = pd.to_numeric(X_gm['mosquito_count'], errors='coerce').fillna(0)
                X_gm['mosquito_count'] = X_gm['mosquito_count'] * (1 - gm_pct/100.0)
                X_gm['mosquito_count'] = np.maximum(1, X_gm['mosquito_count'])
                gm_pred = self.model.predict(X_gm)
                base_mean = float(np.mean(base_pred)); gm_mean = float(np.mean(gm_pred))
                diff = gm_mean - base_mean
                diff_pct = (diff/base_mean*100) if base_mean!=0 else 0.0
                min_mos = float(X_gm['mosquito_count'].min())
                rec = simple_recommendation(base_mean, gm_mean, min_mos, threshold)
                summary = (
                    f"GM Reduction: {gm_pct:.0f}%\n"
                    f"Average disease (without GM): {base_mean:.2f}\n"
                    f"Average disease (with GM): {gm_mean:.2f}\n"
                    f"Change: {diff:.2f} ({diff_pct:.2f}%)\n"
                    f"Minimum mosquito count after GM: {min_mos:.2f}\n"
                    f"Threshold: {threshold:.2f}\n\nRecommendation:\n{rec}\n"
                )
                result_text.delete(1.0, tk.END); result_text.insert(1.0, summary)
                # save last sim df for export
                self.last_sim_df = pd.DataFrame({'without_gm': base_pred, 'with_gm': gm_pred})
                # plot small time series
                for child in plot_holder.winfo_children(): child.destroy()
                fig, ax = plt.subplots(figsize=(6,2.5))
                idx = np.arange(len(base_pred))
                ax.plot(idx[:200], base_pred[:200], label='Without GM', alpha=0.8)
                ax.plot(idx[:200], gm_pred[:200], label='With GM', alpha=0.8)
                ax.set_title("First 200 samples (predicted)")
                ax.legend(fontsize=8)
                canvas = FigureCanvasTkAgg(fig, master=plot_holder)
                canvas.draw(); canvas.get_tk_widget().pack(fill='both', expand=True)
            except Exception as e:
                messagebox.showerror("Simulation error", f"{e}")

        ttk.Button(sim, text="Run Simulation", command=run_it).pack(pady=6)
        ttk.Button(sim, text="Export results CSV", command=lambda: self._export_last_sim(sim)).pack(pady=0)

    def _export_last_sim(self, parent):
        if self.last_sim_df is None:
            messagebox.showwarning("No results", "Run a simulation first.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv", parent=parent)
        if not p: return
        try:
            self.last_sim_df.to_csv(p, index=False)
            messagebox.showinfo("Saved", f"Simulation exported to {p}")
        except Exception as e:
            messagebox.showerror("Save error", f"{e}")

    def save_model(self):
        if self.model is None:
            messagebox.showwarning("No model", "Train a model first.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".joblib")
        if not p: return
        try:
            joblib.dump(self.model, p)
            messagebox.showinfo("Saved", f"Model saved to {p}")
        except Exception as e:
            messagebox.showerror("Save error", f"{e}")

# ---------- Run ----------
if __name__ == "__main__":
    root = tk.Tk()
    app = SuperSimpleApp(root)
    root.mainloop()
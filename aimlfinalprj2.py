#!/usr/bin/env python3
"""
Save and run with your Python environment (install dependencies: pandas, numpy,
scikit-learn, matplotlib, joblib). The GUI is simple and designed for non-technical users.
"""

# -----------------------------
# Standard library imports
# -----------------------------
import os  # utilities for file paths and file operations (used when saving files)

# -----------------------------
# Third-party libraries (must be installed via pip)
# -----------------------------
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Text
# tkinter (and ttk) provide GUI elements (windows, buttons, dialogs, text boxes).
# - ttk gives themed widgets with a nicer look.
# - filedialog and messagebox are used for open/save dialogs and simple popups.

import pandas as pd      # pandas: dataframes for tabular data; used to load CSVs and export simulation results
import numpy as np       # numpy: numerical operations and arrays; used for demo data generation and numeric conversions

# scikit-learn (sklearn) supplies classical ML models & metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# matplotlib for plotting; FigureCanvasTkAgg embeds plots in Tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# joblib: simple model save/load 
import joblib

# -----------------------------
# Helper functions
# -----------------------------

def auto_choose_features_and_target(df):
    """
    Automatically choose a sensible target column and numeric features.

    Explanation (why):
    - Non-technical users may not know which column is target; prefer common names
      like 'disease_cases'. If not present, pick a numeric column with 'case' in its name
      or fall back to the last numeric column.
    - Features are numeric columns except the chosen target. If 'mosquito_count' exists,
      keep it as a feature because the simulation relies on it.
    """
    numcols = df.select_dtypes(include=[np.number]).columns.tolist()
    target = None
    # prefer exact names commonly used in datasets
    for name in ['disease_cases', 'cases', 'disease', 'disease_case', 'disease_count']:
        if name in df.columns:
            target = name
            break
    if target is None:
        # if no exact match, find any numeric column with 'case' in its name
        for c in numcols:
            if 'case' in c.lower():
                target = c
                break
    if target is None and numcols:
        # fallback: last numeric column (a simple heuristic)
        target = numcols[-1]
    # choose features as numeric columns excluding the target
    if target:
        features = [c for c in numcols if c != target]
    else:
        features = numcols
    # ensure mosquito_count is included if present (important for simulation)
    if 'mosquito_count' in df.columns and 'mosquito_count' not in features:
        features.insert(0, 'mosquito_count')
    return features, target


def simple_recommendation(base_mean, gm_mean, min_mosquito, threshold):
    """
    Produce a plain-language recommendation based on:
    - base_mean: predicted average disease without GM
    - gm_mean: predicted average disease with GM
    - min_mosquito: minimum mosquito_count after GM
    - threshold: ecological threshold (minimum safe mosquito count)

    Rules (simple and conservative):
    - If the minimum mosquito count after GM is below the threshold -> high risk
    - Else if average disease decreases -> GM appears beneficial
    - Otherwise -> no improvement
    """
    if min_mosquito < threshold:
        return "High risk — mosquito count would drop below safe threshold. Do NOT proceed."
    if gm_mean < base_mean:
        return "Good: GM reduces predicted disease cases. Proceed with caution."
    return "No improvement: GM does not reduce predicted disease cases."

# NEW --- Time-series feature engineering (makes model smarter)
def add_time_features(df):
    """
    Create lag & rolling features so the model learns delayed effects.
    This makes the project look much more advanced.
    """
    df = df.copy()

    # sort by date if dataset has date column
    if 'date' in df.columns:
        df = df.sort_values('date')

    lag_cols = ['mosquito_count', 'rainfall', 'past_cases']

    for col in lag_cols:
        if col in df.columns:
            df[f'{col}_lag7'] = df[col].shift(7)
            df[f'{col}_lag14'] = df[col].shift(14)
            df[f'{col}_rolling7'] = df[col].rolling(7).mean()

    df = df.dropna()
    return df

# NEW --- realistic mosquito population decay simulation
def simulate_gm_population(mosquito_series, reduction_pct):
    """
    GM mosquitoes reduce population gradually (not instantly).
    This simulates a realistic biological decline over time.
    """
    mos = mosquito_series.copy().astype(float)
    decay = reduction_pct / 100 / 30  # spread effect over ~30 days

    for i in range(1, len(mos)):
        mos.iloc[i] = mos.iloc[i-1] * (1 - decay)

    return np.maximum(1, mos)

# NEW --- decision intelligence score
def calculate_risk_score(base_mean, gm_mean, min_mosquito, threshold):
    reduction = (base_mean - gm_mean) / base_mean * 100
    eco_risk = max(0, (threshold - min_mosquito) / threshold * 100)

    score = reduction - eco_risk

    if score > 20:
        level = "SAFE"
    elif score > 5:
        level = "MODERATE"
    else:
        level = "RISKY"

    return score, level


# -----------------------------
# Main GUI application class
# -----------------------------
class SuperSimpleApp:
    """
    A compact Tkinter application providing 3 main actions for non-technical users:
    1) Load CSV or generate demo data
    2) Train a RandomForest model automatically
    3) Run a simple simulation that reduces mosquito_count by a chosen percent and
       shows predicted impact and a clear recommendation

    The code below creates the window, widgets, and connects buttons to functions.
    """
    def __init__(self, root):
        # store root Tk instance
        self.root = root
        root.title("Simple GM Mosquito Simulator")
        root.geometry("880x640")

        # internal state variables (keeps track of data, models, last results)
        self.df = None
        self.features = []
        self.target = None
        self.model = None
        self.last_sim_df = None
        self.last_fig = None

        # ---------- Top label / instructions ----------
        ttk.Label(root, text="Simple GM Mosquito App — Follow the 4 big buttons below",
                  font=("Arial", 14, "bold")).pack(pady=8)

        # ---------- Buttons row ----------
        # Buttons are intentionally labeled with numbers for non-technical flow
        row = ttk.Frame(root); row.pack(fill='x', padx=12)
        ttk.Button(row, text="1) Load CSV", command=self.load_csv, width=18).pack(side='left', padx=6, pady=6)
        ttk.Button(row, text="Or: Generate Demo Data", command=self.generate_demo, width=18).pack(side='left', padx=6, pady=6)
        ttk.Button(row, text="2) Train Model", command=self.train_model, width=18).pack(side='left', padx=6, pady=6)
        ttk.Button(row, text="3) Run Simulation", command=self.open_simulation_window, width=18).pack(side='left', padx=6, pady=6)

        # ---------- Data preview area ----------
        status_frame = ttk.Frame(root); status_frame.pack(fill='both', expand=False, padx=12, pady=6)
        ttk.Label(status_frame, text="Data preview (first 10 rows):", font=("Arial", 11)).pack(anchor='w')
        self.preview_text = Text(status_frame, height=10, width=104)
        self.preview_text.pack(side='left', padx=0)
        scroll = ttk.Scrollbar(status_frame, orient='vertical', command=self.preview_text.yview)
        scroll.pack(side='left', fill='y')
        self.preview_text.config(yscrollcommand=scroll.set)

        # ---------- Model metrics area ----------
        metrics_frame = ttk.Frame(root); metrics_frame.pack(fill='x', padx=12, pady=6)
        self.mae_label = ttk.Label(metrics_frame, text="MAE: -", font=("Arial", 11))
        self.mae_label.pack(side='left', padx=6)
        self.r2_label = ttk.Label(metrics_frame, text="R²: -", font=("Arial", 11))
        self.r2_label.pack(side='left', padx=6)
        ttk.Button(metrics_frame, text="Save Model", command=self.save_model).pack(side='right', padx=6)

        # ---------- Plot area (True vs Predicted) ----------
        plot_frame = ttk.LabelFrame(root, text="Training: True vs Predicted (validation set)")
        plot_frame.pack(fill='both', expand=True, padx=12, pady=8)
        self.plot_container = ttk.Frame(plot_frame)
        self.plot_container.pack(fill='both', expand=True)

        # ---------- Quick help text ----------
        ttk.Label(root, text="Quick help: 1) Load or generate data → 2) Train → 3) Run simulation → Export results from simulation window",
                  foreground="gray").pack(pady=6)

    # -----------------------------
    # Data loading / demo generation
    # -----------------------------
    def load_csv(self):
        """
        Open a file dialog and load a CSV into a pandas DataFrame.
        After loading, call _after_data_load() to auto-choose features and show a preview.
        """
        p = filedialog.askopenfilename(filetypes=[("CSV","*.csv"),("All files","*.*")])
        if not p: return
        try:
            self.df = pd.read_csv(p)
            self._after_data_load()
            messagebox.showinfo("Loaded", f"Loaded {os.path.basename(p)} ({len(self.df)} rows).")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load CSV: {e}")

    def generate_demo(self):
        """
        Create a simple synthetic dataset with seasonal mosquito counts, temperature, rainfall,
        past cases and disease_cases. This is useful for trying the app without real data.
        """
        np.random.seed(1)
        n = 365
        day = np.arange(n)
        # seasonal mosquito pattern + noise
        mosquito = 500 + 200*np.sin(2*np.pi*(day-60)/365) + np.random.normal(0,40,n)
        mosquito = np.maximum(10, mosquito)
        temp = 20 + 8*np.sin(2*np.pi*day/365) + np.random.normal(0,2,n)
        rain = np.maximum(0, 50 + 30*np.sin(2*np.pi*(day-90)/365) + np.random.normal(0,10,n))
        past = np.random.poisson(20, n)
        # disease cases synthesised as linear function of mosquito + past + weather + noise
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
        """
        Called after loading or generating data. It:
        - automatically chooses sensible features and a target using helper
        - displays the first 10 rows to the user in the preview box
        - shows a popup with chosen target/features so the user knows what will be used
        """
        self.features, self.target = auto_choose_features_and_target(self.df)
        self.preview_text.delete(1.0, tk.END)
        try:
            self.preview_text.insert(1.0, self.df.head(10).to_string(index=False))
        except Exception:
            # fallback if DataFrame contains objects that .to_string can't handle directly
            self.preview_text.insert(1.0, repr(self.df.head(10)))
        choices = f"Auto-chosen target: {self.target}\nAuto-chosen features: {', '.join(self.features)}"
        messagebox.showinfo("Auto selection", choices)

    # -----------------------------
    # Model training (supervised learning)
    # -----------------------------
    def train_model(self):
        """
        Train a RandomForestRegressor using the selected features and target.
        This function:
        - prepares X (features) and y (target)
        - encodes non-numeric features by factorizing
        - splits data into train and validation (80/20 simple split)
        - trains RandomForest and computes MAE and R² on validation
        - shows metrics and a scatter plot (true vs predicted)

        This is intentionally simple for non-technical users.
        """
        if self.df is None:
            messagebox.showwarning("No data", "Load or generate data first (button 1 or 2).")
            return
        if not self.features or not self.target:
            messagebox.showwarning("No selection", "No features/target chosen.")
            return
        try:
            # NEW --- add time-series engineered features
            df_ml = add_time_features(self.df)

            # reselect features after feature engineering
            self.features, self.target = auto_choose_features_and_target(df_ml)

            X = df_ml[self.features].copy()

            # encode non-numeric columns
            for c in X.columns:
                if X[c].dtype == 'O':
                    X[c], _ = pd.factorize(X[c])

            y = df_ml[self.target].values
            # simple split: 80% train, 20% validation (no random shuffle to keep time-order)
            n = len(X)
            split = int(n * 0.8)
            X_train, X_val = X.iloc[:split], X.iloc[split:]
            y_train, y_val = y[:split], y[split:]
            # create and train RandomForest (an ensemble model from sklearn)
            model = RandomForestRegressor(n_estimators=150, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            # compute simple metrics to show to user
            mae = mean_absolute_error(y_val, preds)
            r2 = r2_score(y_val, preds)
            self.model = model
            self.mae_label.config(text=f"MAE: {mae:.3f}")
            self.r2_label.config(text=f"R²: {r2:.3f}")
            # plot validation true vs predicted
            self._plot_true_vs_pred(y_val, preds)
            messagebox.showinfo("Trained", "Model trained successfully. See metrics and plot.")
        except Exception as e:
            messagebox.showerror("Train error", f"Training failed: {e}")

    def _plot_true_vs_pred(self, y_true, y_pred):
        """
        Draw a scatter plot: true vs predicted values on validation set.
        A red dashed line y=x is drawn to help visually evaluate model accuracy.
        """
        # clear any previous plot widgets
        for child in self.plot_container.winfo_children():
            child.destroy()
        fig, ax = plt.subplots(figsize=(7,4))
        ax.scatter(y_true, y_pred, alpha=0.6)
        # draw y=x reference line
        mn = min(float(np.min(y_true)), float(np.min(y_pred)))
        mx = max(float(np.max(y_true)), float(np.max(y_pred)))
        ax.plot([mn,mx],[mn,mx],'r--',linewidth=1)
        ax.set_xlabel("True disease cases")
        ax.set_ylabel("Predicted disease cases")
        ax.set_title("Validation: True vs Predicted")
        canvas = FigureCanvasTkAgg(fig, master=self.plot_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        # remember last figure in case user wants to save it later
        self.last_fig = fig

    # -----------------------------
    # Simulation window (reduce mosquito_count by slider and re-predict)
    # -----------------------------
    def open_simulation_window(self):
        """
        Open a small popup window where the user can move a slider to choose GM reduction %.
        The function will:
        - re-encode features the same way as training
        - predict baseline and post-GM disease cases
        - show a short plain-language summary and a small time-series plot (first 200 samples)
        - allow exporting results as CSV
        """
        if self.model is None:
            messagebox.showwarning("No model", "Train a model first (button 3).")
            return
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
            """
            Core simulation logic executed when user clicks "Run Simulation".
            Steps:
            1) read slider and threshold
            2) encode features the same way as training (factorize strings)
            3) compute base_pred and gm_pred
            4) compute means, percent difference and min mosquito count after GM
            5) produce recommendation and plot a small chart
            """
            try:
                gm_pct = float(gm_var.get())
                threshold = float(thresh_entry.get())
                # VERY IMPORTANT — use same feature engineering as training
                df_ml = add_time_features(self.df)
                X_raw = df_ml[self.features].copy()
                # encoding must match training encoding; here we used factorize again (simple approach)
                for c in X_raw.columns:
                    if X_raw[c].dtype == 'O':
                        X_raw[c], _ = pd.factorize(X_raw[c])
                # baseline predictions
                base_pred = self.model.predict(X_raw)
                # ensure mosquito_count exists in features
                if 'mosquito_count' not in X_raw.columns:
                    messagebox.showwarning("Missing column", "Feature 'mosquito_count' not found in features.")
                    return
                # simulate GM effect by reducing mosquito_count value
                X_gm = X_raw.copy()
                X_gm['mosquito_count'] = pd.to_numeric(X_gm['mosquito_count'], errors='coerce').fillna(0)
                # NEW --- gradual GM simulation
                X_gm['mosquito_count'] = simulate_gm_population(
                    X_gm['mosquito_count'], gm_pct
                )
                gm_pred = self.model.predict(X_gm)
                # compute summary statistics
                base_mean = float(np.mean(base_pred)); gm_mean = float(np.mean(gm_pred))
                diff = gm_mean - base_mean
                diff_pct = (diff/base_mean*100) if base_mean!=0 else 0.0
                min_mos = float(X_gm['mosquito_count'].min())
                rec = simple_recommendation(base_mean, gm_mean, min_mos, threshold)
                score, level = calculate_risk_score(base_mean, gm_mean, min_mos, threshold)
                summary = (
                        f"GM Reduction: {gm_pct:.0f}%\n"
                        f"Average disease (without GM): {base_mean:.2f}\n"
                        f"Average disease (with GM): {gm_mean:.2f}\n"
                        f"Change: {diff:.2f} ({diff_pct:.2f}%)\n"
                        f"Minimum mosquito count after GM: {min_mos:.2f}\n"
                        f"Threshold: {threshold:.2f}\n\n"
                        f"Risk Score: {score:.2f}\n"
                        f"Decision Level: {level}\n\n"
                         f"Recommendation:\n{rec}\n"
                )
                result_text.delete(1.0, tk.END); result_text.insert(1.0, summary)
                # store last sim results for export
                self.last_sim_df = pd.DataFrame({'without_gm': base_pred, 'with_gm': gm_pred})
                # small plot of first 200 samples to keep plot readable
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
        """
        Export the last simulation stored in self.last_sim_df to CSV using a save dialog.
        """
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
        """
        Save the trained sklearn model using joblib so users can reuse it later.
        """
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


# -----------------------------
# Program entry point
# -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = SuperSimpleApp(root)
    root.mainloop()

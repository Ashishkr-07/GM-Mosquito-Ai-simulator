#!/usr/bin/env python3
"""
aiml.py - GM Mosquito AI Impact Study GUI
Save as aiml.py and run with your project's venv Python, e.g.:
"/Users/abhishek/Desktop/Python Project/venv/bin/python" "/Users/abhishek/Desktop/Python Project/aiml.py"
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Text
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta
import joblib
import os

@dataclass
class AppState:
    df: Optional[pd.DataFrame] = None
    features: list = field(default_factory=list)
    target: Optional[str] = None
    model: Optional[RandomForestRegressor] = None
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
    # mapping to ensure consistent categorical encoding between train and simulate
    cat_mappings: Dict[str, Any] = field(default_factory=dict)
    model_path: Optional[str] = None
    last_sim_df: Optional[pd.DataFrame] = None

class GMMosquitoAIGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GM Mosquito AI Impact Study")
        self.root.geometry("1000x700")
        self.state = AppState()
        
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.create_data_tab()
        self.create_train_tab()
        self.create_simulate_tab()
        self.create_report_tab()
    
    def create_data_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text='DATA')
        
        # Buttons frame
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Load CSV...", command=self.load_csv).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Generate Demo Data", command=self.generate_demo_data).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Load Model...", command=self.load_model).pack(side='left', padx=5)
        
        # Selection frame
        select_frame = ttk.Frame(tab)
        select_frame.pack(pady=10, fill='x', padx=20)
        
        # Target column
        ttk.Label(select_frame, text="Select Target Column:").grid(row=0, column=0, sticky='w', pady=5)
        self.target_combo = ttk.Combobox(select_frame, state='readonly', width=30)
        self.target_combo.grid(row=0, column=1, sticky='w', pady=5, padx=10)
        
        # Features
        ttk.Label(select_frame, text="Select Features (multi-select):").grid(row=1, column=0, sticky='nw', pady=5)
        features_frame = ttk.Frame(select_frame)
        features_frame.grid(row=1, column=1, sticky='w', pady=5, padx=10)
        
        self.features_listbox = tk.Listbox(features_frame, selectmode='multiple', height=6, width=40)
        self.features_listbox.pack(side='left')
        scrollbar = ttk.Scrollbar(features_frame, orient='vertical', command=self.features_listbox.yview)
        scrollbar.pack(side='left', fill='y')
        self.features_listbox.config(yscrollcommand=scrollbar.set)
        
        # Test split slider
        ttk.Label(select_frame, text="Test Split:").grid(row=2, column=0, sticky='w', pady=5)
        slider_frame = ttk.Frame(select_frame)
        slider_frame.grid(row=2, column=1, sticky='w', pady=5, padx=10)
        
        self.test_split_var = tk.DoubleVar(value=0.2)
        self.test_split_slider = ttk.Scale(slider_frame, from_=0.1, to=0.4, orient='horizontal',
                                           variable=self.test_split_var, length=200)
        self.test_split_slider.pack(side='left')
        self.test_split_label = ttk.Label(slider_frame, text="0.20")
        self.test_split_label.pack(side='left', padx=10)
        # modern Tcl/Tk usage:
        self.test_split_var.trace_add('write', lambda *args: self.update_split_label())
        
        # Apply button
        ttk.Button(select_frame, text="Apply selection", command=self.apply_selection).grid(row=3, column=1, sticky='w', pady=10, padx=10)
        
        # Preview
        ttk.Label(tab, text="Data Preview (top 200 rows):").pack(pady=5)
        preview_frame = ttk.Frame(tab)
        preview_frame.pack(fill='both', expand=True, padx=20, pady=5)
        
        self.preview_text = Text(preview_frame, height=15, width=100)
        self.preview_text.pack(side='left', fill='both', expand=True)
        preview_scroll = ttk.Scrollbar(preview_frame, orient='vertical', command=self.preview_text.yview)
        preview_scroll.pack(side='left', fill='y')
        self.preview_text.config(yscrollcommand=preview_scroll.set)
    
    def create_train_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text='TRAIN')
        
        ttk.Button(tab, text="Train model", command=self.train_model).pack(pady=20)
        
        # Metrics frame
        metrics_frame = ttk.LabelFrame(tab, text="Validation Metrics")
        metrics_frame.pack(pady=10, padx=20, fill='x')
        
        self.mae_label = ttk.Label(metrics_frame, text="MAE: -", font=('Arial', 12))
        self.mae_label.pack(pady=5)
        self.r2_label = ttk.Label(metrics_frame, text="R²: -", font=('Arial', 12))
        self.r2_label.pack(pady=5)
        
        # Plot frame
        self.train_plot_frame = ttk.Frame(tab)
        self.train_plot_frame.pack(fill='both', expand=True, padx=20, pady=10)
    
    def create_simulate_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text='SIMULATE')
        
        # Controls
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
        self.threshold_entry.insert(0, "100")
        self.threshold_entry.grid(row=1, column=1, sticky='w', pady=5, padx=10)
        
        ttk.Button(control_frame, text="Run simulation", command=self.run_simulation).grid(row=2, column=1, sticky='w', pady=10, padx=10)
        ttk.Button(control_frame, text="Export simulation CSV", command=self.export_simulation_csv).grid(row=2, column=0, sticky='w', pady=10, padx=10)
        
        # Results
        self.sim_plot_frame = ttk.Frame(tab)
        self.sim_plot_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        summary_frame = ttk.LabelFrame(tab, text="Simulation Summary")
        summary_frame.pack(pady=10, padx=20, fill='x')
        
        self.sim_summary_text = Text(summary_frame, height=8, width=80)
        self.sim_summary_text.pack(pady=5, padx=5)
    
    def create_report_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text='REPORT')
        
        ttk.Label(tab, text="Notes:").pack(pady=10)
        
        self.notes_text = Text(tab, height=15, width=80)
        self.notes_text.pack(padx=20, pady=5)
        
        ttk.Button(tab, text="Save summary report", command=self.save_report).pack(pady=20)
    
    def update_split_label(self, *args):
        self.test_split_label.config(text=f"{self.test_split_var.get():.2f}")
    
    def update_gm_label(self, *args):
        self.gm_slider_label.config(text=f"{self.gm_reduction_var.get():.0f}%")
    
    def load_csv(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filename:
            try:
                self.state.df = pd.read_csv(filename)
                self.update_data_view()
                messagebox.showinfo("Success", "CSV loaded successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV: {str(e)}")
    
    def generate_demo_data(self):
        np.random.seed(42)
        n = 365
        
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n)]
        
        regions = np.random.choice(['North', 'South', 'East', 'West'], n)
        
        # Seasonal patterns
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
            'region': regions,
            'mosquito_count': mosquito_count,
            'avg_temp': avg_temp,
            'rainfall': rainfall,
            'humidity': humidity,
            'past_cases': past_cases,
            'disease_cases': disease_cases
        })
        
        self.update_data_view()
        messagebox.showinfo("Success", "Demo data generated successfully")
    
    def update_data_view(self):
        if self.state.df is not None:
            self.preview_text.delete(1.0, tk.END)
            try:
                self.preview_text.insert(1.0, self.state.df.head(200).to_string())
            except Exception:
                # large or complex DF; fallback to simple repr
                self.preview_text.insert(1.0, repr(self.state.df.head(200)))
            
            columns = list(self.state.df.columns)
            self.target_combo['values'] = columns
            
            self.features_listbox.delete(0, tk.END)
            for col in columns:
                self.features_listbox.insert(tk.END, col)
    
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
        
        messagebox.showinfo("Success", f"Target: {target}\nFeatures: {', '.join(features)}")
    
    def _encode_features_for_model(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Encode categorical columns consistently. If fit=True, record the mappings.
        Returns transformed X (numeric).
        """
        X_enc = X.copy()
        cat_cols = X_enc.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in cat_cols:
            if fit:
                # create mapping from category -> code, and store categories
                cat = pd.Categorical(X_enc[col])
                categories = list(cat.categories)
                mapping = {cat_val: code for code, cat_val in enumerate(categories)}
                self.state.cat_mappings[col] = {'mapping': mapping, 'categories': categories}
                X_enc[col] = X_enc[col].map(mapping).fillna(-1).astype(int)
            else:
                # use existing mapping if available
                info = self.state.cat_mappings.get(col)
                if info is not None:
                    mapping = info['mapping']
                    X_enc[col] = X_enc[col].map(mapping).fillna(-1).astype(int)
                else:
                    # fallback to ad-hoc coding (but record it if fit is True later)
                    X_enc[col] = pd.Categorical(X_enc[col]).codes
        return X_enc
    
    def train_model(self):
        if self.state.df is None or not self.state.features or not self.state.target:
            messagebox.showwarning("Warning", "Please load data and apply selection first")
            return
        
        try:
            X = self.state.df[self.state.features].copy()
            y = self.state.df[self.state.target].copy()
            
            # Encode categoricals and record mapping
            X_enc = self._encode_features_for_model(X, fit=True)
            
            test_size = float(self.test_split_var.get())
            X_train, X_val, y_train, y_val = train_test_split(X_enc, y, test_size=test_size, random_state=42)
            
            model = RandomForestRegressor(n_estimators=300, random_state=42)
            model.fit(X_train, y_train)
            
            predictions = model.predict(X_val)
            
            mae = mean_absolute_error(y_val, predictions)
            r2 = r2_score(y_val, predictions)
            
            self.state.model = model
            self.state.X_train = X_train
            self.state.X_val = X_val
            self.state.y_train = y_train
            self.state.y_val = y_val
            self.state.predictions = predictions
            self.state.mae = mae
            self.state.r2 = r2
            
            self.mae_label.config(text=f"MAE: {mae:.4f}")
            self.r2_label.config(text=f"R²: {r2:.4f}")
            
            self.plot_train_results(y_val, predictions)
            
            # save model automatically to a default path for convenience
            model_path = filedialog.asksaveasfilename(title="Save trained model as...", defaultextension=".joblib",
                                                      filetypes=[("Joblib", "*.joblib"), ("All files", "*.*")])
            if model_path:
                joblib.dump({'model': model, 'cat_mappings': self.state.cat_mappings}, model_path)
                self.state.model_path = model_path
                messagebox.showinfo("Success", f"Model trained and saved to:\n{model_path}")
            else:
                messagebox.showinfo("Success", "Model trained (not saved).")
            
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
    
    def plot_train_results(self, y_true, y_pred):
        for widget in self.train_plot_frame.winfo_children():
            widget.destroy()
        
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_true, y_pred, alpha=0.5)
        
        min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
        max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x')
        
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('True vs Predicted')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.state.last_figure = fig
        
        canvas = FigureCanvasTkAgg(fig, self.train_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
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
            X_sim = self._encode_features_for_model(X_sim_raw, fit=False)
            
            # Base prediction
            base_pred = self.state.model.predict(X_sim)
            
            # GM prediction - adjust mosquito_count by name
            if 'mosquito_count' not in X_sim.columns:
                messagebox.showerror("Error", "Feature mosquito_count not present after encoding")
                return
            
            X_gm = X_sim.copy()
            # ensure mosquito_count is numeric
            X_gm['mosquito_count'] = pd.to_numeric(X_gm['mosquito_count'], errors='coerce').fillna(0)
            X_gm['mosquito_count'] = X_gm['mosquito_count'] * (1 - gm_pct / 100.0)
            X_gm['mosquito_count'] = np.maximum(1, X_gm['mosquito_count'])
            
            gm_pred = self.state.model.predict(X_gm)
            
            # Ecological risk
            min_mosquito = float(X_gm['mosquito_count'].min())
            ecological_risk = "HIGH" if min_mosquito < threshold else "OK"
            
            # Summary
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
            
            # keep last sim dataframe for export
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
    
    def load_model(self):
        path = filedialog.askopenfilename(title="Load model (joblib)", filetypes=[("Joblib", "*.joblib"), ("All files", "*.*")])
        if not path:
            return
        try:
            data = joblib.load(path)
            if isinstance(data, dict) and 'model' in data:
                self.state.model = data['model']
                self.state.cat_mappings = data.get('cat_mappings', {})
                self.state.model_path = path
                messagebox.showinfo("Success", f"Model loaded from {path}")
            else:
                # maybe they saved just the sklearn model
                self.state.model = data
                messagebox.showinfo("Success", f"Model loaded from {path} (no cat mappings found)")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
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
            
            # ask user where to save
            fname = filedialog.asksaveasfilename(title="Save report", defaultextension=".txt",
                                                 filetypes=[("Text file", "*.txt"), ("All files", "*.*")])
            if not fname:
                return
            
            with open(fname, 'w') as f:
                f.write(report)
            
            # save last plot if exists
            if self.state.last_figure is not None:
                plot_path = os.path.splitext(fname)[0] + "_plot.png"
                try:
                    self.state.last_figure.savefig(plot_path, dpi=150, bbox_inches='tight')
                except Exception:
                    # fallback: try saving current plt
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                messagebox.showinfo("Success", f"Report saved as {fname}\nPlot saved as {plot_path}")
            else:
                messagebox.showinfo("Success", f"Report saved as {fname}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save report: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = GMMosquitoAIGUI(root)
    root.mainloop()
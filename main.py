import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
import joblib
import os
import json
import csv
from datetime import datetime

# ====================== THEME ======================

BG        = "#f5f3ee"
CARD_BG   = "#ffffff"
BORDER    = "#d6d0c4"
TEXT_PRI  = "#1a1714"
TEXT_SEC  = "#6b6558"
TEXT_HINT = "#9c9488"
SURFACE   = "#ede9e1"

CAT_COLORS = {
    "Excellent":  {"bar": "#3b6d11", "badge_bg": "#eaf3de", "badge_fg": "#27500a"},
    "Good":       {"bar": "#185FA5", "badge_bg": "#E6F1FB", "badge_fg": "#0C447C"},
    "Average":    {"bar": "#854F0B", "badge_bg": "#FAEEDA", "badge_fg": "#633806"},
    "Needs Work": {"bar": "#A32D2D", "badge_bg": "#FCEBEB", "badge_fg": "#791F1F"},
}

def get_category(score):
    if score >= 85: return "Excellent"
    if score >= 70: return "Good"
    if score >= 50: return "Average"
    return "Needs Work"


# ====================== MODEL ======================

FEATURES = ['study_hours', 'attendance', 'previous_score', 'sleep_hours',
            'extracurricular_enc', 'screen_time', 'study_group_enc']

def train_and_save():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd
    np.random.seed(42)
    n = 2000
    df = pd.DataFrame({
        'study_hours':    np.random.uniform(1, 12, n),
        'attendance':     np.random.uniform(50, 100, n),
        'previous_score': np.random.uniform(30, 100, n),
        'sleep_hours':    np.random.uniform(4, 10, n),
        'extracurricular': np.random.choice(['Yes','No'], n),
        'screen_time':    np.random.uniform(0, 8, n),
        'study_group':    np.random.choice(['Yes','No'], n),
    })
    le  = LabelEncoder(); df['extracurricular_enc'] = le.fit_transform(df['extracurricular'])
    le2 = LabelEncoder(); df['study_group_enc']     = le2.fit_transform(df['study_group'])
    df['final_score'] = (
        df['study_hours']*4.2 + df['attendance']*0.38 + df['previous_score']*0.34
        + df['sleep_hours']*1.4 + df['extracurricular_enc']*3.5
        + df['study_group_enc']*4.0 - df['screen_time']*1.8
        + np.random.normal(0, 4, n)
    ).clip(0, 100)
    m = LinearRegression()
    m.fit(df[FEATURES], df['final_score'])
    joblib.dump(m, 'student_model.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    joblib.dump(le2, 'label_encoder2.pkl')

def load_model():
    if not all(os.path.exists(f) for f in ['student_model.pkl','label_encoder.pkl','label_encoder2.pkl']):
        train_and_save()
    return joblib.load('student_model.pkl'), joblib.load('label_encoder.pkl'), joblib.load('label_encoder2.pkl')

model, le, le2 = load_model()

HISTORY_FILE = 'prediction_history.json'
def load_history():
    try:
        if os.path.exists(HISTORY_FILE):
            return json.load(open(HISTORY_FILE))
    except Exception: pass
    return []
def save_history(h): json.dump(h, open(HISTORY_FILE,'w'), indent=2)


# ====================== REUSABLE WIDGETS ======================

def lbl(parent, text, size=13, weight="normal", color=TEXT_PRI, **kw):
    return ctk.CTkLabel(parent, text=text, font=ctk.CTkFont(size=size, weight=weight),
                        text_color=color, **kw)

def section_lbl(parent, text):
    return lbl(parent, text, size=11, weight="bold", color=TEXT_HINT)

def card(parent, **kw):
    return ctk.CTkFrame(parent, fg_color=CARD_BG, border_color=BORDER,
                        border_width=1, corner_radius=12, **kw)


class ToggleGroup(ctk.CTkFrame):
    def __init__(self, parent, options, default):
        super().__init__(parent, fg_color=CARD_BG, border_color=BORDER,
                        border_width=1, corner_radius=8)
        self._var = ctk.StringVar(value=default)
        self._btns = {}
        for i, opt in enumerate(options):
            b = ctk.CTkButton(
                self, text=opt, width=90, height=34,
                fg_color=SURFACE if opt == default else CARD_BG,
                text_color=TEXT_PRI, hover_color=SURFACE,
                border_width=0, corner_radius=6,
                font=ctk.CTkFont(size=13),
                command=lambda o=opt: self._select(o)
            )
            b.grid(row=0, column=i, sticky="ew", padx=2, pady=2)
            self._btns[opt] = b
        self.columnconfigure(list(range(len(options))), weight=1)

    def _select(self, val):
        self._var.set(val)
        for k, b in self._btns.items():
            b.configure(fg_color=SURFACE if k == val else CARD_BG)

    def get(self): return self._var.get()


class SliderRow(ctk.CTkFrame):
    def __init__(self, parent, label_text, lo, hi, default, unit=""):
        super().__init__(parent, fg_color="transparent")
        self._unit = unit
        self._var  = ctk.DoubleVar(value=default)
        lbl(self, label_text, size=13, color=TEXT_PRI, width=160, anchor="w").pack(side="left")
        self._val = lbl(self, self._fmt(default), size=13, color=TEXT_PRI, width=50, anchor="e")
        self._val.pack(side="right")
        s = ctk.CTkSlider(
            self, from_=lo, to=hi, variable=self._var,
            button_color=CARD_BG, button_hover_color=SURFACE,
            progress_color="#c2bdb3", fg_color=SURFACE,
            command=self._update
        )
        s.pack(side="left", fill="x", expand=True, padx=8)

    def _fmt(self, v):
        v = float(v)
        if self._unit == "h":  return f"{v:.1f}h"
        if self._unit == "%":  return f"{v:.0f}%"
        if self._unit == "yr": return f"Year {int(v)}"
        return f"{v:.0f}"

    def _update(self, v): self._val.configure(text=self._fmt(v))
    def get(self): return self._var.get()


# ====================== MAIN APP ======================

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Student Performance Predictor")
        self.geometry("680x860")
        self.minsize(600, 700)
        self.configure(fg_color=BG)
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        self._history = load_history()
        self._build()

    # ── Shell ────────────────────────────────────────────────────────────────

    def _build(self):
        self._scroll = ctk.CTkScrollableFrame(self, fg_color=BG,
                                            scrollbar_button_color=BORDER,
                                            scrollbar_button_hover_color=BORDER)
        self._scroll.pack(fill="both", expand=True, padx=20, pady=16)

        # Tab bar
        tf = ctk.CTkFrame(self._scroll, fg_color=BG, border_color=BORDER,
                        border_width=1, corner_radius=10)
        tf.pack(fill="x", pady=(0, 14))
        tf.columnconfigure((0, 1, 2), weight=1)
        self._tab_btns = {}
        for i, name in enumerate(("Predict", "Analysis", "History")):
            b = ctk.CTkButton(
                tf, text=name, height=42, corner_radius=8,
                fg_color=CARD_BG if i == 0 else "transparent",
                text_color=TEXT_PRI if i == 0 else TEXT_SEC,
                hover_color=CARD_BG, border_width=0,
                font=ctk.CTkFont(size=14, weight="bold" if i == 0 else "normal"),
                command=lambda n=name: self._switch(n)
            )
            b.grid(row=0, column=i, sticky="ew", padx=3, pady=3)
            self._tab_btns[name] = b

        self._panels = {}
        for name in ("Predict", "Analysis", "History"):
            f = ctk.CTkFrame(self._scroll, fg_color="transparent")
            self._panels[name] = f
        self._panels["Predict"].pack(fill="x")
        self._build_predict(self._panels["Predict"])
        self._build_analysis(self._panels["Analysis"])
        self._build_history_panel(self._panels["History"])

    def _switch(self, name):
        for k, f in self._panels.items():
            f.pack(fill="x") if k == name else f.pack_forget()
        for k, b in self._tab_btns.items():
            active = k == name
            b.configure(
                fg_color=CARD_BG if active else "transparent",
                text_color=TEXT_PRI if active else TEXT_SEC,
                font=ctk.CTkFont(size=14, weight="bold" if active else "normal")
            )

    # ── Predict panel ────────────────────────────────────────────────────────

    def _build_predict(self, p):
        # Study habits
        c1 = card(p); c1.pack(fill="x", pady=(0, 10))
        i1 = ctk.CTkFrame(c1, fg_color="transparent"); i1.pack(fill="x", padx=18, pady=16)
        section_lbl(i1, "STUDY HABITS").pack(anchor="w", pady=(0, 10))
        self._study  = SliderRow(i1, "Study hours/day",   0.0, 12.0,  5.0, "h"); self._study.pack(fill="x", pady=5)
        self._att    = SliderRow(i1, "Attendance",         50,  100,   75,  "%"); self._att.pack(fill="x", pady=5)
        self._sleep  = SliderRow(i1, "Sleep hours/night", 4.0, 10.0,  7.0, "h"); self._sleep.pack(fill="x", pady=5)

        # Academic background
        c2 = card(p); c2.pack(fill="x", pady=(0, 10))
        i2 = ctk.CTkFrame(c2, fg_color="transparent"); i2.pack(fill="x", padx=18, pady=16)
        section_lbl(i2, "ACADEMIC BACKGROUND").pack(anchor="w", pady=(0, 10))
        self._prev  = SliderRow(i2, "Previous score", 0,  100,  65, "");   self._prev.pack(fill="x", pady=5)
        self._grade = SliderRow(i2, "Grade level",    1,    4,   2, "yr"); self._grade.pack(fill="x", pady=5)

        # Lifestyle
        c3 = card(p); c3.pack(fill="x", pady=(0, 10))
        i3 = ctk.CTkFrame(c3, fg_color="transparent"); i3.pack(fill="x", padx=18, pady=16)
        section_lbl(i3, "LIFESTYLE").pack(anchor="w", pady=(0, 10))

        re = ctk.CTkFrame(i3, fg_color="transparent"); re.pack(fill="x", pady=5)
        lbl(re, "Extracurricular", size=13, color=TEXT_PRI, width=160, anchor="w").pack(side="left")
        self._extra = ToggleGroup(re, ["Yes", "No"], "Yes"); self._extra.pack(side="left", fill="x", expand=True)

        self._screen = SliderRow(i3, "Screen time/day", 0.0, 10.0, 3.0, "h"); self._screen.pack(fill="x", pady=5)

        rg = ctk.CTkFrame(i3, fg_color="transparent"); rg.pack(fill="x", pady=5)
        lbl(rg, "Study group", size=13, color=TEXT_PRI, width=160, anchor="w").pack(side="left")
        self._group = ToggleGroup(rg, ["Yes", "No"], "No"); self._group.pack(side="left", fill="x", expand=True)

        # Predict button
        ctk.CTkButton(
            p, text="⊕  Calculate predicted score",
            font=ctk.CTkFont(size=14, weight="bold"), height=48, corner_radius=10,
            fg_color=CARD_BG, text_color=TEXT_PRI, hover_color=SURFACE,
            border_color=BORDER, border_width=1, command=self._predict
        ).pack(fill="x", pady=(0, 10))

        # Result card (hidden until first prediction)
        self._res_card = card(p)
        self._score_lbl = lbl(self._res_card, "—", size=44, weight="bold", color=TEXT_PRI)
        self._score_lbl.pack(anchor="w", padx=20, pady=(18, 2))
        self._cat_lbl   = lbl(self._res_card, "", size=13, color=TEXT_HINT)
        self._cat_lbl.pack(anchor="w", padx=20)
        self._prog = ctk.CTkProgressBar(self._res_card, height=8, corner_radius=4,
                                        fg_color=SURFACE, progress_color="#185FA5")
        self._prog.set(0); self._prog.pack(fill="x", padx=20, pady=10)

        mf = ctk.CTkFrame(self._res_card, fg_color="transparent")
        mf.pack(fill="x", padx=20, pady=(0, 10))
        mf.columnconfigure((0, 1), weight=1)
        self._conf_lbl = self._metric_card(mf, "Confidence", 0)
        self._pct_lbl  = self._metric_card(mf, "Percentile (est.)", 1)

        section_lbl(self._res_card, "RECOMMENDATIONS").pack(anchor="w", padx=20, pady=(6, 4))
        self._rec_frame = ctk.CTkFrame(self._res_card, fg_color="transparent")
        self._rec_frame.pack(fill="x", padx=20, pady=(0, 16))

    def _metric_card(self, parent, title, col):
        f = ctk.CTkFrame(parent, fg_color=SURFACE, corner_radius=8, border_width=0)
        f.grid(row=0, column=col, sticky="ew", padx=4)
        lbl(f, title, size=11, color=TEXT_HINT).pack(pady=(10, 2))
        v = lbl(f, "—", size=20, weight="bold", color=TEXT_PRI)
        v.pack(pady=(0, 10))
        return v

    # ── Analysis panel ───────────────────────────────────────────────────────

    def _build_analysis(self, p):
        lbl(p, "Factor contribution & sensitivity analysis",
            size=13, color=TEXT_HINT).pack(pady=(8, 6))
        self._fc_card   = card(p); self._fc_card.pack(fill="x", pady=(0, 10))
        self._sens_card = card(p); self._sens_card.pack(fill="x")
        lbl(self._fc_card,   "Run a prediction first.", size=13, color=TEXT_HINT).pack(pady=30)
        lbl(self._sens_card, "Run a prediction first.", size=13, color=TEXT_HINT).pack(pady=30)

    def _fill_analysis(self, inp, score, bar_color):
        for f in (self._fc_card, self._sens_card):
            for w in f.winfo_children(): w.destroy()

        section_lbl(self._fc_card, "FACTOR CONTRIBUTION (%)").pack(anchor="w", padx=18, pady=(14, 8))
        pairs = [("Study hours",inp['study']*4.2),("Attendance",inp['att']*0.38),
                ("Prev. score",inp['prev']*0.34),("Sleep",inp['sleep']*1.4),
                ("Extracurricular",inp['extra']*3.5),("Study group",inp['group']*4.0)]
        total = sum(v for _,v in pairs) or 1
        for name, contrib in pairs:
            self._bar_row(self._fc_card, name, contrib / total * 100, bar_color)
        ctk.CTkFrame(self._fc_card, height=8, fg_color="transparent").pack()

        section_lbl(self._sens_card, "SCORE SENSITIVITY").pack(anchor="w", padx=18, pady=(14, 8))
        for lbl_text, mod in [
            ("+2h study",        {**inp, 'study':  min(12, inp['study'] + 2)}),
            ("+10% attendance",  {**inp, 'att':    min(100,inp['att']   + 10)}),
            ("+1h sleep",        {**inp, 'sleep':  min(12, inp['sleep'] + 1)}),
            ("-2h screen",       {**inp, 'screen': max(0,  inp['screen']- 2)}),
            ("Join study group", {**inp, 'group':  1}),
        ]:
            d = round(self._compute(mod) - score, 1)
            c = "#3b6d11" if d > 0 else "#A32D2D" if d < 0 else TEXT_HINT
            sign = "+" if d > 0 else ""
            self._bar_row(self._sens_card, lbl_text, abs(d) * 4, c, f"{sign}{d} pts", c)
        ctk.CTkFrame(self._sens_card, height=8, fg_color="transparent").pack()

    def _bar_row(self, parent, name, pct, bar_color, suffix=None, val_color=None):
        row = ctk.CTkFrame(parent, fg_color="transparent"); row.pack(fill="x", padx=18, pady=3)
        lbl(row, name, size=12, color=TEXT_SEC, width=130, anchor="w").pack(side="left")
        bg = ctk.CTkFrame(row, height=5, corner_radius=3, fg_color=SURFACE)
        bg.pack(side="left", fill="x", expand=True, padx=8)
        bg.update_idletasks()
        w = max(6, int(min(pct, 100) * bg.winfo_width() / 100)) if bg.winfo_width() > 10 else max(6, int(min(pct,100)*2))
        ctk.CTkFrame(bg, height=5, width=w, corner_radius=3, fg_color=bar_color).place(x=0, y=0)
        lbl(row, suffix or f"{pct:.0f}%", size=12, color=val_color or TEXT_HINT,
            width=56, anchor="e").pack(side="right")

    # ── History panel ────────────────────────────────────────────────────────

    def _build_history_panel(self, p):
        top = ctk.CTkFrame(p, fg_color="transparent"); top.pack(fill="x", pady=(0, 8))
        lbl(top, "Prediction history", size=14, weight="bold", color=TEXT_PRI).pack(side="left")
        for text, cmd in (("Export CSV", self._export), ("Clear", self._clear)):
            ctk.CTkButton(top, text=text, width=90, height=30,
                        fg_color=CARD_BG, text_color=TEXT_SEC, border_color=BORDER,
                        border_width=1, hover_color=SURFACE, corner_radius=8,
                        font=ctk.CTkFont(size=12), command=cmd).pack(side="right", padx=3)
        self._hist_card = card(p); self._hist_card.pack(fill="x")
        self._render_history()

    def _render_history(self):
        for w in self._hist_card.winfo_children(): w.destroy()
        if not self._history:
            lbl(self._hist_card, "No predictions recorded yet.", size=13, color=TEXT_HINT).pack(pady=30)
            return
        for i, e in enumerate(reversed(self._history)):
            row = ctk.CTkFrame(self._hist_card,
                            fg_color=SURFACE if i % 2 else CARD_BG,
                            corner_radius=0)
            row.pack(fill="x")
            inner = ctk.CTkFrame(row, fg_color="transparent"); inner.pack(fill="x", padx=16, pady=9)
            lbl(inner, f"#{len(self._history)-i}", size=12, color=TEXT_HINT, width=30, anchor="w").pack(side="left")
            lbl(inner, f"{e['score']:.1f} / 100", size=13, weight="bold", color=TEXT_PRI, width=80).pack(side="left")
            c = CAT_COLORS.get(e['category'], CAT_COLORS["Good"])
            ctk.CTkLabel(inner, text=e['category'],
                        fg_color=c['badge_bg'], text_color=c['badge_fg'],
                        font=ctk.CTkFont(size=11), corner_radius=99,
                        padx=8, pady=2).pack(side="left", padx=6)
            lbl(inner, e['timestamp'], size=11, color=TEXT_HINT).pack(side="right")

    # ── Core prediction ──────────────────────────────────────────────────────

    def _get_inputs(self):
        return dict(study=self._study.get(), att=self._att.get(), sleep=self._sleep.get(),
                    prev=self._prev.get(), screen=self._screen.get(),
                    extra=1 if self._extra.get()=="Yes" else 0,
                    group=1 if self._group.get()=="Yes" else 0)

    def _compute(self, inp):
        X = np.array([[inp['study'],inp['att'],inp['prev'],inp['sleep'],
                       inp['extra'],inp['screen'],inp['group']]])
        return float(np.clip(model.predict(X)[0], 0, 100))

    def _predict(self):
        try:
            inp   = self._get_inputs()
            score = round(self._compute(inp), 1)
            cat   = get_category(score)
            col   = CAT_COLORS[cat]

            self._res_card.pack(fill="x", pady=(0, 10))
            self._score_lbl.configure(text=f"{score:.1f} / 100", text_color=col['bar'])
            self._cat_lbl.configure(text=f"{cat} performance")
            self._prog.configure(progress_color=col['bar'])
            self._prog.set(score / 100)
            self._conf_lbl.configure(text=f"{min(95, 68+score*0.22):.0f}%")
            self._pct_lbl.configure(text=f"{min(99,max(1,int(10+score*0.82)))}th")

            for w in self._rec_frame.winfo_children(): w.destroy()
            for rec in self._build_recs(inp, score):
                r = ctk.CTkFrame(self._rec_frame, fg_color="transparent"); r.pack(fill="x", pady=2)
                ctk.CTkLabel(r, text="•", text_color=rec['color'],
                            font=ctk.CTkFont(size=16), width=14).pack(side="left", anchor="n")
                ctk.CTkLabel(r, text=rec['text'], font=ctk.CTkFont(size=12),
                            text_color=TEXT_PRI, wraplength=400,
                            anchor="w", justify="left").pack(side="left", fill="x")

            self._fill_analysis(inp, score, col['bar'])
            entry = dict(score=score, category=cat, inputs=inp,
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"))
            self._history.append(entry); save_history(self._history); self._render_history()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    @staticmethod
    def _build_recs(inp, score):
        r = []
        if inp['study'] < 4:  r.append({'text':'Increase study to 4–6h/day for meaningful gains.','color':'#A32D2D'})
        elif inp['study'] >= 8: r.append({'text':'Great effort — space sessions to avoid burnout.','color':'#3b6d11'})
        if inp['att'] < 75:   r.append({'text':'Attendance below 75% strongly predicts underperformance. Aim for 85%+.','color':'#A32D2D'})
        if inp['sleep'] < 6:  r.append({'text':'Sleep deprivation impairs memory consolidation. Target 7–9 hours.','color':'#854F0B'})
        if inp['screen'] > 5: r.append({'text':'High screen time crowds out study. Try capping at 3–4h/day.','color':'#854F0B'})
        if inp['extra'] == 0: r.append({'text':'Extracurriculars correlate with better focus and discipline.','color':'#185FA5'})
        if inp['group'] == 0 and inp['study'] < 5:
            r.append({'text':'A study group improves accountability and comprehension.','color':'#185FA5'})
        if not r: r.append({'text':"Strong trajectory — stay consistent and keep pushing.",'color':'#3b6d11'})
        return r

    def _export(self):
        if not self._history:
            messagebox.showinfo("Export","No history to export."); return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV","*.csv")],
                                            initialfile="predictions.csv")
        if not path: return
        with open(path,'w',newline='') as f:
            keys = ['timestamp','score','category','study','att','prev','sleep','screen','extra','group']
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for e in self._history:
                w.writerow({'timestamp':e['timestamp'],'score':e['score'],
                            'category':e['category'],**e['inputs']})
        messagebox.showinfo("Exported",f"Saved {len(self._history)} records to:\n{path}")

    def _clear(self):
        if self._history and messagebox.askyesno("Clear","Delete all history?"):
            self._history.clear(); save_history(self._history); self._render_history()


# ====================== RUN ======================
if __name__ == "__main__":
    app = App()
    app.mainloop()

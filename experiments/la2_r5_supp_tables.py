"""補充表格：直接由結果檔產生 LaTeX（確保文字與數據一致）。輸出到 TABDIR。"""
import os, json, glob, pickle
import numpy as np
import pandas as pd
from common import RESULTS

OUT = os.path.join(RESULTS, "la2")
TABDIR = os.environ.get("TABDIR", os.path.join(RESULTS, "la2", "supp_tables"))
os.makedirs(TABDIR, exist_ok=True)
ST = ["ε", "φ", "ν", "σ", "κ", "υ", "η", "δ"]
TEX = {"ε": r"$\eps$", "φ": r"$\varphi$", "ν": r"$\nu$", "σ": r"$\sigma$", "κ": r"$\kappa$", "υ": r"$\upsilon$",
       "η": r"$\eta$", "δ": r"$\delta$"}


def w(name, s):
    open(os.path.join(TABDIR, name + ".tex"), "w").write(s)


def pc(x, d=1):
    return "---" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}"


def ci(v, d=1, scale=100):
    return f"{scale * v['diff']:+.{d}f} [{scale * v['lo']:+.{d}f}, {scale * v['hi']:+.{d}f}]"


def tab_methods():
    d = json.load(open(os.path.join(OUT, "r1_main.json")))
    s = d["summary"]
    rows = [("ngram1", "$n$-gram, $n=1$ (lookup)"), ("ngram2", "$n$-gram, $n=2$"), ("ngram4", "$n$-gram, $n=4$"),
            ("ngram8", "$n$-gram, $n=8$"), ("ngram16", "$n$-gram, $n=16$"), ("obs_probe", "probe, current perception"),
            ("rand_probe", "probe, random-init transformer"), ("bo_probe", "probe, bo transformer"),
            ("gru_bo_probe", "probe, GRU-bo"), ("filter", "finite-state filter"),
            ("aux", "aux transformer (state head)"), ("gru_aux", "GRU-aux (state head)"), ("dual", "dual, free-running")]
    L = [r"\begin{tabular}{l" + "r" * 10 + "}", r"\toprule",
         "Method & " + " & ".join(TEX[x] for x in ST) + r" & All & Balanced\\", r"\midrule"]
    for k, lab in rows:
        L.append(lab + " & " + " & ".join(pc(s[k][x]) for x in ST) + f" & {pc(s[k]['all'])} & {pc(s[k]['balanced'])}" + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    w("tab_methods", "\n".join(L))
    # per seed（由 r1_perseed.json；la2_r1_offline.py 的逐回合預測可重建此檔）
    seeds = json.load(open(os.path.join(OUT, "r1_perseed.json")))
    L = [r"\begin{tabular}{r" + "r" * 9 + "rr}", r"\toprule",
         "Seed & " + " & ".join(TEX[x] for x in ST) + r" & All & aux & GRU-aux\\", r"\midrule"]
    for sd in seeds:
        a = sd["dual"]
        L.append(f"{sd['seed']} & " + " & ".join(pc(a[x]) for x in ST) + f" & {pc(a['all'])} & {pc(sd['aux']['all'])} & "
                 f"{pc(sd['gru_aux']['all'])}" + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    w("tab_perseed", "\n".join(L))
    # paired
    p = d["pairs"]
    comp = [("dual", "ngram8", "8-step $n$-gram"), ("dual", "ngram16", "16-step $n$-gram"), ("dual", "filter", "finite-state filter"),
            ("dual", "bo_probe", "bo probe"), ("dual", "aux", "aux"), ("dual", "gru_aux", "GRU-aux"),
            ("aux", "bo_probe", "bo probe"), ("aux", "ngram8", "8-step $n$-gram"), ("aux", "ngram16", "16-step $n$-gram"),
            ("gru_aux", "filter", "finite-state filter")]
    L = [r"\begin{tabular}{llrrrr}", r"\toprule", r"Model & minus & All steps & $\sigma$ & $\varphi$ & $\eps$\\", r"\midrule"]
    for a, b, lab in comp:
        L.append({"dual": "dual", "aux": "aux", "gru_aux": "GRU-aux"}[a] + f" & {lab} & " + " & ".join(
            ci(p[f"{a}|{b}|{x}"]) for x in ("all", "σ", "φ", "ε")) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    w("tab_paired", "\n".join(L))
    # intervention
    iv = d["intervene"]
    L = [r"\begin{tabular}{rrrrr}", r"\toprule", r"Seed & Cases & Follows (\%) & JS orig. & JS swapped\\", r"\midrule"]
    for i, x in enumerate(iv):
        L.append(f"{i} & {x['n']:,} & {100 * x['follow_frac']:.1f} & {x['js_orig']:.4f} & {x['js_swapped']:.4f}" + r"\\")
    L.append(r"\midrule")
    L.append(f"mean & & {100 * np.mean([x['follow_frac'] for x in iv]):.1f} & {np.mean([x['js_orig'] for x in iv]):.4f} & "
             f"{np.mean([x['js_swapped'] for x in iv]):.4f}" + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    w("tab_intervene", "\n".join(L))


def tab_act():
    d = json.load(open(os.path.join(OUT, "r1_act.json")))["summary"]
    lab = [("bo", "bo transformer"), ("aux", "aux transformer"), ("dual_teacher_states", "dual, teacher's states in context"),
           ("dual_own_states", "dual, own spoken states in context"), ("gru_bo", "GRU-bo"), ("gru_aux", "GRU-aux")]
    L = [r"\begin{tabular}{lrrr}", r"\toprule", r"Model & Action NLL (nats) & range over seeds & Top-1 (\%)\\", r"\midrule"]
    for k, l in lab:
        x = d[k]
        L.append(f"{l} & {x['action_nll']:.3f} & {x['nll_min']:.3f}--{x['nll_max']:.3f} & {100 * x['action_acc']:.1f}" + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    w("tab_act", "\n".join(L))


def tab_ctx():
    d = json.load(open(os.path.join(OUT, "r1_ctx.json")))
    rows = pd.DataFrame(d["rows"])
    L = [r"\begin{tabular}{lr" + "r" * 6 + "}", r"\toprule",
         r"Predictor & History (steps) & All & $\sigma$ & $\varphi$ & $\eps$ & $\kappa$ & Action NLL\\", r"\midrule"]
    for mode, lab in (("ngram", "$n$-gram"), ("aux", "aux"), ("dual", "dual")):
        for _, r in rows[rows["mode"] == mode].sort_values("ctx").iterrows():
            nll = f"{r['action_nll']:.3f}" if mode != "ngram" else "---"
            L.append(f"{lab} & {int(r.ctx)} & {pc(r['all'])} & {pc(r['σ'])} & {pc(r['φ'])} & {pc(r['ε'])} & {pc(r['κ'])} & {nll}" + r"\\")
    L.append(r"\midrule")
    for c in (4, 8, 16):
        L.append(f"dual $-$ aux & {c} & " + " & ".join(ci(d["pairs"][f"ctx{c}|dual-aux|{x}"]) for x in ("all", "σ", "φ", "ε"))
                 + r" & & \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    w("tab_ctx", "\n".join(L))


def tab_noise():
    d = json.load(open(os.path.join(OUT, "r1_noise.json")))
    m = json.load(open(os.path.join(OUT, "r1_main.json")))
    L = [r"\begin{tabular}{rrrrrrrrrr}", r"\toprule",
         r"$p$ (\%) & Trips & Dual & 8-gram & Filter & Dual $\sigma$ & 8-gram $\sigma$ & Dual $\varphi$ & 8-gram $\varphi$ & Filter $\varphi$\\",
         r"\midrule"]
    s = m["summary"]
    L.append(f"0 & {m['n_trips']:,} & {pc(s['dual']['all'])} & {pc(s['ngram8']['all'])} & {pc(s['filter']['all'])} & "
             f"{pc(s['dual']['σ'])} & {pc(s['ngram8']['σ'])} & {pc(s['dual']['φ'])} & {pc(s['ngram8']['φ'])} & {pc(s['filter']['φ'])}" + r"\\")
    for r in d["rows"]:
        L.append(f"{int(round(100 * r['p']))} & {r['n_trips']:,} & {pc(r['dual']['all'])} & {pc(r['ngram8']['all'])} & "
                 f"{pc(r['filter']['all'])} & {pc(r['dual']['σ'])} & {pc(r['ngram8']['σ'])} & {pc(r['dual']['φ'])} & "
                 f"{pc(r['ngram8']['φ'])} & {pc(r['filter']['φ'])}" + r"\\")
    L += [r"\midrule", r"\multicolumn{10}{l}{Paired differences, dual minus baseline (percentage points, trip bootstrap):}\\"]
    for r in d["rows"]:
        p = r["p"]
        L.append(f"{int(round(100 * p))} & & \\multicolumn{{4}}{{l}}{{vs 8-gram: {ci(d['pairs'][f'p{p}|dual|ngram8|all'])}}} & "
                 f"\\multicolumn{{4}}{{l}}{{vs filter: {ci(d['pairs'][f'p{p}|dual|filter|all'])}}}" + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    w("tab_noise", "\n".join(L))


def tab_gen():
    d = json.load(open(os.path.join(OUT, "r1_gen.json")))
    envs = [("train", "two-patch (new seed)"), ("barrier", "barrier (unseen map)"), ("food_fast", "food fast"),
            ("food_slow", "food slow"), ("decay_fast", r"decay fast ($\rho=0.94$)"), ("decay_slow", r"decay slow ($\rho=0.99$)")]
    meth = [("ngram8", "8-gram"), ("filter", "filter"), ("bo_probe", "bo probe"), ("aux", "aux"), ("gru_aux", "GRU-aux"), ("dual", "dual")]
    L = [r"\begin{tabular}{lrr" + "r" * len(meth) + "rr}", r"\toprule",
         r"Environment & Trips & $\sigma$ share & " + " & ".join(l for _, l in meth) + r" & dual $\sigma$ & 8-gram $\sigma$\\",
         r"\midrule"]
    for e, lab in envs:
        x = d[e]; s = x["summary"]
        L.append(f"{lab} & {x['n_trips']} & {pc(x['occupancy']['σ'])} & " + " & ".join(pc(s[k]["all"]) for k, _ in meth)
                 + f" & {pc(s['dual']['σ'])} & {pc(s['ngram8']['σ'])}" + r"\\")
    L += [r"\midrule", r"\multicolumn{11}{l}{Paired differences, dual minus baseline, all steps (percentage points):}\\"]
    for e, lab in envs:
        p = d[e]["pairs"]
        L.append(f"{lab} & \\multicolumn{{10}}{{l}}{{8-gram {ci(p['dual|ngram8|all'])}; filter {ci(p['dual|filter|all'])}; "
                 f"GRU-aux {ci(p['dual|gru_aux|all'])}; aux {ci(p['dual|aux|all'])}}}" + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    w("tab_gen_offline", "\n".join(L))


# ================================================================ 群體表格
COLAB = {"teacher": "teacher", "teacher_nomem": "teacher, no food-site memory", "bo_frozen": "bo, frozen",
         "bo_learn": "bo, learning", "aux_frozen": "aux, frozen", "aux_learn": "aux, learning",
         "dual_frozen": "dual, frozen", "dual_learn": "dual, learning", "dual_learn_A": "dual, learning actions only",
         "dual_learn_step": "dual, step-weighted", "dual_learn_sigma": "dual, search-balanced",
         "dual_learn_reh": "dual, teacher rehearsal", "dual_learn_rehNS": "dual, rehearsal without search",
         "dual_learn_rehS": "dual, rehearsal of search only", "dual_learn_reh ": ""}
ORDER = ["teacher", "teacher_nomem", "bo_frozen", "bo_learn", "aux_frozen", "aux_learn", "dual_frozen", "dual_learn",
         "dual_learn_A", "dual_learn_step", "dual_learn_sigma", "dual_learn_reh", "dual_learn_rehNS", "dual_learn_rehS"]


def _m(row, k, d=1, scale=100):
    return f"{scale * row[k]['mean']:.{d}f}" if k in row else "---"


def tab_colony(env, name, share="rich_share"):
    t = json.load(open(os.path.join(OUT, "r3_colony.json")))["tables"][env]
    L = [r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
         r"Condition & $n$ & Delivered & " + ("Rich share" if share == "rich_share" else "Short share") +
         r" & Timeouts & o$\to$a & (o,s)$\to$a & s$\to$s & Mark rich & Mark poor & Dispersion\\", r"\midrule"]
    for c in ORDER:
        if c not in t:
            continue
        r = t[c]
        dl = r["delivered"]
        disp = f"{r['dispersion']['mean']:.3f}" if "dispersion" in r else "---"
        L.append(f"{COLAB[c]} & {r['n']} & {dl['mean']:.0f} [{dl['lo']:.0f}, {dl['hi']:.0f}] & {_m(r, share)} & "
                 f"{r['timeouts']['mean']:.1f} & {_m(r, 'oa_legal')} & {_m(r, 'osa_legal')} & {_m(r, 'trans_legal')} & "
                 f"{_m(r, 'mark_q1')} & {_m(r, 'mark_q0')} & {disp}" + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    w(name, "\n".join(L))


def tab_colony_paired():
    r = json.load(open(os.path.join(OUT, "r3_colony.json")))
    envlab = {"two_patch": "two-patch", "double_bridge": "double bridge", "barrier": "barrier", "food_fast": "food fast",
              "food_slow": "food slow", "decay_fast": "decay fast", "decay_slow": "decay slow"}
    L = [r"\begin{tabular}{llrrrr}", r"\toprule", r"Environment & Comparison & $n$ & Difference (items) & $p$ & Holm $p$\\",
         r"\midrule"]
    last = None
    order = ["two_patch", "double_bridge", "barrier", "food_fast", "food_slow", "decay_fast", "decay_slow"]
    for s in sorted(r["paired"], key=lambda x: order.index(x["env"])):
        e = s["env"]
        L.append(f"{envlab[e] if e != last else ''} & {COLAB[s['a']]} $-$ {COLAB[s['b']]} & {s['n']} & "
                 f"{s['mean']:+.1f} [{s['ci_lo']:+.1f}, {s['ci_hi']:+.1f}] & {s['p']:.2g} & {s['p_holm']:.2g}" + r"\\")
        last = e
    L += [r"\bottomrule", r"\end{tabular}"]
    w("tab_colony_paired", "\n".join(L))


def tab_stability():
    t = json.load(open(os.path.join(OUT, "r3_colony.json")))["tables"]["two_patch"]
    L = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r"Condition & $\sigma$ practiced & Tracking after & $\sigma$ after & $\varphi$ after & $\eps$ after & Spoken $\sigma$ & Spoken $\eps$ & Spoken $\varphi$\\",
         r"\midrule"]
    for c in ORDER:
        if c not in t or c.startswith("bo") or c == "teacher_nomem":
            continue
        r = t[c]
        L.append(f"{COLAB[c]} & {_m(r, 'practiced_sigma')} & {_m(r, 'stab_all')} & {_m(r, 'stab_σ')} & {_m(r, 'stab_φ')} & "
                 f"{_m(r, 'stab_ε')} & {_m(r, 'occ_σ')} & {_m(r, 'occ_ε')} & {_m(r, 'occ_φ')}" + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    w("tab_stability", "\n".join(L))


def tab_marking():
    t = json.load(open(os.path.join(OUT, "r3_colony.json")))["tables"]["two_patch"]
    L = [r"\begin{tabular}{lrrrrr}", r"\toprule", r"Condition & $n$ & Rich $-$ poor & Seeds $>0$ & Rich, weak trail & Rich, strong trail\\",
         r"\midrule"]
    for c in ORDER:
        if c not in t or "mark_diff" not in t[c]:
            continue
        r = t[c]; m = r["mark_diff"]
        L.append(f"{COLAB[c]} & {r['n']} & {m['mean']:+.3f} [{m['ci_lo']:+.3f}, {m['ci_hi']:+.3f}] & {m['n_pos']}/{r['n']} & "
                 f"{r['mark_q1_weak']['mean']:.3f} & {r['mark_q1_strong']['mean']:.3f}" + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    w("tab_marking", "\n".join(L))


def tab_by_model():
    d = pd.read_csv(os.path.join(OUT, "r2_per_seed.csv"))
    d = d[d.env == "two_patch"]
    p = d.pivot_table(index="seed", columns="cond", values="delivered")
    p["ts"] = p.index % 10
    L = [r"\begin{tabular}{l" + "r" * 10 + "}", r"\toprule", "Comparison & " + " & ".join(str(i) for i in range(10)) + r"\\",
         r"\midrule"]
    for a, b in (("bo_learn", "bo_frozen"), ("aux_learn", "aux_frozen"), ("dual_learn", "dual_frozen"),
                 ("dual_learn_reh", "dual_learn"), ("dual_learn_rehS", "dual_learn")):
        x = (p[a] - p[b]).groupby(p.ts).mean()
        L.append(f"{COLAB[a]} $-$ {COLAB[b]} & " + " & ".join(f"{v:+.0f}" for v in x) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    w("tab_by_model", "\n".join(L))



def tab_gen_colony():
    r = json.load(open(os.path.join(OUT, "r3_colony.json")))
    T = r["tables"]
    P = {(x["env"], x["a"], x["b"]): x for x in r["paired"]}
    envs = [("barrier", "barrier"), ("food_fast", "food fast"), ("food_slow", "food slow"),
            ("decay_fast", r"decay fast ($\rho=0.94$)"), ("decay_slow", r"decay slow ($\rho=0.99$)")]
    def f(e, a, b):
        x = P.get((e, a, b))
        if not x:
            return "---"
        base = T[e][b]["delivered"]["mean"]
        return f"{x['rel_pct']:+.1f} [{100 * x['ci_lo'] / base:+.1f}, {100 * x['ci_hi'] / base:+.1f}]"
    L = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r" & Teacher & \multicolumn{3}{c}{Frozen vs teacher (\%)} & \multicolumn{3}{c}{Learning vs frozen (\%)}\\",
         r"Environment & items & bo & aux & dual & bo & aux & dual\\", r"\midrule"]
    for e, lab in envs:
        L.append(f"{lab} & {T[e]['teacher']['delivered']['mean']:.0f} & " + " & ".join(f(e, f"{m}_frozen", "teacher") for m in ("bo", "aux", "dual"))
                 + " & " + " & ".join(f(e, f"{m}_learn", f"{m}_frozen") for m in ("bo", "aux", "dual")) + r"\\")
    L += [r"\midrule"]
    for e, lab in (("food_fast", "food fast"), ("food_slow", "food slow")):
        L.append(f"\\multicolumn{{8}}{{l}}{{{lab}: dual teacher rehearsal $-$ dual learning {f(e, 'dual_learn_reh', 'dual_learn')}\\%; "
                 f"teacher's $\\sigma$ share {100 * T[e]['teacher']['occ_σ']['mean']:.1f}\\%}}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    w("tab_gen_colony", "\n".join(L))


def tab_teacher_search():
    """補充表 S19：教師回合中系統搜尋的比例（la2_e7_followup.py 的 teacherS 條件，two-patch，20 個種子）。"""
    d = pd.read_csv(os.path.join(OUT, "e7_per_seed.csv"))
    t = d[d.cond == "teacherS"]
    rows = [("trips containing $\\sigma$, successful trips", t.sig_in_success.mean()),
            ("trips containing $\\sigma$, timed-out trips", t.sig_in_fail.mean()),
            ("share of $\\sigma$ steps, trip-weighted (as sampled by self-imitation)", t.sigfrac_success.mean()),
            ("share of $\\sigma$ steps, step-weighted", t["occ_σ"].mean())]
    L = [r"\begin{tabular}{lr}", r"\toprule", r"Quantity & \%\\", r"\midrule"]
    L += [f"{a} & {pc(b)}" + r"\\" for a, b in rows]
    L += [r"\bottomrule", r"\end{tabular}"]
    w("tab_teacher_search", "\n".join(L))


ALL = ["tab_methods", "tab_act", "tab_ctx", "tab_noise", "tab_gen", "tab_colony_tp", "tab_colony_db", "tab_colony_paired",
       "tab_by_model", "tab_marking", "tab_stability", "tab_teacher_search", "tab_gen_colony"]


def tab_colony_tp():
    tab_colony("two_patch", "tab_colony_tp")


def tab_colony_db():
    tab_colony("double_bridge", "tab_colony_db", "short_share")


if __name__ == "__main__":
    import sys
    for f in (sys.argv[1:] or ALL):
        globals()[f]()
        print("wrote", f)



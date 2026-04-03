#!/usr/bin/env python3
"""
For each outer felt shell radius r3 (swept from 2.5" to 4.0"), find the
alumina felt thickness that keeps T_alumina ≤ 1650 °C (1923.15 K) via
binary search, then record the resulting tube T_max.

Usage:
    python3 optimize_alumina_thickness.py [--n N] [--r3-min IN] [--r3-max IN]
                                          [--al-min IN] [--al-max IN]
                                          [--tol K]

Saves: alumina_thickness_sweep.png
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ── CLI ───────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--n',      type=int,   default=8,   help='r3 sample count (default 8)')
ap.add_argument('--r3-min', type=float, default=3, metavar='IN', help='min r3 [in]')
ap.add_argument('--r3-max', type=float, default=4.5, metavar='IN', help='max r3 [in]')
ap.add_argument('--al-min', type=float, default=0.05, metavar='IN',
                help='min alumina thickness to search [in] (default 0.05)')
ap.add_argument('--al-max', type=float, default=0.70, metavar='IN',
                help='max alumina thickness to search [in] (default 0.70)')
ap.add_argument('--tol',    type=float, default=30.0, metavar='K',
                help='temperature convergence tolerance [K] (default 30)')
args = ap.parse_args()

T_TARGET_K = 1650.0 + 273.15   # alumina service limit in K
FEM_SCRIPT  = Path(__file__).parent / 'furnace_fem.py'
R2_IN       = 2.0               # tube outer radius [inches], fixed

r3_sweep = np.linspace(args.r3_min, args.r3_max, args.n)

# ── Output parsers ────────────────────────────────────────────────────────────
RE_TUBE    = re.compile(r'^Tube T_max\s*=\s*([\d.]+)\s*K',        re.MULTILINE)
RE_ALUMINA = re.compile(r'^Alumina felt T_max\s*=\s*([\d.]+)\s*K', re.MULTILINE)
RE_SPLIT   = re.compile(
    r'EM power split.*?tube\s+([\d.]+)\s+shell\s+([\d.]+)',
    re.DOTALL
)

def run_fem(r3_in, r4_in):
    proc = subprocess.run(
        [sys.executable, str(FEM_SCRIPT),
         f'--r3={r3_in:.4f}', f'--r4={r4_in:.4f}', '--no-plot'],
        capture_output=True, text=True
    )
    if proc.returncode != 0:
        return None
    out = proc.stdout
    m = RE_TUBE.search(out);    T_tube = float(m.group(1)) if m else np.nan
    m = RE_ALUMINA.search(out); T_al   = float(m.group(1)) if m else np.nan
    m = RE_SPLIT.search(out)
    P_tube  = float(m.group(1)) if m else np.nan
    P_shell = float(m.group(2)) if m else np.nan
    return T_tube, T_al, P_tube, P_shell

# ── Results ───────────────────────────────────────────────────────────────────
al_opt    = np.full(args.n, np.nan)   # optimal alumina thickness [inches]
T_tube    = np.full(args.n, np.nan)
T_alumina = np.full(args.n, np.nan)
P_tube    = np.full(args.n, np.nan)
P_shell   = np.full(args.n, np.nan)

# ── Outer sweep ───────────────────────────────────────────────────────────────
print(f"Sweeping r3 = {args.r3_min:.2f}\" – {args.r3_max:.2f}\"  ({args.n} points)")
print(f"Binary-searching alumina thickness in [{args.al_min:.2f}\", {args.al_max:.2f}\"]"
      f"  to hit T_alumina ≈ {T_TARGET_K:.0f} K  (±{args.tol:.0f} K)")
print()

for i, r3_in in enumerate(r3_sweep):
    felt_in = r3_in - R2_IN
    print(f"[{i+1}/{args.n}]  r3 = {r3_in:.3f}\"  (felt = {felt_in:.3f}\")", flush=True)

    # Bracket check: what temperature does the THINNEST alumina give?
    res_lo = run_fem(r3_in, r3_in + args.al_min)
    if res_lo is None:
        print("  ERROR at al_lo — skipping"); continue
    T_al_lo = res_lo[1]
    print(f"  al={args.al_min:.3f}\"  T_al={T_al_lo:.0f} K", flush=True)

    if T_al_lo >= T_TARGET_K:
        # Even minimum thickness exceeds the limit — can't reach target by adding more
        # Record at minimum thickness
        al_opt[i]    = args.al_min
        T_tube[i]    = res_lo[0]
        T_alumina[i] = T_al_lo
        P_tube[i]    = res_lo[2]
        P_shell[i]   = res_lo[3]
        print(f"  ↳ T_al already ≥ target at min thickness — using al={args.al_min:.3f}\"")
        continue

    res_hi = run_fem(r3_in, r3_in + args.al_max)
    if res_hi is None:
        print("  ERROR at al_hi — skipping"); continue
    T_al_hi = res_hi[1]
    print(f"  al={args.al_max:.3f}\"  T_al={T_al_hi:.0f} K", flush=True)

    if T_al_hi <= T_TARGET_K:
        # Even maximum thickness stays below limit — report at max thickness
        al_opt[i]    = args.al_max
        T_tube[i]    = res_hi[0]
        T_alumina[i] = T_al_hi
        P_tube[i]    = res_hi[2]
        P_shell[i]   = res_hi[3]
        print(f"  ↳ T_al still below target at max thickness — using al={args.al_max:.3f}\"")
        continue

    # Binary search: T_al increases with alumina thickness
    # (thicker alumina → more resistance → hotter inner surface)
    al_lo, al_hi = args.al_min, args.al_max
    best = res_hi   # fallback
    best_al = al_hi
    for _ in range(7):
        al_mid = 0.5 * (al_lo + al_hi)
        res = run_fem(r3_in, r3_in + al_mid)
        if res is None:
            print(f"  ERROR at al={al_mid:.3f}\" — stopping bisection"); break
        T_al_mid = res[1]
        print(f"  al={al_mid:.3f}\"  T_al={T_al_mid:.0f} K", flush=True)
        if abs(T_al_mid - T_TARGET_K) < args.tol:
            best, best_al = res, al_mid
            break
        if T_al_mid < T_TARGET_K:
            al_lo = al_mid   # need more thickness to reach target
        else:
            al_hi = al_mid   # too thick, back off
            best, best_al = res, al_mid

    al_opt[i]    = best_al
    T_tube[i]    = best[0]
    T_alumina[i] = best[1]
    P_tube[i]    = best[2]
    P_shell[i]   = best[3]
    print(f"  → alumina = {best_al:.3f}\"  "
          f"T_tube={T_tube[i]:.0f} K  T_al={T_alumina[i]:.0f} K")

# ── Summary ───────────────────────────────────────────────────────────────────
valid = ~np.isnan(T_tube)
print()
print("─" * 68)
print(f"{'r3 [in]':>8}  {'felt [in]':>9}  {'al [in]':>7}  "
      f"{'T_tube [K]':>10}  {'T_tube [°C]':>11}  {'T_al [K]':>8}")
print("─" * 68)
for i in range(args.n):
    if valid[i]:
        print(f"{r3_sweep[i]:8.3f}  {r3_sweep[i]-R2_IN:9.3f}  {al_opt[i]:7.3f}  "
              f"{T_tube[i]:10.0f}  {T_tube[i]-273.15:11.0f}  {T_alumina[i]:8.0f}")
print("─" * 68)

if valid.any():
    i_opt = int(np.nanargmax(T_tube))
    print(f"\nBest tube temperature at r3 = {r3_sweep[i_opt]:.3f}\"  "
          f"(felt = {r3_sweep[i_opt]-R2_IN:.3f}\",  alumina = {al_opt[i_opt]:.3f}\")")
    print(f"  T_tube = {T_tube[i_opt]:.1f} K  ({T_tube[i_opt]-273.15:.1f} °C)")
    print(f"  T_alumina = {T_alumina[i_opt]:.1f} K  ({T_alumina[i_opt]-273.15:.1f} °C)")

# ── Plot ──────────────────────────────────────────────────────────────────────
felt_in = r3_sweep - R2_IN

fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
fig.suptitle(
    f'Alumina thickness optimisation  '
    f'(T_alumina constrained ≈ {T_TARGET_K-273.15:.0f} °C,  P = const)\n'
    f'Felt OD swept from {args.r3_min:.2f}\" to {args.r3_max:.2f}\"',
    fontsize=10
)

def _shade(ax):
    ax.grid(True, alpha=0.3)
    if valid.any():
        ax.axvline(r3_sweep[i_opt] - R2_IN, color='green', lw=1.5, ls='--',
                   label=f'Best: felt={r3_sweep[i_opt]-R2_IN:.2f}\"')

# Panel 1: tube temperature
ax = axes[0]
ax.plot(felt_in[valid], T_tube[valid], 'o-', color='firebrick', lw=2)
_shade(ax)
ax.set_ylabel('Tube T_max  [K]')
ax.set_title('Tube peak temperature at alumina limit')
ax.legend(fontsize=9)

# Panel 2: alumina thickness
ax = axes[1]
ax.plot(felt_in[valid], al_opt[valid], 's-', color='goldenrod', lw=2)
_shade(ax)
ax.set_ylabel('Alumina thickness  [inches]')
ax.set_title(f'Required alumina thickness to hold T_alumina ≈ {T_TARGET_K-273.15:.0f} °C')
ax.legend(fontsize=9)

# Panel 3: EM power split
ax = axes[2]
ax.plot(felt_in[valid], P_tube[valid],  'o-',  color='steelblue', lw=2,   label='P tube')
ax.plot(felt_in[valid], P_shell[valid], 's--', color='orange',    lw=1.5, label='P shell')
_shade(ax)
ax.set_xlabel('Felt shell thickness  [inches]')
ax.set_ylabel('EM power absorbed  [W]')
ax.set_title('EM power split')
ax.legend(fontsize=9)

# secondary x-axis showing r3 on top of upper subplot only
ax2 = axes[0].twiny()
ax2.set_xlim(axes[0].get_xlim()[0] + R2_IN, axes[0].get_xlim()[1] + R2_IN)
ax2.set_xlabel('r₃  [inches]', fontsize=9)

fig.tight_layout()
out_png = Path(__file__).parent / 'alumina_thickness_sweep.png'
fig.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"\nSaved {out_png}")
plt.show()

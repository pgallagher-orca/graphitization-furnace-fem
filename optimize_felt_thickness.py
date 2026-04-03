#!/usr/bin/env python3
"""
Sweep the outer felt-shell outer radius (r3) at constant power to find the
thickness that maximises the graphite tube temperature.

Usage:
    python3 optimize_felt_thickness.py [--n N] [--r3-min INCHES] [--r3-max INCHES]
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
ap.add_argument('--n',      type=int,   default=10,  help='Number of r3 samples')
ap.add_argument('--r3-min', type=float, default=2.5, metavar='INCHES',
                help='Minimum outer-shell outer radius [in]  (default 2.5)')
ap.add_argument('--r3-max', type=float, default=5, metavar='INCHES',
                help='Maximum outer-shell outer radius [in]  (default 5.0)')
args = ap.parse_args()

FEM_SCRIPT = Path(__file__).parent / 'furnace_fem.py'
R2_IN = 2.0   # tube outer radius (inner face of felt shell) [inches]

r3_sweep = np.linspace(args.r3_min, args.r3_max, args.n)

# ── Output parsers ────────────────────────────────────────────────────────────
# Match the dedicated per-material print lines, not the Newton iteration lines.
RE_TUBE    = re.compile(r'^Tube T_max\s*=\s*([\d.]+)\s*K',   re.MULTILINE)
RE_ALUMINA = re.compile(r'^Alumina felt T_max\s*=\s*([\d.]+)\s*K', re.MULTILINE)
RE_SPLIT   = re.compile(
    r'EM power split.*?tube\s+([\d.]+)\s+shell\s+([\d.]+)\s+cap\s+([\d.]+)\s+door\s+([\d.]+)',
    re.DOTALL
)

def _parse(stdout):
    m = RE_TUBE.search(stdout)
    T_tube = float(m.group(1)) if m else np.nan

    m = RE_ALUMINA.search(stdout)
    T_al = float(m.group(1)) if m else np.nan

    m = RE_SPLIT.search(stdout)
    P_tube  = float(m.group(1)) if m else np.nan
    P_shell = float(m.group(2)) if m else np.nan

    return T_tube, T_al, P_tube, P_shell

# ── Results storage ───────────────────────────────────────────────────────────
T_tube    = np.full(args.n, np.nan)
T_alumina = np.full(args.n, np.nan)
P_tube    = np.full(args.n, np.nan)
P_shell   = np.full(args.n, np.nan)

# ── Sweep ─────────────────────────────────────────────────────────────────────
print(f"Sweeping r3 from {args.r3_min:.2f}\" to {args.r3_max:.2f}\" "
      f"in {args.n} steps  (felt = {args.r3_min - R2_IN:.2f}\" – "
      f"{args.r3_max - R2_IN:.2f}\")")
print()

for i, r3_in in enumerate(r3_sweep):
    print(f"[{i+1:2d}/{args.n}]  r3 = {r3_in:.3f}\"  "
          f"(felt = {r3_in - R2_IN:.3f}\")", flush=True)

    proc = subprocess.run(
        [sys.executable, str(FEM_SCRIPT), f'--r3={r3_in:.4f}', '--no-plot'],
        capture_output=True, text=True
    )

    if proc.returncode != 0:
        print(f"  ERROR (returncode {proc.returncode})")
        print(proc.stderr[-600:] if proc.stderr else '  (no stderr)')
        continue

    T_tube[i], T_alumina[i], P_tube[i], P_shell[i] = _parse(proc.stdout)
    print(f"         T_tube={T_tube[i]:.0f} K  T_alumina={T_alumina[i]:.0f} K  "
          f"P_tube={P_tube[i]:.0f} W  P_shell={P_shell[i]:.0f} W")

# ── Find optimum ──────────────────────────────────────────────────────────────
valid = ~np.isnan(T_tube)
if not valid.any():
    print("No valid results — check errors above.")
    sys.exit(1)

i_opt = int(np.nanargmax(T_tube))
print()
print("─" * 62)
print(f"Optimum  r3 = {r3_sweep[i_opt]:.3f}\"  "
      f"(felt = {r3_sweep[i_opt] - R2_IN:.3f}\")")
print(f"         Tube T_max  = {T_tube[i_opt]:.1f} K  "
      f"({T_tube[i_opt] - 273.15:.1f} °C)")
print(f"         Alumina T   = {T_alumina[i_opt]:.1f} K  "
      f"({T_alumina[i_opt] - 273.15:.1f} °C)")
print(f"         P_tube={P_tube[i_opt]:.1f} W   P_shell={P_shell[i_opt]:.1f} W")
print("─" * 62)

# ── Plot ──────────────────────────────────────────────────────────────────────
felt_in = r3_sweep - R2_IN

fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
fig.suptitle(
    f'Felt shell thickness sweep  '
    f'(tube OD = {R2_IN}\", 0.1\" alumina layer, P = const)\n'
    f'Optimum: felt = {r3_sweep[i_opt] - R2_IN:.2f}\"  →  '
    f'T_tube = {T_tube[i_opt]:.0f} K  ({T_tube[i_opt]-273.15:.0f} °C)',
    fontsize=10
)

# Panel 1: temperatures
ax = axes[0]
ax.plot(felt_in[valid], T_tube[valid],    'o-',  color='firebrick', lw=2,   label='Tube T_max')
ax.plot(felt_in[valid], T_alumina[valid], '^:',  color='goldenrod', lw=1.5, label='Alumina T_max')
ax.axvline(r3_sweep[i_opt] - R2_IN, color='green', lw=1.5, ls='--',
           label=f'Optimum  felt = {r3_sweep[i_opt]-R2_IN:.2f}\"')
ax.set_ylabel('Temperature  [K]')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
# secondary axis in r3 inches
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim()[0] + R2_IN, ax.get_xlim()[1] + R2_IN)
ax2.set_xlabel('r₃  [inches]', fontsize=9)

# Panel 2: EM power split
ax = axes[1]
P_total = np.where(valid, P_tube + P_shell, np.nan)  # approximate (ignores cap/door)
ax.plot(felt_in[valid], P_tube[valid],  'o-',  color='steelblue', lw=2,   label='P tube')
ax.plot(felt_in[valid], P_shell[valid], 's--', color='orange',    lw=1.5, label='P shell')
ax.axvline(r3_sweep[i_opt] - R2_IN, color='green', lw=1.5, ls='--')
ax.set_xlabel('Felt shell thickness  [inches]')
ax.set_ylabel('EM power absorbed  [W]')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig.tight_layout()
out_png = Path(__file__).parent / 'felt_thickness_sweep.png'
fig.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"Saved {out_png}")
plt.show()

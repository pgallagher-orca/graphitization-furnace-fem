# Graphite Induction Furnace — 2D Axisymmetric FEM

Steady-state temperature solver for a graphite tube furnace heated by induction.
Written in plain Python (NumPy / SciPy / Matplotlib), no external FEM library required.

## Quick start

```bash
python3 furnace_fem.py
```

Produces interactive Matplotlib windows and saves several PNG figures.

### Runtime flags

| Flag | Default | Description |
|---|---|---|
| `--r3 INCHES` | 4.8 | Outer felt shell outer radius (alumina inner radius) |
| `--r4 INCHES` | r3 + 0.1 | Alumina felt outer radius |
| `--no-plot` | off | Skip all matplotlib output (batch / headless mode) |

---

## Geometry

The furnace is rotationally symmetric about the tube axis (z). Only the r–z
half-plane is solved. All dimensions are in inches in the source and converted
to metres internally (`IN = 0.0254`).

```
r
^
r4 ┤════════════════════════════════════════╗  alumina felt
r3 ┤════════════════════════════════════════╣  graphite felt (outer shell)
   ║                                        ║
r2 ┤   ╔══════════════════╗────────────────╢  graphite felt (end caps / doors)
   ║   ║  graphite tube   ║                ║
r1 ┤   ╠══════════════════╣                ║
   ║   ║   (bore, void)   ║                ║
 0 ┤   ╚══════════════════╝                ║
   └───┬──────────────────┬────────────────┬──> z
    -7.5"              -3"  0  +3"        +7.5"
         ←doors→  ←shell+caps→  ←doors→
```

| Component | Material | r_inner | r_outer | z-extent |
|---|---|---|---|---|
| Susceptor tube | Molded graphite | r1 = 0.5" | r2 = 1.5" | ±3" |
| End caps (×2) | Graphite felt | 0 | r2 | z_L↔z_tL, z_tR↔z_R |
| Outer shell | Graphite felt | r2 | r3 | ±6" |
| Doors (×2) | Graphite felt | 0 | r3 | ±6"↔±7.5" |
| Alumina layer | Alumina felt | r3 | r4 | ±7.5" |

The tube bore (r < r1, |z| < 3") is not meshed; its inner wall is adiabatic.

---

## Physics

### Governing equation — steady-state thermal

$$\frac{1}{r}\frac{\partial}{\partial r}\!\left(r\, k(T)\frac{\partial T}{\partial r}\right) + \frac{\partial}{\partial z}\!\left(k(T)\frac{\partial T}{\partial z}\right) + Q(r,z) = 0$$

### Boundary conditions — thermal

- **Axis (r = 0):** symmetry (zero radial flux, natural BC)
- **All internal interfaces:** continuity of T and heat flux (enforced by shared nodes)
- **Outer surfaces** (r = r4 cylinder, z = ±z_RD end faces):
  Stefan–Boltzmann radiation to ambient,
  $q_\text{rad} = \sigma \varepsilon (T^4 - T_\text{amb}^4)$,  $T_\text{amb} = 300\,\text{K}$

Per-surface emissivities: alumina cylinder ε = 0.87, door sides and faces ε = 1.0.

### Material properties

| Material | Thermal conductivity | Source |
|---|---|---|
| Graphite tube | T-dependent, from CSV | `mersen_2020_graphite.csv` |
| Graphite felt | T-dependent, from CSV | `sgl_sigratherm_extrapolated.csv` |
| Alumina felt | T-dependent, from CSV | `alumina_mat_thermal_conductivity.csv` |

---

## Electromagnetic (eddy-current) solve

Heat source Q(r,z) is not assumed uniform — it is computed self-consistently by
solving the magnetoquasistatic (MQS) A_φ problem on a 200 × 480 finite-difference
grid, then normalised to P_TOTAL.

### Governing equation — MQS

In the Coulomb gauge, the azimuthal vector potential A_φ satisfies:

$$\left(\frac{\partial^2}{\partial r^2} + \frac{1}{r}\frac{\partial}{\partial r} - \frac{1}{r^2} + \frac{\partial^2}{\partial z^2}\right) A_\phi - \gamma^2(r,z)\,A_\phi = -\mu_0 J_\phi^\text{src}$$

where $\gamma^2 = j\omega\mu_0\sigma(r,z)$ and $\omega = 2\pi \times 9000\,\text{rad/s}$.

### Scattered-field decomposition

The coil vacuum field A_vac is computed analytically (Biot–Savart with elliptic
integrals for each turn). The scattered response of all conductors is then found by
solving the linear system

$$(L - \gamma^2)\,A_\text{scat} = \gamma^2\,A_\text{vac}, \quad A_\text{scat} = 0 \text{ at boundary}$$

with a sparse direct solver. The total field is $A_\text{tot} = A_\text{vac} + A_\text{scat}$.

### Electrical resistivities

| Region | ρ [Ω·m] |
|---|---|
| Graphite tube | 1.5 × 10⁻⁵ |
| Graphite felt (shell, caps, doors) | 1.8 × 10⁻³ |
| Alumina felt | 0 (insulator) |

Skin depths at 9 kHz: tube δ ≈ 20.6 mm (wall = 25.4 mm ≈ 1.2 δ);
felt δ ≈ 225 mm (shell << δ, long-wavelength limit).

### Power density

$$Q(r,z) = \frac{\sigma}{2}\,\omega^2\,|A_\phi|^2 \quad [\text{W m}^{-3}]$$

Normalised so that $\int Q\,dV = P_\text{TOTAL}$ (default 6000 W).

### 1-D sanity check

`solve_1d_eddy()` provides an analytic Bessel-function solution (I₁, K₁) for the
same geometry in the infinite-z limit. Bore shielding ratios from the 1-D and 2-D
solves agree to within ~2%, validating the FD implementation.

---

## Numerical method — thermal FEM

### Mesh

Three structured rectangular regions (tube zone, cap zones, shell zone) plus door
regions are meshed with bilinear Q4 elements and merged via coordinate-key
deduplication. Typical size: ~2200 nodes, ~2100 elements.

### Element stiffness (Q4 axisymmetric)

2 × 2 Gauss quadrature; nodal heat-source values Q_nodes are interpolated to Gauss
points with the same bilinear shape functions used for temperature.

### Nonlinear solver

Newton–Raphson with backtracking line search handles temperature-dependent felt
conductivity and T⁴ radiation simultaneously. A power-continuation warm-start
(6 steps from 1% to 100% load) is used to reach the initial guess.

---

## Optimisation scripts

### `optimize_felt_thickness.py`

Sweeps outer felt shell radius r3 at constant power and finds the thickness that
maximises tube peak temperature. Calls `furnace_fem.py --no-plot` as a subprocess.

```bash
python3 optimize_felt_thickness.py [--n N] [--r3-min IN] [--r3-max IN]
```

### `optimize_alumina_thickness.py`

For each r3 in a sweep range, binary-searches the alumina layer thickness r4 − r3
so that the alumina peak temperature converges to ≤ 1650 °C (the material service
limit). Records tube T_max at each optimised configuration.

```bash
python3 optimize_alumina_thickness.py [--n N] [--r3-min IN] [--r3-max IN]
                                       [--al-min IN] [--al-max IN] [--tol K]
```

### Representative results (P = 6000 W)

Unconstrained felt sweep (0.1" alumina fixed):

| Felt thickness | Tube T_max |
|---|---|
| 0.5" | 3474 K |
| **1.73"** | **4092 K** ← optimum |
| 3.7" | 3490 K |

Alumina-constrained sweep (T_alumina pinned at 1650 °C):

| Felt | Alumina | Tube T_max |
|---|---|---|
| 1.00" | 0.101" | 3887 K |
| **1.64"** | **0.141"** | **4110 K** ← optimum |
| 2.50" | 0.212" | 3968 K |

---

## Output files

| File | Description |
|---|---|
| `furnace_temperature.png` | Geometry cross-section + temperature map |
| `furnace_boundary_profile.png` | Radiated power vs. arc length on outer surface |
| `furnace_coil_fields.png` | H-field (streamlines), 2πr Q(r,z), dP/dr, dP/dz |
| `furnace_1d_sanity.png` | 1-D vs 2-D eddy-current comparison at z = 0 |
| `felt_thickness_sweep.png` | Tube T_max vs felt thickness |
| `alumina_thickness_sweep.png` | Tube T_max and alumina thickness vs felt radius |

---

## Dependencies

| Package | Purpose |
|---|---|
| numpy | Array arithmetic |
| scipy | Sparse matrices, direct solver, interpolation, special functions |
| pandas | Reading quoted-string CSV |
| matplotlib | Plotting and interactive hover |

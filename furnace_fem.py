#!/usr/bin/env python3
"""
2D axisymmetric FEM – steady-state temperature in a graphite induction furnace.

Geometry  (all dims in inches, stored in metres internally):
  ① Graphite susceptor tube : ID=2", OD=3", L=6",  axis along z, centred at z=0
  ② Graphite felt end cap   : OD=3", L=3",  at z=+3" → +6"  (one end only)
  ③ Graphite felt outer shell: ID=3", OD=10", L=12", centred at z=0

Physics:
  • Steady-state axisymmetric conduction  (T-dependent k for felt)
  • Stefan-Boltzmann radiation (ε=1) on all outer free surfaces to T_AMB

Coordinate system: r = radial, z = axial (along tube axis).
"""

import numpy as np
from collections import OrderedDict
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import interp1d
import matplotlib
for _backend in ("TkAgg", "Qt5Agg", "Qt6Agg", "GTK3Agg", "Agg"):
    try:
        matplotlib.use(_backend)
        break
    except Exception:
        pass
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.collections as mcollections
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm

# ── Physical constants ────────────────────────────────────────────────────────
SIGMA = 5.670374419e-8   # Stefan-Boltzmann  [W m⁻² K⁻⁴]
T_AMB = 300.0            # ambient / radiation sink temperature [K]
IN    = 0.0254           # 1 inch in metres

# ── Material thermal conductivities ─────────────────────────────────────────

# Graphite tube: Mersen 2020, interpolated from CSV (T in °C → converted to K)
_tube_csv = np.loadtxt(
    "mersen_2020_graphite.csv",
    delimiter=",", skiprows=1, usecols=(0, 1)   # columns: T [°C], k [W/mK]
)
_k_tube_interp = interp1d(
    _tube_csv[:, 0] + 273.15, _tube_csv[:, 1],   # °C → K
    kind="linear", bounds_error=False,
    fill_value=(_tube_csv[0, 1], _tube_csv[-1, 1])   # clamp outside range
)

def k_tube(T):
    """Temperature-dependent thermal conductivity of graphite tube [W m⁻¹ K⁻¹]."""
    return float(_k_tube_interp(T))

# Graphite felt: SGL Sigratherm, interpolated from CSV (T in °C → converted to K)
_felt_csv = np.loadtxt(
    "sgl_sigratherm_extrapolated.csv",
    delimiter=",", skiprows=1, usecols=(0, 1)   # columns: T [°C], k [W/mK]
)
_k_felt_interp = interp1d(
    _felt_csv[:, 0] + 273.15, _felt_csv[:, 1],   # °C → K
    kind="linear", bounds_error=False,
    fill_value=(_felt_csv[0, 1], _felt_csv[-1, 1])   # clamp outside range
)

def k_felt(T):
    """Temperature-dependent thermal conductivity of graphite felt [W m⁻¹ K⁻¹]."""
    return float(_k_felt_interp(T))

# Alumina felt: CSV already has T in Kelvin
_alumina_csv = np.loadtxt(
    "alumina_mat_thermal_conductivity.csv",
    delimiter=",", skiprows=1, usecols=(0, 1)   # columns: T [K], k [W/mK]
)
_k_alumina_interp = interp1d(
    _alumina_csv[:, 0], _alumina_csv[:, 1],     # T already in K
    kind="linear", bounds_error=False,
    fill_value=(_alumina_csv[0, 1], _alumina_csv[-1, 1])
)

def k_alumina(T):
    """Temperature-dependent thermal conductivity of alumina felt [W m⁻¹ K⁻¹]."""
    return float(_k_alumina_interp(T))


# ── Inductive heating ────────────────────────────────────────────────────────
P_TOTAL = 6000.0   # W, total power deposited uniformly in susceptor tube

# ── Geometry [m] ─────────────────────────────────────────────────────────────
r1  = 0.9 * IN    # tube inner radius
r2  = 2.0 * IN    # tube outer radius = felt OD/2 = outer-shell inner radius
r3  = 4.8 * IN    # outer-shell outer radius (= alumina felt inner radius)
z_L = -6.0 * IN   # outer-shell / domain left end
z_tL= -3.0 * IN   # tube left end
z_tR=  3.0 * IN   # tube right end  (= end-cap left face)
z_R =  6.0 * IN   # end-cap / outer-shell right end
z_LD = z_L - 1.5 * IN  # left door outer face
z_RD = z_R + 1.5 * IN  # right door outer face

# ── Runtime overrides (used by optimisation sweep) ────────────────────────────
import argparse as _argparse
_ap = _argparse.ArgumentParser(add_help=False)
_ap.add_argument('--r3', type=float, default=None, metavar='INCHES',
                 help='Outer-shell outer radius [in]; overrides compiled value')
_ap.add_argument('--r4', type=float, default=None, metavar='INCHES',
                 help='Alumina felt outer radius [in]; overrides r3+0.1" default')
_ap.add_argument('--no-plot', action='store_true',
                 help='Skip all matplotlib output (batch/headless mode)')
_CLI, _ = _ap.parse_known_args()
if _CLI.r3 is not None:
    r3 = _CLI.r3 * IN
r4 = r3 + 0.1 * IN    # alumina layer: 0.1" thick by default, tracks r3
if _CLI.r4 is not None:
    r4 = _CLI.r4 * IN
NO_PLOT = _CLI.no_plot

# ── Induction EM parameters ──────────────────────────────────────────────────
# Electrical resistivities used in the 2D eddy-current EM solve.
# σ = 1/ρ;  skin depth δ = √(2ρ/ωμ₀) at ω = 2π·9000 rad/s
RHO_TUBE = 1.5e-5   # graphite tube resistivity  [Ω·m]
RHO_FELT = 1.8e-3   # graphite felt resistivity   [Ω·m]  (also used for doors)
N_TURNS  = 8        # number of induction coil turns
R_COIL   = 5.5 * IN # coil radius  [m]

# ── Mesh parameters ──────────────────────────────────────────────────────────
NZ1   = 12    # elements in z: z_L  → z_tL   (outer shell only, 3" zone)
NZ2   = 24    # elements in z: z_tL → z_tR   (tube + outer shell, 6" zone)
NZ3   = 12    # elements in z: z_tR → z_R    (end cap + outer shell, 3" zone)
NR_T  = 6     # elements in r: r1  → r2      (tube)
NR_S  = 20    # elements in r: r2  → r3      (outer shell)
NR_C1 = 10    # elements in r: 0   → r1      (end-cap inner, solid felt)
NR_C2 = NR_T  # elements in r: r1  → r2      (end-cap outer = NR_T, for interface alignment)
NR_A  = 4     # elements in r: r3  → r4      (alumina felt outer ring)
NZ_D  = 6     # elements in z through each door (1.5" thick)

# ── Material emissivities (Stefan-Boltzmann radiation) ───────────────────────
EPS_GRAPHITE = 1.00   # graphite tube and felt
EPS_ALUMINA  = 0.87   # alumina felt
EPS_DOOR     = 1.00   # rigidized graphite felt door

# ── Material IDs ─────────────────────────────────────────────────────────────
MAT_TUBE    = 0
MAT_CAP     = 1
MAT_SHELL   = 2
MAT_ALUMINA = 3
MAT_DOOR    = 4
MAT_NAMES  = {MAT_TUBE: "Graphite tube", MAT_CAP: "Graphite felt (end cap)",
              MAT_SHELL: "Graphite felt (outer shell)", MAT_ALUMINA: "Alumina felt",
              MAT_DOOR: "Graphite felt (door)"}
MAT_COLORS = {MAT_TUBE: "#6c757d", MAT_CAP: "#adb5bd", MAT_SHELL: "#ced4da",
              MAT_ALUMINA: "#fff3cd", MAT_DOOR: "#495057"}


# ═══════════════════════════════════════════════════════════════════════════════
#  MESH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def build_mesh():
    """
    Build structured Q4 axisymmetric mesh for the three material regions.
    Shared boundary nodes are deduplicated via coordinate-key lookup.

    Returns
    -------
    nodes    : (N, 2) float array  [r, z]
    elements : (M, 5) int array    [n0, n1, n2, n3, material_id]
               node ordering: bottom-left, bottom-right, top-right, top-left
               in (r,z) space (r increases to right, z increases upward)
    """
    coord_map  = OrderedDict()   # (r_rounded, z_rounded) → global node index
    nodes_list = []              # list of [r, z]

    def node(r, z, prec=9):
        key = (round(float(r), prec), round(float(z), prec))
        if key not in coord_map:
            coord_map[key] = len(nodes_list)
            nodes_list.append([r, z])
        return coord_map[key]

    # ── Coordinate arrays ─────────────────────────────────────────────────────
    z_shell = np.concatenate([
        np.linspace(z_L,  z_tL, NZ1+1),
        np.linspace(z_tL, z_tR, NZ2+1)[1:],
        np.linspace(z_tR, z_R,  NZ3+1)[1:]
    ])
    z_tube = np.linspace(z_tL, z_tR, NZ2+1)
    z_cap  = np.linspace(z_tR, z_R,  NZ3+1)

    r_shell   = np.linspace(r2, r3, NR_S+1)
    r_alumina = np.linspace(r3, r4, NR_A+1)
    r_tube  = np.linspace(r1, r2, NR_T+1)
    r_cap   = np.concatenate([
        np.linspace(0,  r1, NR_C1+1),
        np.linspace(r1, r2, NR_C2+1)[1:]
    ])

    elements = []

    def add_region(r_arr, z_arr, mat):
        nr, nz = len(r_arr), len(z_arr)
        grid = [[node(r_arr[ir], z_arr[iz]) for ir in range(nr)]
                for iz in range(nz)]
        for iz in range(nz - 1):
            for ir in range(nr - 1):
                elements.append([
                    grid[iz  ][ir  ],   # bottom-left  (r_lo, z_lo)
                    grid[iz  ][ir+1],   # bottom-right (r_hi, z_lo)
                    grid[iz+1][ir+1],   # top-right    (r_hi, z_hi)
                    grid[iz+1][ir  ],   # top-left     (r_lo, z_hi)
                    mat
                ])

    z_cap_L  = np.linspace(z_L,  z_tL, NZ1+1)   # left end cap: z_L → z_tL
    z_door_L = np.linspace(z_LD, z_L,  NZ_D+1)  # left door:    z_LD → z_L
    z_door_R = np.linspace(z_R,  z_RD, NZ_D+1)  # right door:   z_R  → z_RD

    add_region(r_shell,   z_shell,  MAT_SHELL)
    add_region(r_alumina, z_shell,  MAT_ALUMINA)
    add_region(r_tube,    z_tube,   MAT_TUBE)
    add_region(r_cap,     z_cap,    MAT_CAP)     # right end cap
    add_region(r_cap,     z_cap_L,  MAT_CAP)     # left end cap

    # Doors: three radial sections each, sharing interface nodes at z_L / z_R
    # via coordinate-key deduplication.
    add_region(r_cap,     z_door_L, MAT_DOOR)
    add_region(r_shell,   z_door_L, MAT_DOOR)
    add_region(r_alumina, z_door_L, MAT_DOOR)
    add_region(r_cap,     z_door_R, MAT_DOOR)
    add_region(r_shell,   z_door_R, MAT_DOOR)
    add_region(r_alumina, z_door_R, MAT_DOOR)

    return np.array(nodes_list, dtype=float), np.array(elements, dtype=int)


# ═══════════════════════════════════════════════════════════════════════════════
#  ELEMENT ROUTINES  (Q4 bilinear, 2×2 Gauss, axisymmetric)
# ═══════════════════════════════════════════════════════════════════════════════

_GP  = 1.0 / np.sqrt(3.0)
_PTS = [(-_GP, -_GP), (_GP, -_GP), (_GP, _GP), (-_GP, _GP)]   # weights all = 1


def q4_matrices(r_lo, r_hi, z_lo, z_hi, k_func, T_nodes, Q_nodes):
    """
    4×4 element stiffness Ke and 4×1 force Fe for a Q4 axisymmetric element.

    k_func  : callable k(T)  → conductivity [W/mK] evaluated at each Gauss point
    Q_nodes : (4,) nodal volumetric source [W/m³]; interpolated to Gauss points
              via the same bilinear shape functions used for T
    T_nodes : (4,) nodal temperatures used to interpolate T at Gauss points
    Node index convention: 0=BL, 1=BR, 2=TR, 3=TL
    """
    dr    = r_hi - r_lo
    dz    = z_hi - z_lo
    r_c   = 0.5 * (r_lo + r_hi)
    J_det = 0.25 * dr * dz

    Ke = np.zeros((4, 4))
    Fe = np.zeros(4)

    for xi, eta in _PTS:
        N = 0.25 * np.array([
            (1 - xi) * (1 - eta),
            (1 + xi) * (1 - eta),
            (1 + xi) * (1 + eta),
            (1 - xi) * (1 + eta)
        ])
        dN_dxi  = 0.25 * np.array([-(1 - eta),  (1 - eta),  (1 + eta), -(1 + eta)])
        dN_deta = 0.25 * np.array([-(1 - xi),  -(1 + xi),   (1 + xi),  (1 - xi) ])

        dN_dr = dN_dxi  * (2.0 / dr)
        dN_dz = dN_deta * (2.0 / dz)

        r    = r_c + 0.5 * xi * dr
        T_gp = float(N @ T_nodes)     # temperature at Gauss point
        Q_gp = float(N @ Q_nodes)     # heat source at Gauss point
        k    = k_func(T_gp)

        Ke += k * (np.outer(dN_dr, dN_dr) + np.outer(dN_dz, dN_dz)) * r * J_det
        Fe += Q_gp * N * r * J_det

    return Ke, Fe


# ═══════════════════════════════════════════════════════════════════════════════
#  BOUNDARY EDGE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def find_radiation_edges(nodes, elements):
    """
    Return list of (n_a, n_b, eps) for all mesh edges on outer free surfaces:
      • r = r4              (outer cylindrical surface: alumina + door sides)
      • z = z_LD            (left outer face of left door)
      • z = z_RD            (right outer face of right door)

    Emissivity per edge:
      • r = r4, z in [z_L, z_R]      → EPS_ALUMINA
      • r = r4, z outside [z_L, z_R] → EPS_DOOR
      • z = z_LD or z = z_RD         → EPS_DOOR

    Identification strategy: edges belonging to exactly one element are on the
    mesh boundary; we then filter by geometric position.
    """
    from collections import Counter
    edge_cnt = Counter()
    for elem in elements:
        nn = elem[:4]
        for i in range(4):
            edge_cnt[tuple(sorted([nn[i], nn[(i+1) % 4]]))] += 1

    rad_edges = []
    tol = 1e-7

    for (na, nb), cnt in edge_cnt.items():
        if cnt != 1:
            continue
        r_a, z_a = nodes[na]
        r_b, z_b = nodes[nb]

        # outer cylindrical surface: alumina where z in [z_L, z_R], door elsewhere
        if abs(r_a - r4) < tol and abs(r_b - r4) < tol:
            z_mid = 0.5 * (z_a + z_b)
            eps = EPS_ALUMINA if (z_L - tol <= z_mid <= z_R + tol) else EPS_DOOR
            rad_edges.append((na, nb, eps))
        # left door outer face
        elif abs(z_a - z_LD) < tol and abs(z_b - z_LD) < tol:
            rad_edges.append((na, nb, EPS_DOOR))
        # right door outer face
        elif abs(z_a - z_RD) < tol and abs(z_b - z_RD) < tol:
            rad_edges.append((na, nb, EPS_DOOR))

    return rad_edges


# ═══════════════════════════════════════════════════════════════════════════════
#  RADIATION BOUNDARY CONTRIBUTIONS  (2-node line element, 2-pt Gauss)
# ═══════════════════════════════════════════════════════════════════════════════

def radiation_terms(na, nb, nodes, T, eps):
    """
    Compute the 2-node radiation residual and tangent matrix for edge (na, nb).

    eps         : surface emissivity for this edge
    Residual:   res[i]    = ∫ ε σ(T⁴ − T_AMB⁴) N_i r ds
    Tangent:    K_rad[i,j] = ∫ 4ε σT³ N_i N_j r ds
    """
    r_a, z_a = nodes[na]
    r_b, z_b = nodes[nb]
    T_a, T_b = T[na], T[nb]

    half_len = 0.5 * np.hypot(r_b - r_a, z_b - z_a)

    res  = np.zeros(2)
    Krad = np.zeros((2, 2))

    for s in (-_GP, _GP):
        N   = np.array([0.5 * (1 - s), 0.5 * (1 + s)])
        r   = N[0] * r_a + N[1] * r_b
        Tp  = N[0] * T_a + N[1] * T_b

        res  += eps * SIGMA * (Tp**4 - T_AMB**4)  * N * r * half_len
        Krad += eps * SIGMA * 4.0 * Tp**3 * np.outer(N, N) * r * half_len

    return res, Krad


# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

def assemble(nodes, elements, T, rad_edges, q_scale=1.0, Q_nodes=None):
    """
    Assemble global conductivity matrix K_base, radiation tangent K_rad,
    heat-source vector F_Q, and radiation residual F_rad.

    q_scale : multiply volumetric source by this factor (power continuation).
    Q_nodes : (N_nodes,) EM-derived volumetric heat source [W/m³] at every
              FEM node (from solve_em_fields → RegularGridInterpolator).
    Returns (J, K_base, F_Q, F_rad) where J = K_base + K_rad.
    """
    N_nodes = len(nodes)
    rows_K, cols_K, vals_K = [], [], []
    F_Q = np.zeros(N_nodes)

    for elem in elements:
        n   = elem[:4]
        mat = elem[4]

        rs = nodes[n, 0]
        zs = nodes[n, 1]
        r_lo, r_hi = rs.min(), rs.max()
        z_lo, z_hi = zs.min(), zs.max()

        if mat == MAT_TUBE:
            k_func = k_tube
        elif mat == MAT_ALUMINA:
            k_func = k_alumina
        elif mat == MAT_DOOR:
            k_func = k_felt   # doors are now graphite felt
        else:
            k_func = k_felt

        Q_e = Q_nodes[n] * q_scale if Q_nodes is not None else np.zeros(4)

        Ke, Fe = q4_matrices(r_lo, r_hi, z_lo, z_hi, k_func, T[n], Q_e)

        for i in range(4):
            F_Q[n[i]] += Fe[i]
            for j in range(4):
                rows_K.append(n[i])
                cols_K.append(n[j])
                vals_K.append(Ke[i, j])

    K_base = sp.csr_matrix((vals_K, (rows_K, cols_K)), shape=(N_nodes, N_nodes))

    # ── Radiation boundary ──────────────────────────────────────────────────
    rows_R, cols_R, vals_R = [], [], []
    F_rad = np.zeros(N_nodes)

    for (na, nb, eps) in rad_edges:
        res, Krad = radiation_terms(na, nb, nodes, T, eps)
        for li, gi in enumerate([na, nb]):
            F_rad[gi] += res[li]
            for lj, gj in enumerate([na, nb]):
                rows_R.append(gi)
                cols_R.append(gj)
                vals_R.append(Krad[li, lj])

    K_rad = sp.csr_matrix((vals_R, (rows_R, cols_R)), shape=(N_nodes, N_nodes))
    J = K_base + K_rad
    return J, K_base, F_Q, F_rad


# ═══════════════════════════════════════════════════════════════════════════════
#  NONLINEAR SOLVER  (Newton-Raphson)
# ═══════════════════════════════════════════════════════════════════════════════

def solve_temperature(nodes, elements, rad_edges, n_iter=60, rtol=1e-6,
                      Q_nodes=None):
    """
    Solve the nonlinear steady-state heat equation using Newton-Raphson with
    power-continuation warm-start and back-tracking line search.

    Q_nodes : (N_nodes,) EM-derived volumetric heat source [W/m³] at every FEM
              node.  If None, the analytic skin-depth model is used.

    Governing equation (residual = 0):
        R = K_base @ T + F_rad(T) − F_Q = 0

    Newton update:
        (K_base + K_rad_tangent) @ dT = −R
    """
    # ── Warm-start: ramp load from q_scale=0.01 to 1.0 ──────────────────
    print("  Warm-start (power continuation) …")
    N = len(nodes)
    T = np.full(N, T_AMB + 1.0)

    for q_scale in [0.01, 0.05, 0.15, 0.35, 0.65, 1.0]:
        for _ in range(8):
            J, K_base, F_Q, F_rad = assemble(nodes, elements, T, rad_edges,
                                              q_scale=q_scale, Q_nodes=Q_nodes)
            R  = K_base @ T + F_rad - F_Q
            dT = spla.spsolve(J.tocsc(), -R)
            T  = np.maximum(T + dT, T_AMB)
        print(f"    q_scale={q_scale:.2f}  T_max={T.max():.1f} K")

    # ── Full Newton-Raphson with line search ──────────────────────────────
    print("  Full Newton-Raphson …")
    for it in range(n_iter):
        J, K_base, F_Q, F_rad = assemble(nodes, elements, T, rad_edges,
                                         Q_nodes=Q_nodes)
        R      = K_base @ T + F_rad - F_Q
        R_norm = np.linalg.norm(R)

        dT = spla.spsolve(J.tocsc(), -R)

        # Back-tracking line search
        alpha = 1.0
        for _ in range(10):
            T_try = np.maximum(T + alpha * dT, T_AMB)
            _, Kb, Fq, Fr = assemble(nodes, elements, T_try, rad_edges,
                                         Q_nodes=Q_nodes)
            if np.linalg.norm(Kb @ T_try + Fr - Fq) < R_norm:
                break
            alpha *= 0.5

        T = np.maximum(T + alpha * dT, T_AMB)
        step_norm = np.linalg.norm(alpha * dT) / (np.linalg.norm(T) + 1e-12)
        print(f"  iter {it+1:3d}  |R|={R_norm:.3e}  α={alpha:.3f}"
              f"  T_max={T.max():.1f} K")

        if step_norm < rtol:
            print(f"  Converged after {it+1} iterations.")
            break

    return T


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_geometry(ax, nodes, elements):
    """Draw a 2-D cross-section (z horizontal, r vertical) of the mesh."""
    patches_by_mat = {MAT_TUBE: [], MAT_CAP: [], MAT_SHELL: [], MAT_ALUMINA: [], MAT_DOOR: []}

    for elem in elements:
        n   = elem[:4]
        mat = elem[4]
        # node columns: [r, z]; for the plot we want x=z, y=r
        r_v = nodes[n, 0]
        z_v = nodes[n, 1]
        # quad vertex order: BL, BR, TR, TL  → closed polygon; convert m → cm
        xy = np.column_stack([z_v[[0,1,2,3]] * 100, r_v[[0,1,2,3]] * 100])
        patches_by_mat[mat].append(mpatches.Polygon(xy, closed=True))

    for mat, plist in patches_by_mat.items():
        pc = mcollections.PatchCollection(
            plist, facecolor=MAT_COLORS[mat], edgecolor='none', alpha=0.9)
        ax.add_collection(pc)

    legend_handles = [
        mpatches.Patch(facecolor=MAT_COLORS[m], label=MAT_NAMES[m])
        for m in [MAT_TUBE, MAT_CAP, MAT_SHELL, MAT_ALUMINA, MAT_DOOR]
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8)

    ax.set_xlim((z_LD - 0.01) * 100, (z_RD + 0.01) * 100)
    ax.set_ylim(-0.2, (r4 + 0.005) * 100)
    ax.set_xlabel("z  [cm]")
    ax.set_ylabel("r  [cm]")
    ax.set_title("Geometry cross-section (axisymmetric about r = 0)")
    ax.set_aspect('equal')
    ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.3)
    ax.axvline(0, color='k', lw=0.5, ls='--', alpha=0.3)

    # Label dimensions
    ax.annotate("", xy=(z_tR * 100, r1 * 50), xytext=(z_tL * 100, r1 * 50),
                arrowprops=dict(arrowstyle="<->", color='black', lw=0.8))
    ax.text(0, r1 * 50 + 0.2, f'tube L=6"', ha='center', fontsize=7)


def plot_temperature(ax, fig, nodes, elements, T):
    """
    Colour-filled temperature map with hover tooltip.
    Uses tricontourf on a triangulation derived from the Q4 elements.
    """
    r_nodes = nodes[:, 0] * 100   # m → cm
    z_nodes = nodes[:, 1] * 100

    # Build triangulation from Q4 elements (split each quad into 2 triangles)
    tris = []
    for elem in elements:
        n = elem[:4]
        tris.append([n[0], n[1], n[2]])
        tris.append([n[0], n[2], n[3]])
    tris = np.array(tris)

    triang = mtri.Triangulation(z_nodes, r_nodes, tris)

    levels = np.linspace(T.min(), T.max(), 60)
    cs = ax.tricontourf(triang, T, levels=levels, cmap='inferno')

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    fig.colorbar(cs, cax=cax, label='Temperature  [K]')
    ax.set_xlabel("z  [cm]")
    ax.set_ylabel("r  [cm]")
    ax.set_title("Steady-state temperature field")
    ax.set_aspect('equal')
    ax.axhline(0, color='w', lw=0.4, ls='--', alpha=0.4)
    ax.axvline(0, color='w', lw=0.4, ls='--', alpha=0.4)

    # ── Hover annotation ────────────────────────────────────────────────────
    annot = ax.annotate("", xy=(0, 0), xytext=(12, 12),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                        fontsize=8)
    annot.set_visible(False)

    # Precompute element centroids in cm for nearest-neighbour lookup
    centroids_z = np.array([nodes[e[:4], 1].mean() for e in elements]) * 100
    centroids_r = np.array([nodes[e[:4], 0].mean() for e in elements]) * 100
    T_centroids  = np.array([T[e[:4]].mean() for e in elements])

    def on_motion(event):
        if event.inaxes is not ax:
            annot.set_visible(False)
            fig.canvas.draw_idle()
            return
        z_cur, r_cur = event.xdata, event.ydata
        if z_cur is None or r_cur is None:
            return
        dist2 = (centroids_z - z_cur)**2 + (centroids_r - r_cur)**2
        idx   = dist2.argmin()
        T_val = T_centroids[idx]
        mat   = elements[idx, 4]
        annot.xy = (z_cur, r_cur)
        annot.set_text(f"T = {T_val:.1f} K\n{MAT_NAMES[mat]}\n"
                       f"z={centroids_z[idx]:.2f} cm  "
                       f"r={centroids_r[idx]:.2f} cm")
        annot.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    return cs




# ═══════════════════════════════════════════════════════════════════════════════
#  INDUCTION COIL H-FIELD AND POWER DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════

def solve_em_fields():
    """
    Solve the 2D axisymmetric magnetoquasistatic problem for A_φ(r,z) and
    return a dict containing the H-field, Q distribution, and a
    RegularGridInterpolator suitable for driving the thermal FEM.

    Physics
    -------
    Scattered-field decomposition  A = A_vac + A_scat:
      A_vac   — vacuum Biot-Savart field (integrated from Hz_vac along r)
      A_scat  — back-field from eddy currents in the graphite tube, found by
                solving  (L − γ²χ) A_scat = γ²χ A_vac  with Dirichlet BCs

    Q = (σ/2) ω² |A_φ|²  in tube (σ_tube) and felt (σ_felt), normalised to
    P_TOTAL.  Q is also provided as a RegularGridInterpolator on the (r,z)
    grid for direct use in the thermal FEM.
    """
    from scipy.special import ellipk as _K, ellipe as _E
    from scipy.interpolate import RegularGridInterpolator

    z_turns = np.linspace(z_tL, z_tR, N_TURNS)
    mu0     = 4.0 * np.pi * 1e-7

    # ── Evaluation / solve grid ───────────────────────────────────────────────
    Nr, Nz = 200, 480
    r_max  = R_COIL * 1.15
    z_min  = z_LD - 0.02
    z_max  = z_RD + 0.02
    r_vec  = np.linspace(0.0, r_max, Nr)
    z_vec  = np.linspace(z_min, z_max, Nz)
    dr     = r_vec[1] - r_vec[0]
    dz_g   = z_vec[1] - z_vec[0]
    RR, ZZ = np.meshgrid(r_vec, z_vec, indexing='ij')   # (Nr, Nz)

    # ── Vacuum Biot-Savart H-field ────────────────────────────────────────────
    Hr_vac = np.zeros((Nr, Nz))
    Hz_vac = np.zeros((Nr, Nz))
    for z0 in z_turns:
        dZ     = ZZ - z0
        beta2  = (R_COIL + RR)**2 + dZ**2
        beta   = np.sqrt(beta2)
        alpha2 = np.maximum((R_COIL - RR)**2 + dZ**2, 1e-20)
        k2     = np.clip(4.0 * R_COIL * RR / beta2, 0.0, 1.0 - 1e-9)
        Kv_    = _K(k2)
        Ev_    = _E(k2)
        Hz_vac += (1.0/(2*np.pi*beta) *
                   (Kv_ + (R_COIL**2 - RR**2 - dZ**2)/alpha2 * Ev_))
        rs      = np.maximum(RR, 1e-14)
        Hr_vac += np.where(RR < 1e-12, 0.0,
                           dZ/(2*np.pi*rs*beta) *
                           (-Kv_ + (R_COIL**2 + rs**2 + dZ**2)/alpha2 * Ev_))

    # ── A_vac by radial integration of Hz_vac ────────────────────────────────
    # A_φ,vac(r,z) = (μ₀/r) ∫₀ʳ r′ H_z,vac(r′,z) dr′
    # (Derived from H_z = (1/μ₀r)∂(rA)/∂r; BC A(0,z)=0.)
    integrand = r_vec[:, None] * Hz_vac              # r′ Hz: (Nr, Nz)
    cum = np.concatenate([
        np.zeros((1, Nz)),
        np.cumsum(0.5*(integrand[:-1, :] + integrand[1:, :]) * dr, axis=0)
    ], axis=0)                                        # ∫₀^{r_i}: (Nr, Nz)
    A_vac = np.where(RR > 1e-12, mu0 * cum / np.maximum(RR, 1e-12), 0.0)

    # ── 2D eddy-current FD solve for A_scat ──────────────────────────────────
    # Equation: (L − γ²(r,z)) A_scat = γ²(r,z) A_vac,  A_scat = 0 at boundary
    # L = ∂²/∂r² + (1/r)∂/∂r − 1/r² + ∂²/∂z²
    # γ²(r,z) = jωμ₀σ(r,z) — spatially varying, nonzero in all conductors.
    #
    # Conductors and their resistivities (σ = 1/ρ):
    #   Graphite tube  ρ = RHO_TUBE  (r1→r2, z_tL→z_tR)
    #   Felt shell     ρ = RHO_FELT  (r2→r3, z_L→z_R)
    #   Felt end caps  ρ = RHO_FELT  (r<r2,  cap z-ranges)
    #   Felt doors     ρ = RHO_FELT  (r<r4,  door z-ranges)
    #   Alumina ring   σ ≈ 0         (r3→r4, z_L→z_R  — insulator)
    omega      = 2.0 * np.pi * 9000.0    # rad/s  (9 kHz)
    sigma_tube = 1.0 / RHO_TUBE
    sigma_felt = 1.0 / RHO_FELT

    in_tube  = (RR >= r1) & (RR <= r2) & (ZZ >= z_tL) & (ZZ <= z_tR)
    in_shell = (RR >= r2) & (RR <= r3) & (ZZ >= z_L)  & (ZZ <= z_R)
    in_cap   = (RR <= r2) & (((ZZ >= z_L)  & (ZZ <= z_tL)) |
                               ((ZZ >= z_tR) & (ZZ <= z_R)))
    in_door  = (RR <= r4) & (((ZZ >= z_LD) & (ZZ <= z_L)) |
                               ((ZZ >= z_R)  & (ZZ <= z_RD)))
    in_felt  = in_shell | in_cap | in_door

    sigma_map  = np.where(in_tube, sigma_tube,
                 np.where(in_felt, sigma_felt, 0.0))   # (Nr, Nz) real
    gamma2_map = 1j * omega * mu0 * sigma_map           # (Nr, Nz) complex

    n_cond = int((sigma_map > 0).sum())
    print(f"  Conductors in EM grid: {n_cond} cells  "
          f"(tube {int(in_tube.sum())}, felt {int(in_felt.sum())}  "
          f"[shell {int(in_shell.sum())}, cap {int(in_cap.sum())}, "
          f"door {int(in_door.sum())}])")

    N_dof = Nr * Nz

    # Vectorised interior stencil (i = 1…Nr-2, j = 1…Nz-2)
    i_int, j_int = np.meshgrid(np.arange(1, Nr-1),
                                np.arange(1, Nz-1), indexing='ij')
    i_int = i_int.ravel();  j_int = j_int.ravel()
    ri    = r_vec[i_int]

    k_c  = i_int*Nz + j_int
    k_rp = (i_int+1)*Nz + j_int
    k_rm = (i_int-1)*Nz + j_int
    k_zp = i_int*Nz + (j_int+1)
    k_zm = i_int*Nz + (j_int-1)

    g2_int = gamma2_map[i_int, j_int]   # complex γ²(r,z) at interior nodes
    v_c  = (-2.0/dr**2 - 1.0/ri**2 - 2.0/dz_g**2 - g2_int)
    v_rp =  1.0/dr**2 + 1.0/(2.0*ri*dr)
    v_rm =  1.0/dr**2 - 1.0/(2.0*ri*dr)
    v_z  =  1.0/dz_g**2 * np.ones_like(ri)

    row_int = np.tile(k_c, 5)
    col_int = np.concatenate([k_c, k_rp, k_rm, k_zp, k_zm])
    val_int = np.concatenate([v_c, v_rp, v_rm, v_z, v_z]).astype(complex)

    # Dirichlet A_scat = 0 on all four boundaries
    bnd = np.unique(np.concatenate([
        np.arange(Nz),                   # i=0  (axis r=0)
        np.arange((Nr-1)*Nz, N_dof),     # i=Nr-1
        np.arange(0, N_dof, Nz),         # j=0
        np.arange(Nz-1, N_dof, Nz),      # j=Nz-1
    ]))
    row_all = np.concatenate([row_int, bnd])
    col_all = np.concatenate([col_int, bnd])
    val_all = np.concatenate([val_int, np.ones(len(bnd), dtype=complex)])

    rhs = np.zeros(N_dof, dtype=complex)
    rhs[k_c] = g2_int * A_vac[i_int, j_int]

    M = sp.csr_matrix((val_all, (row_all, col_all)), shape=(N_dof, N_dof))
    print("Solving 2D eddy-current A_φ problem …", flush=True)
    A_scat_flat = spla.spsolve(M, rhs)
    A_scat = A_scat_flat.reshape(Nr, Nz)    # complex (Nr, Nz)

    A_tot = A_vac + A_scat                  # complex total vector potential

    # ── H-field from A_tot ────────────────────────────────────────────────────
    # H_z = (1/μ₀r) ∂(r A_φ)/∂r;   H_r = −(1/μ₀) ∂A_φ/∂z
    # Magnitude uses |H| (amplitude); streamlines use Re{H} (field at t=0).
    rA_c   = RR * A_tot
    drA_dr = np.gradient(rA_c, r_vec, axis=0)

    Hz_cplx        = drA_dr / (mu0 * np.maximum(RR, 1e-14))
    Hz_cplx[0, :] = 2.0 * A_tot[1, :] / (mu0 * dr)   # L'Hôpital at r=0
    Hr_cplx        = -np.gradient(A_tot, z_vec, axis=1) / mu0

    H_mag  = np.sqrt(np.abs(Hz_cplx)**2 + np.abs(Hr_cplx)**2)
    Hz_mod = np.real(Hz_cplx)
    Hr_mod = np.real(Hr_cplx)

    # Bore shielding: mean |H_z| at axis vs outer tube surface over tube z-span
    tube_z = (z_vec >= z_tL) & (z_vec <= z_tR)
    ir2    = int(np.argmin(np.abs(r_vec - r2)))
    H_bore = float(np.mean(np.abs(Hz_cplx[0,   tube_z])))
    H_r2   = float(np.mean(np.abs(Hz_cplx[ir2, tube_z])))
    shield = H_bore / H_r2 if H_r2 > 0 else 0.0
    print(f"Bore shielding  |H_z(r=0)| / |H_z(r=r2)| = {shield:.3f}  "
          f"({shield*100:.1f}%)")

    # ── Q = (σ/2) ω² |A_φ|² — self-consistent, normalised to P_TOTAL ─────────
    # sigma_map already encodes all conductors (tube + felt shell + cap + door).
    A2 = np.abs(A_tot)**2
    QQ = 0.5 * sigma_map * omega**2 * A2   # W/m³, zero where σ=0

    dVol   = 2.0 * np.pi * RR * dr * dz_g
    P_calc = float(np.sum(QQ * dVol))
    if P_calc > 0:
        QQ = QQ * (P_TOTAL / P_calc)
        def _P(mask): return float(np.sum(0.5*sigma_map*omega**2*A2*mask*dVol))/P_calc*P_TOTAL
        P_tube_W = _P(in_tube)
        P_shell_W= _P(in_shell)
        P_cap_W  = _P(in_cap)
        P_door_W = _P(in_door)
        print(f"EM power split (W):  tube {P_tube_W:.1f}  "
              f"shell {P_shell_W:.1f}  cap {P_cap_W:.1f}  door {P_door_W:.1f}  "
              f"/ total {P_TOTAL:.1f}")
    Q_Wcm3 = QQ * 1e-6    # W/m³ → W/cm³

    # ── RegularGridInterpolator for FEM use ───────────────────────────────────
    q_interp = RegularGridInterpolator(
        (r_vec, z_vec), QQ,
        method='linear', bounds_error=False, fill_value=0.0
    )

    return dict(
        r_vec=r_vec, z_vec=z_vec, dr=dr, dz_g=dz_g,
        QQ=QQ, Q_Wcm3=Q_Wcm3, q_interp=q_interp,
        A_tot=A_tot, Hz_cplx=Hz_cplx, Hr_cplx=Hr_cplx,
        H_mag=H_mag, Hz_mod=Hz_mod, Hr_mod=Hr_mod,
        shield=shield, sigma_map=sigma_map, omega=omega,
        in_tube=in_tube, in_felt=in_felt,
        N_TURNS=N_TURNS, R_COIL=R_COIL, z_turns=z_turns,
    )


def plot_coil_fields(em=None):
    """
    Four-panel figure from the EM solve results.  If *em* is None, calls
    solve_em_fields() first so the function can still be used standalone.
    """
    if em is None:
        em = solve_em_fields()

    r_vec   = em['r_vec'];    z_vec   = em['z_vec']
    dr      = em['dr'];       dz_g    = em['dz_g']
    QQ      = em['QQ'];       Q_Wcm3  = em['Q_Wcm3']
    A_tot   = em['A_tot']
    Hz_cplx = em['Hz_cplx']; Hr_cplx = em['Hr_cplx']
    H_mag   = em['H_mag'];   Hz_mod  = em['Hz_mod'];  Hr_mod = em['Hr_mod']
    shield  = em['shield']
    in_tube = em['in_tube'];  in_felt = em['in_felt']
    N_TURNS = em['N_TURNS'];  R_COIL  = em['R_COIL'];  z_turns = em['z_turns']
    Nr, Nz  = len(r_vec), len(z_vec)

    # ── Figure ────────────────────────────────────────────────────────────────
    rcm = r_vec * 100
    zcm = z_vec * 100

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(
        f'Induction coil: {N_TURNS} turns, r = {R_COIL/IN:.1f}",  '
        f'z = {z_tL/IN:.0f}" → {z_tR/IN:.0f}"  (I = 1 A/turn)  |  '
        f'bore shielding {shield*100:.1f}%',
        fontsize=11
    )

    def _geom_lines(ax):
        for r_b in (r1, r2, r3, r4):
            ax.axhline(r_b*100, color='white', lw=0.7, ls='--', alpha=0.5)
        ax.axhline(R_COIL*100, color='red', lw=0.9, ls='-', alpha=0.6)
        for z_b in (z_L, z_R):
            ax.axvline(z_b*100, color='cyan', lw=0.7, ls='--', alpha=0.5)
        for z_b in (z_tL, z_tR):
            ax.axvline(z_b*100, color='yellow', lw=0.8, ls=':', alpha=0.7)
        ax.scatter([zt*100 for zt in z_turns], [R_COIL*100]*N_TURNS,
                   c='red', s=35, zorder=6,
                   label=f'Coil turns  r = {R_COIL/IN:.1f}"')

    # Left panel: |H| amplitude + streamlines (Re part at t=0)
    H_clip = np.clip(H_mag, 0.0, np.percentile(H_mag, 97))
    cf1 = ax1.contourf(zcm, rcm, H_clip, levels=80, cmap='plasma')
    fig.colorbar(cf1, ax=ax1, label='|H|/I  [A m⁻¹ per A]',
                 fraction=0.035, pad=0.03)
    ax1.streamplot(zcm, rcm, Hz_mod, Hr_mod,
                   color='white', linewidth=0.55, density=1.8,
                   arrowsize=0.8, broken_streamlines=False)
    _geom_lines(ax1)
    ax1.set_xlabel('z  [cm]');  ax1.set_ylabel('r  [cm]')
    ax1.set_title('H-field (2D eddy-current solve — countercurrents + fringing)')
    ax1.set_aspect('equal')
    ax1.set_xlim(zcm[0], zcm[-1]);  ax1.set_ylim(0, rcm[-1])
    ax1.legend(fontsize=8, loc='upper right')

    # Right panel: 2πr Q (circumferential integral) with hover
    # Integrating Q [W cm⁻³] over φ gives 2π r Q [W cm⁻²] — power per unit r-z area
    dP_dA = 2.0 * np.pi * rcm[:, np.newaxis] * Q_Wcm3   # (Nr, Nz)  W cm⁻²
    dP_dA_plot = np.where(dP_dA > 0, dP_dA, np.nan)
    cf2 = ax2.contourf(zcm, rcm, dP_dA_plot, levels=60, cmap='inferno')
    fig.colorbar(cf2, ax=ax2, label='2πr Q  [W cm⁻²]',
                 fraction=0.035, pad=0.03)
    ax2.set_facecolor('#111111')
    _geom_lines(ax2)
    ax2.set_xlabel('z  [cm]');  ax2.set_ylabel('r  [cm]')
    ax2.set_title('Power dissipation 2πr Q(r,z)  [W cm⁻²]  — hover for value')
    ax2.set_aspect('equal')
    ax2.set_xlim(zcm[0], zcm[-1]);  ax2.set_ylim(0, rcm[-1])
    ax2.legend(fontsize=8, loc='upper right')

    _ann_kw = dict(xytext=(12, 12), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.88),
                   fontsize=8)
    annot_H = ax1.annotate('', xy=(0, 0), **_ann_kw)
    annot_Q = ax2.annotate('', xy=(0, 0), **_ann_kw)
    annot_H.set_visible(False)
    annot_Q.set_visible(False)

    def _mat_name(r_m, z_m):
        if r1 <= r_m <= r2 and z_tL <= z_m <= z_tR:
            return 'Graphite tube'
        if r2 <= r_m <= r3 and z_L <= z_m <= z_R:
            return 'Graphite felt (shell)'
        if r_m <= r2 and (z_L <= z_m <= z_tL or z_tR <= z_m <= z_R):
            return 'Graphite felt (end cap)'
        if r_m <= r4 and (z_LD <= z_m <= z_L or z_R <= z_m <= z_RD):
            return 'Graphite felt (door)'
        return 'Air / insulation'

    def _on_hover(event):
        if event.inaxes is ax1:
            annot_Q.set_visible(False)
            zc, rc = event.xdata, event.ydata
            if zc is None or rc is None:
                annot_H.set_visible(False)
            else:
                iz = int(np.clip(np.searchsorted(zcm, zc) - 1, 0, Nz - 1))
                ir = int(np.clip(np.searchsorted(rcm, rc) - 1, 0, Nr - 1))
                h  = H_mag[ir, iz]
                annot_H.xy = (zc, rc)
                annot_H.set_text(
                    f'|H| = {h:.2f} A/m per A\n{_mat_name(r_vec[ir], z_vec[iz])}\n'
                    f'r = {rc:.2f} cm   z = {zc:.2f} cm'
                )
                annot_H.set_visible(True)
        elif event.inaxes is ax2:
            annot_H.set_visible(False)
            zc, rc = event.xdata, event.ydata
            if zc is None or rc is None:
                annot_Q.set_visible(False)
            else:
                iz = int(np.clip(np.searchsorted(zcm, zc) - 1, 0, Nz - 1))
                ir = int(np.clip(np.searchsorted(rcm, rc) - 1, 0, Nr - 1))
                q  = dP_dA[ir, iz]
                annot_Q.xy = (zc, rc)
                annot_Q.set_text(
                    f'2πrQ = {q:.4f} W/cm²\n{_mat_name(r_vec[ir], z_vec[iz])}\n'
                    f'r = {rc:.2f} cm   z = {zc:.2f} cm'
                )
                annot_Q.set_visible(True)
        else:
            annot_H.set_visible(False)
            annot_Q.set_visible(False)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', _on_hover)

    # ── dP/dr at z=0  and  dP/dz ─────────────────────────────────────────────
    # All integrals in cm-based units (Q already in W/cm³, rcm/zcm in cm).
    #
    # dP/dr(r) at z=0:
    #   power per unit r per unit z in the r–z half-plane, folded by 2π:
    #   dP/(dr·dz) = 2π r Q(r, z=0)   [W cm⁻²]
    #
    # dP/dz(z):
    #   power per unit z, integrated over r from 0 to r4 and over φ:
    #   dP/dz = ∫₀^{r4} 2π r Q(r, z) dr   [W cm⁻¹]

    iz0     = int(np.argmin(np.abs(z_vec)))           # index of z ≈ 0
    ir4     = int(np.searchsorted(r_vec, r4))         # index of r ≈ r4

    dPdr_Wcm2 = 2.0*np.pi * rcm * Q_Wcm3[:, iz0]     # (Nr,)  W/cm²

    # Trapezoidal integration along r-axis for every z
    dPdz_Wcm  = 2.0*np.pi * np.trapezoid(
        Q_Wcm3[:ir4+1, :] * rcm[:ir4+1, None],
        rcm[:ir4+1], axis=0
    )                                                  # (Nz,)  W/cm

    # ── ax3: dP/dr at z=0 ────────────────────────────────────────────────────
    ax3.plot(rcm, dPdr_Wcm2, color='C1', lw=1.5)
    ax3.fill_between(rcm, dPdr_Wcm2, alpha=0.25, color='C1')
    ax3.set_xlabel('r  [cm]')
    ax3.set_ylabel('dP/(dr·dz)  [W cm⁻²]')
    ax3.set_title(f'Radial power profile at z = 0  (z = {z_vec[iz0]*100:.2f} cm)')
    ax3.set_xlim(0, r4*100 * 1.05)
    ax3.set_ylim(bottom=0)
    ax3.grid(True, alpha=0.3)
    # Mark material boundaries
    for r_b, lbl in ((r1, 'r₁'), (r2, 'r₂'), (r3, 'r₃'), (r4, 'r₄')):
        ax3.axvline(r_b*100, color='gray', lw=0.8, ls='--')
        ax3.text(r_b*100, ax3.get_ylim()[1], lbl,
                 ha='center', va='bottom', fontsize=8, color='gray')

    # ── ax4: dP/dz ────────────────────────────────────────────────────────────
    ax4.plot(zcm, dPdz_Wcm, color='C2', lw=1.5)
    ax4.fill_between(zcm, dPdz_Wcm, alpha=0.25, color='C2')
    ax4.set_xlabel('z  [cm]')
    ax4.set_ylabel('dP/dz  [W cm⁻¹]')
    ax4.set_title('Axial power profile  (integrated over r: 0 → r₄, full φ)')
    ax4.set_xlim(zcm[0], zcm[-1])
    ax4.set_ylim(bottom=0)
    ax4.grid(True, alpha=0.3)
    # Mark axial boundaries
    for z_b, lbl in ((z_tL, 'z_tL'), (z_tR, 'z_tR'),
                     (z_L,  'z_L'),  (z_R,  'z_R')):
        ax4.axvline(z_b*100, color='gray', lw=0.8, ls='--')
        ax4.text(z_b*100, ax4.get_ylim()[1], lbl,
                 ha='center', va='bottom', fontsize=7, color='gray')

    # Refresh axis limits after text (ax3 y-limit may have shifted)
    ax3.set_ylim(bottom=0)
    ax4.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig('furnace_coil_fields.png', dpi=150, bbox_inches='tight')
    print('Saved furnace_coil_fields.png')
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  BOUNDARY TEMPERATURE PROFILE
# ═══════════════════════════════════════════════════════════════════════════════

def plot_boundary_profile(nodes, rad_edges, T):
    """
    Pop up a separate figure showing temperature vs arc length along the outer
    boundary, traversed as:
        left door outer face  (r: 0 → r4,  z = z_LD)
      → outer surface         (z: z_LD → z_RD, r = r4)
      → right door outer face (r: r4 → 0,  z = z_RD)
    """
    tol = 1e-7

    left_nodes  = set()
    outer_nodes = set()
    right_nodes = set()

    for (na, nb, _eps) in rad_edges:
        r_a, z_a = nodes[na]
        r_b, z_b = nodes[nb]
        if abs(z_a - z_LD) < tol and abs(z_b - z_LD) < tol:
            left_nodes.add(na);  left_nodes.add(nb)
        elif abs(r_a - r4) < tol and abs(r_b - r4) < tol:
            outer_nodes.add(na); outer_nodes.add(nb)
        elif abs(z_a - z_RD) < tol and abs(z_b - z_RD) < tol:
            right_nodes.add(na); right_nodes.add(nb)

    left_sorted  = sorted(left_nodes,  key=lambda n: nodes[n, 0])           # r ↑
    outer_sorted = sorted(outer_nodes, key=lambda n: nodes[n, 1])           # z ↑
    right_sorted = sorted(right_nodes, key=lambda n: nodes[n, 0], reverse=True)  # r ↓

    # Concatenate, dropping the shared corner nodes at the joins
    path = left_sorted + outer_sorted[1:] + right_sorted[1:]

    r_path = np.array([nodes[n, 0] for n in path])
    z_path = np.array([nodes[n, 1] for n in path])
    T_path = np.array([T[n]        for n in path])

    ds = np.hypot(np.diff(r_path), np.diff(z_path))
    s  = np.concatenate([[0.0], np.cumsum(ds)]) * 100   # cm

    # Arc-length positions of the two corners
    s_c1 = s[len(left_sorted) - 1]               # end of left face
    s_c2 = s[len(left_sorted) + len(outer_sorted) - 2]  # end of outer surface

    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.plot(s, T_path, color='steelblue', lw=1.5)
    ax.axvline(s_c1, color='gray', ls='--', lw=0.8)
    ax.axvline(s_c2, color='gray', ls='--', lw=0.8)

    y_top = ax.get_ylim()[1]
    ax.text(s_c1 / 2,            y_top, 'left face',     ha='center', va='top', fontsize=8, color='gray')
    ax.text((s_c1 + s_c2) / 2,  y_top, 'outer surface', ha='center', va='top', fontsize=8, color='gray')
    ax.text((s_c2 + s[-1]) / 2, y_top, 'right face',    ha='center', va='top', fontsize=8, color='gray')

    ax.set_xlabel('Arc length along outer boundary  [cm]')
    ax.set_ylabel('Temperature  [K]')
    ax.set_title('Outer boundary temperature profile')
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig('furnace_boundary_profile.png', dpi=150, bbox_inches='tight')
    print('Saved furnace_boundary_profile.png')
    return fig2


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── EM solve (must come first: provides Q source for thermal FEM) ──────────
    print("Solving EM fields (2D eddy-current A_φ) …")
    em = solve_em_fields()

    # Interpolate EM Q [W/m³] onto every FEM node in one vectorised call
    print("Mapping EM power onto FEM nodes …")

    print("Building mesh …")
    nodes, elements = build_mesh()
    print(f"  {len(nodes)} nodes, {len(elements)} elements")

    Q_nodes = em['q_interp'](nodes)   # shape (N_nodes,), W/m³

    print("Finding radiation edges …")
    rad_edges = find_radiation_edges(nodes, elements)
    print(f"  {len(rad_edges)} radiation boundary edges")

    print(f"\nSolving thermal FEM (P_total = {P_TOTAL:.0f} W, EM power model) …")
    T = solve_temperature(nodes, elements, rad_edges, Q_nodes=Q_nodes)

    tube_node_ids = np.unique(elements[elements[:, 4] == MAT_TUBE, :4])
    T_tube_max = T[tube_node_ids].max()

    print(f"\nT_min = {T.min():.1f} K   T_max = {T.max():.1f} K")
    print(f"Tube T_max = {T_tube_max:.1f} K  ({T_tube_max - 273.15:.1f} °C)")

    # ── Alumina felt temperature check ────────────────────────────────────────
    alumina_mask = elements[:, 4] == MAT_ALUMINA
    alumina_node_ids = np.unique(elements[alumina_mask, :4])
    T_alumina_max = T[alumina_node_ids].max()
    T_ALUMINA_LIMIT_K = 1650.0 + 273.15   # 1650 °C in Kelvin
    print(f"Alumina felt T_max = {T_alumina_max:.1f} K  ({T_alumina_max - 273.15:.1f} °C)")
    if T_alumina_max > T_ALUMINA_LIMIT_K:
        print()
        print("!" * 70)
        print("!!! WARNING: ALUMINA FELT EXCEEDS MAXIMUM TEMPERATURE !!!")
        print(f"!!!   T_max = {T_alumina_max - 273.15:.1f} °C  (limit = 1650 °C)           !!!")
        print("!" * 70)
        print()

    # ── Power balance ─────────────────────────────────────────────────────────
    # Accumulate area and radiated power per named surface segment.
    seg_keys  = ["outer_cyl_alumina", "outer_cyl_door", "door_faces"]
    seg_label = {
        "outer_cyl_alumina": "Outer cylinder – alumina    (r=r4, z_L→z_R)",
        "outer_cyl_door":    "Outer cylinder – door sides (r=r4, |z|>z_L)",
        "door_faces":        "Door outer faces            (z=z_LD, z=z_RD)",
    }
    seg_eps   = {"outer_cyl_alumina": EPS_ALUMINA, "outer_cyl_door": EPS_DOOR,
                 "door_faces": EPS_DOOR}
    seg_area  = dict.fromkeys(seg_keys, 0.0)
    seg_power = dict.fromkeys(seg_keys, 0.0)

    tol = 1e-7
    gp  = 1.0 / np.sqrt(3.0)

    for (na, nb, eps) in rad_edges:
        r_a, z_a = nodes[na];  r_b, z_b = nodes[nb]
        T_a, T_b = T[na], T[nb]
        half_len = 0.5 * np.hypot(r_b - r_a, z_b - z_a)

        if abs(r_a - r4) < tol and abs(r_b - r4) < tol:
            z_mid = 0.5 * (z_a + z_b)
            key = "outer_cyl_alumina" if (z_L - tol <= z_mid <= z_R + tol) else "outer_cyl_door"
        else:
            key = "door_faces"

        edge_area  = 0.0
        edge_power = 0.0
        for s in (-gp, gp):
            N    = np.array([0.5*(1-s), 0.5*(1+s)])
            r_gp = float(N @ [r_a, r_b])
            T_gp = float(N @ [T_a, T_b])
            dA   = r_gp * half_len           # integrand for area (×2π outside)
            edge_area  += dA
            edge_power += eps * SIGMA * (T_gp**4 - T_AMB**4) * dA

        seg_area[key]  += edge_area
        seg_power[key] += edge_power

    # Multiply by 2π to convert half-plane integrals to full cylindrical values
    for k in seg_keys:
        seg_area[k]  *= 2 * np.pi
        seg_power[k] *= 2 * np.pi

    P_rad    = sum(seg_power.values())
    P_rad_total = P_rad   # keep for balance check

    # FEM-assembled source
    _, K_base, F_Q_final, _ = assemble(nodes, elements, T, rad_edges,
                                        Q_nodes=Q_nodes)
    P_in_fem = F_Q_final.sum() * 2 * np.pi

    # ── Print surface radiation table ─────────────────────────────────────────
    col = 42
    hdr = f"{'Surface':<{col}}  {'Area (m²)':>10}  {'ε':>5}  {'Power (W)':>10}  {'%':>6}"
    print()
    print(hdr)
    print("-" * len(hdr))
    for k in seg_keys:
        pct = 100.0 * seg_power[k] / P_rad_total if P_rad_total else 0.0
        print(f"{seg_label[k]:<{col}}  {seg_area[k]:>10.5f}  {seg_eps[k]:>5.2f}"
              f"  {seg_power[k]:>10.2f}  {pct:>5.1f}%")
    print("-" * len(hdr))
    print(f"{'TOTAL':<{col}}  {sum(seg_area.values()):>10.5f}  {'—':>5}"
          f"  {P_rad_total:>10.2f}  100.0%")
    print()
    print(f"Power target (P_TOTAL):    {P_TOTAL:.4f} W")
    print(f"Power in  (FEM assembled): {P_in_fem:.4f} W")
    print(f"Power out (radiation):     {P_rad_total:.4f} W")
    print(f"Balance error:             {abs(P_rad_total - P_in_fem):.2e} W")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if not NO_PLOT:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        fig.suptitle("Graphite Induction Furnace – 2D Axisymmetric FEM", fontsize=12)

        plot_geometry(ax1, nodes, elements)
        plot_temperature(ax2, fig, nodes, elements, T)

        plt.tight_layout()
        plt.savefig("furnace_temperature.png", dpi=150, bbox_inches='tight')
        print("Saved furnace_temperature.png")

        plot_boundary_profile(nodes, rad_edges, T)
        plot_coil_fields(em)

        plt.show()


if __name__ == "__main__":
    main()

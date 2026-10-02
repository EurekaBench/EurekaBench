# Ice-shelf forward model with FEniCSx

## The interpreter

FEniCSx (dolfinx 0.9 with ufl, basix, petsc4py, mpi4py, numpy) is installed in
its own environment. Run every solver script with

    /opt/fenicsx/envs/dolfinx/bin/python your_script.py

The default `python3` has numpy, scipy, sympy, jax, matplotlib and pandas but
not dolfinx: use it for analysis and for any network training, and the FEniCSx
interpreter for the forward model. Run one process at a time on
`MPI.COMM_SELF`; there is no mpirun. Save every field you generate under
/workspace/observations as a `.npz` with the grids `x`, `y` (m), `u`, `v`
(m/yr) and `h` (m), so that it survives and can be read with `numpy.load`.

## The model

Domain: the rectangle 0 <= x <= Lx, 0 <= y <= Ly (defaults Lx = 80 km,
Ly = 50 km), a triangle mesh with `nx` cells along x (16 to 400; 128 takes
seconds to a few minutes) and proportionally many along y. Ice flows in at
x = 0 and leaves through the calving front at x = Lx.

Fields: the horizontal velocity (u, v) in a vector P1 space, the thickness h
and the viscosity mu in scalar P1 spaces.

Constants: rho_i = 917 kg/m^3, rho_w = 1030 kg/m^3, g = 9.8 m/s^2 and the
reduced gravity g' = g (1 - rho_i/rho_w). Velocities are prescribed in m/yr
and converted to m/s inside the solver with 1 yr = 3.1536e7 s; convert back
on output.

Momentum balance, the classical isotropic SSA in weak form. With
eps = sym(grad u), the depth-integrated membrane stress is
stress = 2 mu h (eps + div(u) I), whose components are mu h (4 u_x + 2 v_y),
mu h (u_y + v_x) and mu h (2 u_x + 4 v_y), and for every test function phi

    integral( stress : sym(grad phi) ) dx
        = - integral( rho_i g' h grad(h) . phi ) dx
          + boundary_integral( (1/2) rho_i g' h^2 n . phi ) ds

The boundary integral is the ocean-pressure condition. It is assembled on the
whole boundary and acts only where no Dirichlet condition overrides it, which
is the calving front. Dirichlet conditions: at x = 0, u = inlet profile and
v = 0; at y = 0 and y = Ly, u = side profile and v = 0. With mu and h given
this is one linear solve; a direct LU solve is fine at these sizes.

Mass conservation, steady state: d(h u)/dx + d(h v)/dy = 0 with h prescribed
at x = 0. Reach it by pseudo-time stepping the advection equation
h_new + dt div(h_new u) = h_old with SUPG stabilisation
(test = psi + tau u . grad psi, tau = cell_diameter / (2 |u|)), relax
h <- (h_old + h_new) / 2, alternate with the momentum solve, and stop when the
relative change of u and h per step falls below 1e-6, or when the relative
change of u is below 2e-4 and the 99th percentile of |dh/dt| is below 0.1 m/yr
for three consecutive steps. Use dt = clip(0.5 cell_size / speed, 1 yr,
max(1 yr, 0.016 cell_size)) each step, speed being the largest |u| + |v| in
m/yr. A few hundred outer iterations suffice; stop if u or h stops being
finite.

Default inputs, as expressions in x, y with u0 = 1000 m/yr and h0 = 500 m:

    viscosity mu(x, y) = 5e13 (1 - 0.5 cos(2 pi y/Ly)) (2/3 + x/(3 Lx))   [Pa s]
    inlet u(y)         = u0 4 (y/Ly) (1 - y/Ly)                            [m/yr]
    inlet h(y)         = h0 (1 - 3 (y/Ly - 0.5)^4)                         [m]
    side u(x)          = u0 1.5 (x/Lx)^2                                   [m/yr]

Any positive mu(x, y), any boundary profiles and any domain size are inputs you
choose. To simulate a shelf governed by different equations, change the
`stress` expression and the boundary integral to match them; the rest of the
recipe stays the same.

## Recipe

```python
import numpy as np
import ufl
from dolfinx import fem, geometry, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc

RHO_I, RHO_W, G, YEAR = 917.0, 1030.0, 9.8, 3.1536e7
GP = G * (1 - RHO_I / RHO_W)
Lx, Ly, u0, h0, nx = 80e3, 50e3, 1000.0, 500.0, 128
ny = max(int(round(nx * Ly / Lx)), 8)

domain = mesh.create_rectangle(MPI.COMM_SELF, [np.array([0.0, 0.0]), np.array([Lx, Ly])],
                               [nx, ny], mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
Q = fem.functionspace(domain, ("Lagrange", 1))

mu, h, h_in, u_bc = fem.Function(Q), fem.Function(Q), fem.Function(Q), fem.Function(V)
mu.interpolate(lambda c: 5e13 * (1 - 0.5 * np.cos(2 * np.pi * c[1] / Ly)) * (2 / 3 + c[0] / (3 * Lx)))
h.interpolate(lambda c: h0 * (1 - 3 * (c[1] / Ly - 0.5) ** 4))
h_in.x.array[:] = h.x.array
u_bc.interpolate(lambda c: np.vstack([
    np.where(np.isclose(c[0], 0.0), u0 * 4 * (c[1] / Ly) * (1 - c[1] / Ly),
             u0 * 1.5 * (c[0] / Lx) ** 2) / YEAR,
    np.zeros(c.shape[1])]))

wall = lambda x: np.isclose(x[0], 0.0) | np.isclose(x[1], 0.0) | np.isclose(x[1], Ly)
bc_u = fem.dirichletbc(u_bc, fem.locate_dofs_geometrical(V, wall))
bc_h = fem.dirichletbc(h_in, fem.locate_dofs_geometrical(Q, lambda x: np.isclose(x[0], 0.0)))

u, phi = ufl.TrialFunction(V), ufl.TestFunction(V)
n = ufl.FacetNormal(domain)
eps = ufl.sym(ufl.grad(u))
stress = 2 * mu * h * (eps + ufl.div(u) * ufl.Identity(2))   # classical isotropic SSA
a_mom = ufl.inner(stress, ufl.sym(ufl.grad(phi))) * ufl.dx
l_mom = (-RHO_I * GP * h * ufl.inner(ufl.grad(h), phi) * ufl.dx
         + 0.5 * RHO_I * GP * h ** 2 * ufl.inner(n, phi) * ufl.ds)
mom = LinearProblem(a_mom, l_mom, bcs=[bc_u],
                    petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

dt = fem.Constant(domain, PETSc.ScalarType(1.0))
u_sol, h_old = fem.Function(V), fem.Function(Q)
hh, psi = ufl.TrialFunction(Q), ufl.TestFunction(Q)
tau = ufl.CellDiameter(domain) / (2 * ufl.sqrt(ufl.dot(u_sol, u_sol) + 1e-20))
test = psi + tau * ufl.dot(u_sol, ufl.grad(psi))
adv = LinearProblem((hh + dt * ufl.div(hh * u_sol)) * test * ufl.dx, h_old * test * ufl.dx,
                    bcs=[bc_h], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

cell = min(Lx / nx, Ly / ny)
steady = 0
for it in range(400):
    u_prev, h_prev = u_sol.x.array.copy(), h.x.array.copy()
    u_sol.x.array[:] = mom.solve().x.array
    speed = np.abs(u_sol.x.array.reshape(-1, 2)).sum(axis=1).max() * YEAR
    dt_yr = float(np.clip(0.5 * cell / (speed + 1e-2), 1.0, max(1.0, 0.016 * cell)))
    dt.value = dt_yr * YEAR
    h_old.x.array[:] = h.x.array
    h.x.array[:] = 0.5 * h_prev + 0.5 * adv.solve().x.array
    if not (np.all(np.isfinite(u_sol.x.array)) and np.all(np.isfinite(h.x.array))):
        raise RuntimeError("non-finite fields; check the viscosity and the inputs")
    if it == 0:
        continue
    du = np.linalg.norm(u_sol.x.array - u_prev) / (np.linalg.norm(u_sol.x.array) + 1e-30)
    dh = np.linalg.norm(h.x.array - h_prev) / (np.linalg.norm(h.x.array) + 1e-30)
    p99 = np.percentile(np.abs(h.x.array - h_prev) / (0.5 * dt_yr), 99)
    if max(du, dh) < 1e-6:
        break
    steady = steady + 1 if (du < 2e-4 and p99 < 0.1) else 0
    if steady >= 3:
        break

xs, ys = np.linspace(0.0, Lx, nx + 1), np.linspace(0.0, Ly, ny + 1)
gx, gy = np.meshgrid(xs, ys, indexing="ij")
pts = np.vstack([gx.ravel(), gy.ravel(), np.zeros(gx.size)]).T
tree = geometry.bb_tree(domain, domain.topology.dim)
colliding = geometry.compute_colliding_cells(domain, geometry.compute_collisions_points(tree, pts), pts)
cells = [colliding.links(i)[0] for i in range(pts.shape[0])]
uv = u_sol.eval(pts, cells).reshape(-1, 2) * YEAR
hg = h.eval(pts, cells).reshape(-1)
np.savez("/workspace/observations/run1.npz", x=gx, y=gy,
         u=uv[:, 0].reshape(gx.shape), v=uv[:, 1].reshape(gx.shape), h=hg.reshape(gx.shape))
print("iterations", it + 1, "du", du, "dh", dh)
```

Record with every saved file the inputs it was generated from and whether the
iteration converged; a run that stopped at the iteration cap is not steady.
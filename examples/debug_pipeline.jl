using SpinorBEC
using LinearAlgebra

println("=" ^ 60)
println("DEBUG: SpinorBEC Pipeline Step-by-Step")
println("=" ^ 60)

# ============================================================
# Test 1: Basic setup - Rb87 F=1 (simplest case)
# ============================================================
println("\n--- Test 1: Rb87 F=1 basic ground state ---")
grid1 = make_grid(GridConfig((32,), (20.0,)))
atom1 = Rb87
int1 = InteractionParams(10.0, -0.5)
pot1 = HarmonicTrap((1.0,))

gs1 = find_ground_state(;
    grid=grid1, atom=atom1, interactions=int1, potential=pot1,
    dt=0.001, n_steps=5000, tol=1e-8, initial_state=:polar,
)
println("  converged=$(gs1.converged), E=$(gs1.energy)")
psi1 = gs1.workspace.state.psi
println("  norm = $(total_norm(psi1, grid1))")
println("  psi range: $(extrema(abs.(psi1)))")
println("  density range: $(extrema(total_density(psi1, 1)))")

# ============================================================
# Test 2: Eu151 F=6, NO DDI, small p
# ============================================================
println("\n--- Test 2: Eu151 F=6, no DDI, p=10 ---")
grid2 = make_grid(GridConfig((32, 32), (20.0, 20.0)))
atom2 = Eu151
int2 = InteractionParams(378.0, 0.0)
sys2 = SpinSystem(atom2.F)
sm2 = spin_matrices(atom2.F)

psi2 = init_psi(grid2, sys2; state=:ferromagnetic)
println("  init psi shape: $(size(psi2))")
println("  init norm: $(total_norm(psi2, grid2))")
println("  init psi range: $(extrema(abs.(psi2)))")

# Check Zeeman diagonal for p=10
zee = ZeemanParams(10.0, 0.0)
zd = collect(zeeman_diagonal(zee, sys2))
println("  zeeman_diag: $(zd)")
println("  zeeman range: $(extrema(zd))")

# Check what happens in one diagonal potential step
V = evaluate_potential(HarmonicTrap((1.0, 1.0)), grid2)
println("  V_trap range: $(extrema(V))")

n_total = total_density(psi2, 2)
println("  n_total range: $(extrema(n_total))")
println("  c0*n_total range: $(extrema(378.0 .* n_total))")

# Exponent in diagonal step: -(V + zeeman + c0*n) * dt_frac
dt_frac = 0.001 / 4  # dt/4 for the nested splitting
for (c, m) in enumerate(sys2.m_values)
    max_exp = maximum(abs.((V .+ zd[c] .+ 378.0 .* n_total) .* dt_frac))
    println("  m=$m: max |exponent| = $max_exp")
end

# Actually run ground state
println("\n  Running find_ground_state (p=10, no DDI)...")
gs2 = find_ground_state(;
    grid=grid2, atom=atom2, interactions=int2,
    zeeman=ZeemanParams(10.0, 0.0),
    potential=HarmonicTrap((1.0, 1.0)),
    dt=0.001, n_steps=5000, tol=1e-8,
    initial_state=:ferromagnetic,
    enable_ddi=false,
)
println("  converged=$(gs2.converged), E=$(gs2.energy)")
psi2_gs = gs2.workspace.state.psi
println("  norm = $(total_norm(psi2_gs, grid2))")
println("  psi range: $(extrema(abs.(psi2_gs)))")

# Check component populations
pops = component_populations(psi2_gs, grid2, sys2)
println("  populations: m=$(pops.m_values) -> $(round.(pops.populations, digits=6))")

# ============================================================
# Test 3: Eu151 with DDI, small p, very small dt
# ============================================================
println("\n--- Test 3: Eu151 F=6 + DDI, p=10, dt=0.0001 ---")
ddi_params = make_ddi_params(grid2, atom2; c_dd=49.0)

# First check: what does the DDI potential look like?
plans = make_fft_plans(grid2.config.n_points)
bufs = make_ddi_buffers(grid2.config.n_points)

# Compute spin density from ground state
n_comp = sys2.n_components
n_pts = grid2.config.n_points
SpinorBEC._compute_spin_density!(bufs.Fx_r, bufs.Fy_r, bufs.Fz_r,
    psi2_gs, sm2, n_comp, 2, n_pts)
println("  Spin density Fx range: $(extrema(bufs.Fx_r))")
println("  Spin density Fy range: $(extrema(bufs.Fy_r))")
println("  Spin density Fz range: $(extrema(bufs.Fz_r))")

# Compute DDI potential
SpinorBEC.compute_ddi_potential!(ddi_params, bufs)
println("  DDI Phi_x range: $(extrema(bufs.Phi_x))")
println("  DDI Phi_y range: $(extrema(bufs.Phi_y))")
println("  DDI Phi_z range: $(extrema(bufs.Phi_z))")

# Check eigenvalue range of H_ddi at center
ic = CartesianIndex(16, 16)
phi_x = bufs.Phi_x[ic]
phi_y = bufs.Phi_y[ic]
phi_z = bufs.Phi_z[ic]
H_ddi = phi_x * sm2.Fx + phi_y * sm2.Fy + phi_z * sm2.Fz
evals = eigvals(Hermitian(Matrix(H_ddi)))
println("  H_ddi eigenvalues at center: $(round.(evals, digits=4))")

dt_ddi = 0.001 / 2  # dt_half in split step
println("  H_ddi * dt_half range: $(extrema(evals .* dt_ddi))")
println("  exp(max eigenvalue * dt_half) = $(exp(maximum(abs.(evals)) * dt_ddi))")

# Now try with DDI
println("\n  Running find_ground_state (p=10, DDI, dt=0.0002)...")
gs3 = find_ground_state(;
    grid=grid2, atom=atom2, interactions=int2,
    zeeman=ZeemanParams(10.0, 0.0),
    potential=HarmonicTrap((1.0, 1.0)),
    dt=0.0002, n_steps=10000, tol=1e-8,
    psi_init=psi2_gs,
    enable_ddi=true, c_dd=49.0,
)
println("  converged=$(gs3.converged), E=$(gs3.energy)")
println("  norm = $(total_norm(gs3.workspace.state.psi, grid2))")

# ============================================================
# Test 4: One step of real-time dynamics with DDI
# ============================================================
println("\n--- Test 4: Single real-time step with DDI ---")
sp4 = SimParams(; dt=0.001, n_steps=10, save_every=10)
ws4 = make_workspace(;
    grid=grid2, atom=atom2, interactions=int2,
    zeeman=ZeemanParams(1.0, 0.0),
    potential=HarmonicTrap((1.0, 1.0)),
    sim_params=sp4,
    psi_init=gs2.workspace.state.psi,
    enable_ddi=true, c_dd=49.0,
)
println("  initial norm: $(total_norm(ws4.state.psi, grid2))")
println("  initial E: $(total_energy(ws4))")

enable_tracing!()
reset_tracing!()
for i in 1:10
    SpinorBEC.split_step!(ws4)
    n = total_norm(ws4.state.psi, grid2)
    any_nan = any(isnan, ws4.state.psi)
    any_inf = any(isinf, ws4.state.psi)
    println("  step $i: norm=$n, NaN=$any_nan, Inf=$any_inf")
    if any_nan || any_inf
        println("  ABORT: NaN/Inf detected")
        break
    end
end
println("\n--- Timer breakdown ---")
println(TIMER)
disable_tracing!()

println("\n" * "=" ^ 60)
println("DEBUG COMPLETE")
println("=" ^ 60)

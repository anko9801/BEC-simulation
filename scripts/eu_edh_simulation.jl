"""
Reproduction of spin texture formation in Eu151 spinor dipolar BEC.

Reference: "Probing spontaneously formed spin textures in a europium
spinor dipolar gas via tilted-axis spin separation"
(Matsui et al., Institute of Science Tokyo)

Two scenarios:
  (I)  Einstein-de Haas dynamics — spin-polarized BEC at B=0, DDI drives
       spin relaxation with mass circulation formation.
  (II) Flower phase ground state — imaginary-time evolution at B=0 with
       flower-ansatz initial state.

Dimensionless units: ℏ = m = ω_⊥ = 1
  length  → a_ho = √(ℏ/(m ω_⊥))
  energy  → ℏ ω_⊥
  time    → 1/ω_⊥
"""

using SpinorBEC
using SpinorBEC: _component_slice
using LinearAlgebra
using JLD2

# =================================================================
# Physical parameters
# =================================================================

const ω_perp = 2π * 100.0        # radial trap frequency [rad/s]
const λ_z    = 0.5               # ω_z / ω_⊥
const N_atoms = 15_000

# Derived scales (SI, for reference only)
const m_Eu  = Eu151.mass
const a_ho  = sqrt(Units.HBAR / (m_Eu * ω_perp))      # [m]
const a_z   = a_ho / sqrt(λ_z)                          # [m]
const t_unit = 1.0 / ω_perp                             # [s]

println("a_ho  = $(round(a_ho * 1e6; digits=3)) μm")
println("a_z   = $(round(a_z  * 1e6; digits=3)) μm")
println("t_unit = $(round(t_unit * 1e3; digits=2)) ms")

# =================================================================
# Dimensionless parameters
# =================================================================

# Contact interaction (c1 = 0 for Eu; DDI is the spin-dependent part)
a_s_dl = Eu151.a0 / a_ho                             # dimensionless scattering length
c0_3D  = 4π * a_s_dl * N_atoms
c0     = c0_3D / (sqrt(2π) * a_z / a_ho)             # quasi-2D reduction

# DDI coupling
c_dd_SI       = compute_c_dd(Eu151)                   # μ₀μ²/(4π) [J·m³]
c_dd_per_atom = c_dd_SI / (Units.HBAR * ω_perp * a_ho^3)
c_dd_3D       = N_atoms * c_dd_per_atom
c_dd          = c_dd_3D / (sqrt(2π) * a_z / a_ho)    # quasi-2D

ε_dd = compute_a_dd(Eu151) / Eu151.a0

println("\nε_dd = $(round(ε_dd; digits=3))")
println("c0   = $(round(c0;   digits=1))  (2D, dimensionless)")
println("c_dd = $(round(c_dd; digits=1))  (2D, dimensionless)")

# =================================================================
# Grid & atom
# =================================================================

const N_GRID = 64
const L_BOX  = 20.0     # [a_ho]

grid = make_grid(GridConfig((N_GRID, N_GRID), (L_BOX, L_BOX)))

# Dimensionless atom for the solver (mass doesn't enter the propagator)
atom = AtomSpecies("Eu151", 1.0, 6, a_s_dl, 0.0)

interactions = InteractionParams(c0, 0.0)
trap = HarmonicTrap(1.0, 1.0)

# =================================================================
# Helper: flower-ansatz initial state
# =================================================================

"""
Create initial state with flower-phase vortex topology.
Component m gets phase winding (F − m)φ.
"""
function init_flower_ansatz(grid, sys; seed_amp=0.01)
    F = sys.F
    n_comp = sys.n_components
    n_pts = grid.config.n_points
    psi = zeros(ComplexF64, n_pts..., n_comp)

    σ = grid.config.box_size[1] / 8

    for j in 1:n_pts[2], i in 1:n_pts[1]
        x, y = grid.x[1][i], grid.x[2][j]
        r   = sqrt(x^2 + y^2)
        φ   = atan(y, x)
        env = exp(-(x^2 + y^2) / (2σ^2))

        for (c, m) in enumerate(sys.m_values)
            w = F - m   # winding number
            if w == 0
                psi[i, j, c] = env
            else
                r_core = 1.0
                core = r^abs(w) / (r^abs(w) + r_core^abs(w))
                psi[i, j, c] = seed_amp * env * core * exp(im * w * φ)
            end
        end
    end

    dV = cell_volume(grid)
    psi ./= sqrt(sum(abs2, psi) * dV)
    psi
end

# =================================================================
# (I) Einstein-de Haas dynamics
# =================================================================

function run_edh(; dt=0.001, t_total=2.0, n_save=100)
    println("\n" * "="^60)
    println("  (I) Einstein-de Haas dynamics")
    println("="^60)

    sys = SpinSystem(6)
    psi0 = init_psi(grid, sys; state=:ferromagnetic)

    # Small random perturbation to break rotational symmetry
    for c in 2:sys.n_components
        view(psi0, :, :, c) .+= 0.001 .* randn(ComplexF64, N_GRID, N_GRID)
    end
    dV = cell_volume(grid)
    psi0 ./= sqrt(sum(abs2, psi0) * dV)

    n_steps = round(Int, t_total / dt)
    save_every = max(1, n_steps ÷ n_save)
    sp = SimParams(; dt, n_steps, imaginary_time=false, save_every)

    ws = make_workspace(;
        grid, atom, interactions,
        potential=trap,
        zeeman=ZeemanParams(0.0, 0.0),
        sim_params=sp,
        psi_init=psi0,
        enable_ddi=true,
        c_dd,
    )

    println("Running $(n_steps) steps (dt=$dt, t_total=$t_total ω⁻¹) ...")
    println("  Physical time: $(round(t_total * t_unit * 1e3; digits=2)) ms")

    sm = ws.spin_matrices

    result = run_simulation!(ws;
        callback=(ws, step) -> begin
            if step % max(1, n_steps ÷ 10) == 0
                Mz = magnetization(ws.state.psi, ws.grid, sm.system)
                fx, fy, _ = spin_density_vector(ws.state.psi, sm, 2)
                fxy_max = maximum(sqrt.(fx .^ 2 .+ fy .^ 2))
                t_ms = round(ws.state.t * t_unit * 1e3; digits=3)
                println("  t=$(t_ms) ms  Mz=$(round(Mz; digits=3))  max|Fxy|=$(round(fxy_max; sigdigits=3))")
            end
        end,
    )

    # Final observables
    psi_f = result.psi_snapshots[end]
    fx, fy, fz = spin_density_vector(psi_f, sm, 2)
    n_total = total_density(psi_f, 2)

    jldsave("eu_edh_results.jld2";
        psi_snapshots=result.psi_snapshots,
        times=result.times,
        energies=result.energies,
        magnetizations=result.magnetizations,
        norms=result.norms,
        grid_n=N_GRID, grid_L=L_BOX,
        spin_density_x=fx, spin_density_y=fy, spin_density_z=fz,
        density=n_total,
    )
    println("\nSaved → eu_edh_results.jld2")
    println("Final Mz = $(round(result.magnetizations[end]; digits=3))")

    result
end

# =================================================================
# (II) Flower phase ground state
# =================================================================

function run_flower_ground_state(; dt=0.001, n_steps=50_000, tol=1e-10)
    println("\n" * "="^60)
    println("  (II) Flower phase ground state (imaginary time)")
    println("="^60)

    sys = SpinSystem(6)
    psi0 = init_flower_ansatz(grid, sys; seed_amp=0.05)

    sp = SimParams(; dt, n_steps, imaginary_time=true, normalize_every=1,
                   save_every=max(1, n_steps ÷ 20))

    ws = make_workspace(;
        grid, atom, interactions,
        potential=trap,
        zeeman=ZeemanParams(0.0, 0.0),
        sim_params=sp,
        psi_init=psi0,
        enable_ddi=true,
        c_dd,
    )

    println("Running imaginary-time evolution ($n_steps steps, dt=$dt) ...")

    sm = ws.spin_matrices
    E_prev = total_energy(ws)
    converged = false

    for step in 1:n_steps
        split_step!(ws)

        if step % sp.save_every == 0
            E = total_energy(ws)
            dE = abs(E - E_prev)
            Mz = magnetization(ws.state.psi, ws.grid, sm.system)
            println("  step $step: E=$(round(E; sigdigits=8))  dE=$(round(dE; sigdigits=3))  Mz=$(round(Mz; digits=3))")
            if dE < tol
                converged = true
                println("  Converged!")
                break
            end
            E_prev = E
        end
    end

    # Analyze spin texture
    psi_f = ws.state.psi
    fx, fy, fz = spin_density_vector(psi_f, sm, 2)
    n_total = total_density(psi_f, 2)

    # Component populations
    pops = [sum(abs2, view(psi_f, :, :, c)) * cell_volume(grid) for c in 1:sm.system.n_components]

    println("\nComponent populations (m = +6 to -6):")
    for (c, m) in enumerate(sm.system.m_values)
        pop = round(pops[c]; sigdigits=3)
        println("  m=$(lpad(m, 3)): $pop")
    end

    Lz = sum((6 - m) * pops[c] for (c, m) in enumerate(sm.system.m_values))
    println("\nOrbital angular momentum Lz/N = $(round(Lz; digits=3))")

    jldsave("eu_flower_results.jld2";
        psi=psi_f,
        energy=total_energy(ws),
        converged,
        spin_density_x=fx, spin_density_y=fy, spin_density_z=fz,
        density=n_total,
        populations=pops,
        grid_n=N_GRID, grid_L=L_BOX,
    )
    println("Saved → eu_flower_results.jld2")

    ws
end

# =================================================================
# Run
# =================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    # (I) Einstein-de Haas dynamics
    edh_result = run_edh(; dt=0.001, t_total=2.0)

    # (II) Flower phase ground state
    flower_ws = run_flower_ground_state(; dt=0.001, n_steps=50_000)
end

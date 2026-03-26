"""
Reproduction of spin texture formation in Eu151 spinor dipolar BEC.

Reference: Matsui et al., Science 391, 384-388 (2026)
           arXiv:2504.17357

Two scenarios:
  (I)  Einstein-de Haas dynamics — quench to B_weak, DDI drives spin relaxation
  (II) Flower phase ground state — imaginary-time at B ≈ 0
"""

include(joinpath(@__DIR__, "eu151_setup.jl"))
using SpinorBEC: _component_slice
using LinearAlgebra

const c1 = EU_c0 / 36  # antiferromagnetic (Matsui: c₁/c₀ = 1/36)

println("a_ho    = $(round(EU_a_ho * 1e6; digits=3)) μm")
println("t_unit  = $(round(EU_t_unit * 1e3; digits=3)) ms")
println("ε_dd    = $(round(EU_ε_dd; digits=3))")
println("c0      = $(round(EU_c0; digits=1))  (3D)")
println("c1      = $(round(c1; digits=1))  (3D, c0/36)")
println("c_dd    = $(round(EU_c_dd; digits=1))  (3D)")
println("p_weak  = $(round(EU_p_weak; digits=3))  (B = 2.6 nT)")

# =================================================================
# Grid & atom
# =================================================================

const N_GRID = 32
const L_BOX  = 20.0

grid = make_grid(GridConfig((N_GRID, N_GRID, N_GRID), (L_BOX, L_BOX, L_BOX)))
atom = AtomSpecies("Eu151", 1.0, 6, EU_a_s_dl, 0.0)
interactions = InteractionParams(EU_c0, c1)
trap = HarmonicTrap((1.0, 1.0, EU_λ_z))

# =================================================================
# Helper: flower-ansatz initial state (3D)
# =================================================================

"""
Create initial state with flower-phase vortex topology.
Component m gets phase winding (F − m)φ in the xy-plane.
"""
function init_flower_ansatz(grid, sys; seed_amp=0.01)
    F = sys.F
    n_comp = sys.n_components
    n_pts = grid.config.n_points
    psi = zeros(ComplexF64, n_pts..., n_comp)

    σ_xy = grid.config.box_size[1] / 8
    σ_z  = grid.config.box_size[3] / 8

    for k in 1:n_pts[3], j in 1:n_pts[2], i in 1:n_pts[1]
        x, y, z = grid.x[1][i], grid.x[2][j], grid.x[3][k]
        r   = sqrt(x^2 + y^2)
        φ   = atan(y, x)
        env = exp(-(x^2 + y^2) / (2σ_xy^2) - z^2 / (2σ_z^2))

        for (c, m) in enumerate(sys.m_values)
            w = F - m
            if w == 0
                psi[i, j, k, c] = env
            else
                r_core = 1.0
                core = r^abs(w) / (r^abs(w) + r_core^abs(w))
                psi[i, j, k, c] = seed_amp * env * core * exp(im * w * φ)
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

function run_edh(; dt=0.001, t_total_ms=40.0, n_save=100)
    println("\n" * "="^60)
    println("  (I) Einstein-de Haas dynamics (3D)")
    println("="^60)

    t_total = t_total_ms * 1e-3 / EU_t_unit
    println("  Target: $(t_total_ms) ms = $(round(t_total; digits=1)) ω⁻¹")

    sys = SpinSystem(6)

    psi_gs = load_or_compute_gs(grid; trap)
    psi0 = seed_noise(psi_gs, sys.n_components, 3, grid)

    n_steps = round(Int, t_total / dt)
    save_every = max(1, n_steps ÷ n_save)
    sp = SimParams(; dt, n_steps, imaginary_time=false, save_every)

    ws = make_workspace(;
        grid, atom, interactions,
        potential=trap,
        zeeman=ZeemanParams(EU_p_weak, 0.0),
        sim_params=sp,
        psi_init=psi0,
        enable_ddi=true,
        c_dd=EU_c_dd,
    )

    println("  Running $(n_steps) steps (dt=$dt, p=$(round(EU_p_weak; digits=3)))...")
    sm = ws.spin_matrices

    enable_tracing!()
    reset_tracing!()
    result = run_simulation!(ws;
        callback=(ws, step) -> begin
            if step % max(1, n_steps ÷ 10) == 0
                Mz = magnetization(ws.state.psi, ws.grid, sm.system)
                t_ms = round(ws.state.t * EU_t_unit * 1e3; digits=1)
                println("  t=$(t_ms) ms  Mz=$(round(Mz; digits=3))")
            end
        end,
    )

    println("\n--- Timer breakdown ---")
    println(TIMER)
    disable_tracing!()

    psi_f = result.psi_snapshots[end]
    fx, fy, fz = spin_density_vector(psi_f, sm, 3)
    n_total = total_density(psi_f, 3)

    jldsave("eu_edh_results_3d.jld2";
        psi_snapshots=result.psi_snapshots,
        times=result.times,
        energies=result.energies,
        magnetizations=result.magnetizations,
        norms=result.norms,
        grid_n=N_GRID, grid_L=L_BOX,
        spin_density_x=fx, spin_density_y=fy, spin_density_z=fz,
        density=n_total,
    )
    println("\nSaved → eu_edh_results_3d.jld2")
    println("Final Mz = $(round(result.magnetizations[end]; digits=3))")

    result
end

# =================================================================
# (II) Flower phase ground state
# =================================================================

function run_flower_ground_state(; dt=0.001, n_steps=50_000, tol=1e-10)
    println("\n" * "="^60)
    println("  (II) Flower phase ground state (3D, imaginary time)")
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
        c_dd=EU_c_dd,
    )

    println("  Running imaginary-time evolution ($n_steps steps, dt=$dt)...")

    sm = ws.spin_matrices
    E_prev = total_energy(ws)
    converged = false

    enable_tracing!()
    reset_tracing!()
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

    println("\n--- Timer breakdown ---")
    println(TIMER)
    disable_tracing!()

    psi_f = ws.state.psi
    fx, fy, fz = spin_density_vector(psi_f, sm, 3)
    n_total = total_density(psi_f, 3)

    pops = [sum(abs2, view(psi_f, :, :, :, c)) * cell_volume(grid) for c in 1:sm.system.n_components]

    println("\nComponent populations (m = +6 to -6):")
    for (c, m) in enumerate(sm.system.m_values)
        pop = round(pops[c]; sigdigits=3)
        println("  m=$(lpad(m, 3)): $pop")
    end

    Lz = sum((6 - m) * pops[c] for (c, m) in enumerate(sm.system.m_values))
    println("\nOrbital angular momentum Lz/N = $(round(Lz; digits=3))")

    jldsave("eu_flower_results_3d.jld2";
        psi=psi_f,
        energy=total_energy(ws),
        converged,
        spin_density_x=fx, spin_density_y=fy, spin_density_z=fz,
        density=n_total,
        populations=pops,
        grid_n=N_GRID, grid_L=L_BOX,
    )
    println("Saved → eu_flower_results_3d.jld2")

    ws
end

# =================================================================
# Run
# =================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    edh_result = run_edh(; dt=0.001, t_total_ms=40.0)
end

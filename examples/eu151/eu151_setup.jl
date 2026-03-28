include(joinpath(@__DIR__, "eu151_params.jl"))
using JLD2

function load_or_compute_gs(grid; cache_suffix="", trap=HarmonicTrap((1.0, 1.0, EU_λ_z)))
    N_GRID = grid.config.n_points[1]
    gs_cache = joinpath(@__DIR__, "cache_eu151_gs_3d_$(N_GRID)$(cache_suffix).jld2")
    if isfile(gs_cache)
        println("Loading cached ground state from $gs_cache")
        return load(gs_cache, "psi")
    end
    println("Finding ground state (ITP, no DDI)...")
    atom = AtomSpecies("Eu151", 1.0, 6, EU_a_s_dl, 0.0)
    gs = find_ground_state(;
        grid, atom, interactions=InteractionParams(EU_c0, 0.0),
        zeeman=ZeemanParams(100.0, 0.0), potential=trap,
        dt=0.005, n_steps=20000, tol=1e-9,
        initial_state=:ferromagnetic, enable_ddi=false,
    )
    println("  converged=$(gs.converged), E=$(gs.energy)")
    psi_out = copy(gs.workspace.state.psi)
    jldsave(gs_cache; psi=psi_out)
    println("  cached → $gs_cache")
    psi_out
end

include(joinpath(@__DIR__, "eu151_params.jl"))
using Printf, JLD2, FFTW, Random

N_GRID = 64
grid = make_grid(GridConfig((N_GRID,N_GRID,N_GRID), (20.0,20.0,20.0)))
atom = AtomSpecies("Eu151", 1.0, 6, EU_a_s_dl, 0.0)
sys = SpinSystem(6)
sm = spin_matrices(6)
dV = cell_volume(grid)
D = 13
plans = make_fft_plans(grid.config.n_points; flags=FFTW.MEASURE)

psi_gs = load(joinpath(@__DIR__, "cache_eu151_gs_3d_64.jld2"), "psi")
psi = copy(psi_gs)
Random.seed!(42)
SpinorBEC._add_noise!(psi, 0.001, D, 3, grid)

n_dens = SpinorBEC.total_density(psi, 3)
n_peak = maximum(n_dens)

dt = 5e-4
n_steps = 400  # t_final = 0.2 ω⁻¹ ≈ 0.29 ms
θ_est = EU_c_dd * n_peak * 6 * dt / 2

@printf("Peak density: %.4f\n", n_peak)
@printf("Estimated θ/step: %.4f rad\n", θ_est)
@printf("dt=%.1e, %d steps, t_final=%.3f ω⁻¹ (%.2f ms)\n", dt, n_steps, dt*n_steps, dt*n_steps*EU_t_unit*1e3)

sp = SimParams(; dt, n_steps=1)
ws = make_workspace(; grid, atom, interactions=InteractionParams(EU_c0, 0.0),
    zeeman=ZeemanParams(EU_p_weak, 0.0), potential=HarmonicTrap((1.0,1.0,EU_λ_z)),
    sim_params=sp, psi_init=psi, enable_ddi=true, c_dd=EU_c_dd,
    fft_flags=FFTW.MEASURE)

Sz0 = magnetization(psi, grid, sys)
Lz0 = orbital_angular_momentum(psi, grid, plans)
Jz0 = Sz0 + Lz0
@printf("\nInitial: Sz=%.4f Lz=%.4f Jz=%.4f\n\n", Sz0, Lz0, Jz0)

@printf("%6s | %8s | %8s | %8s | %8s | %8s | %8s\n",
    "step", "t(ms)", "Sz", "Lz", "Jz_drift", "m=+6", "m=+5")
println("-"^72)

# Warmup
for _ in 1:3; split_step!(ws); end

t0 = time()
for step in 4:(n_steps+3)
    split_step!(ws)
    if (step-3) % 50 == 0
        t_ms = ws.state.t * EU_t_unit * 1e3
        Sz = magnetization(ws.state.psi, grid, sys)
        Lz = orbital_angular_momentum(ws.state.psi, grid, plans)
        Jz = Sz + Lz
        pop6 = sum(abs2, @view(ws.state.psi[:,:,:,1])) * dV
        pop5 = sum(abs2, @view(ws.state.psi[:,:,:,2])) * dV
        @printf("%6d | %8.4f | %8.4f | %8.4f | %+8.5f | %8.5f | %8.5f\n",
            step-3, t_ms, Sz, Lz, Jz-Jz0, pop6, pop5)
    end
end
wall = time() - t0
@printf("\n%d steps in %.1fs (%.1f ms/step)\n", n_steps, wall, wall/n_steps*1000)

println("\nFinal populations:")
for c in 1:D
    m = 6 - (c-1)
    pop = sum(abs2, @view(ws.state.psi[:,:,:,c])) * dV
    bar = repeat("█", round(Int, pop * 80))
    @printf("  m=%+3d: %.5f %s\n", m, pop, bar)
end

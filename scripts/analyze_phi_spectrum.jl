include(joinpath(@__DIR__, "eu151_setup.jl"))

println("=== φ-spectrum analysis of Eu151 EdH dynamics ===\n")

N_GRID = parse(Int, get(ENV, "PHI_GRID", "32"))
grid = make_grid(GridConfig((N_GRID, N_GRID, N_GRID), (20.0, 20.0, 20.0)))
atom = AtomSpecies("Eu151", 1.0, 6, EU_a_s_dl, 0.0)
sys = SpinSystem(atom.F)
n_comp = sys.n_components
dV = cell_volume(grid)

# Load GS and seed noise
psi_gs = load(joinpath(@__DIR__, "cache_eu151_gs_3d_$(N_GRID).jld2"), "psi")
psi = seed_noise(psi_gs, n_comp, 3, grid)

# Run 2ms with snapshots every 0.25 ω⁻¹
t_end = 2e-3 / EU_t_unit
sp = SimParams(; dt=0.001, n_steps=1)
ws = make_workspace(; grid, atom, interactions=InteractionParams(EU_c0, 0.0),
    zeeman=ZeemanParams(EU_p_weak, 0.0), potential=HarmonicTrap((1.0, 1.0, EU_λ_z)),
    sim_params=sp, psi_init=psi, enable_ddi=true, c_dd=EU_c_dd)

adaptive = AdaptiveDtParams(dt_init=0.002, dt_min=0.0001, dt_max=0.005, tol=0.001)
enable_tracing!()
reset_tracing!()
println("Running 2ms dynamics...")
out = run_simulation_adaptive!(ws; adaptive, t_end, save_interval=0.25)
println("  done: $(out.n_accepted) steps\n")
println(TIMER)
disable_tracing!()

# --- φ-spectrum analysis ---
x = collect(grid.x[1])
y = collect(grid.x[2])
nx, ny, nz = grid.config.n_points

rho_xy = [sqrt(x[i]^2 + y[j]^2) for i in 1:nx, j in 1:ny]
phi_xy = [atan(y[j], x[i]) for i in 1:nx, j in 1:ny]

# Radial bins (width = grid spacing)
dr = grid.dx[1]
n_rbins = ceil(Int, maximum(rho_xy) / dr) + 1

bin_lists = [Tuple{Int,Int}[] for _ in 1:n_rbins]
for j in 1:ny, i in 1:nx
    b = clamp(floor(Int, rho_xy[i, j] / dr) + 1, 1, n_rbins)
    push!(bin_lists[b], (i, j))
end

# Angular mode range: J_z up to F=6, so n up to ~12
n_max = 12
n_range = -n_max:n_max
N_n = length(n_range)

println("Radial bins: $n_rbins (dr=$(round(dr, digits=3)))")
println("Angular modes: n = $(-n_max) to $n_max\n")

snapshots = out.result.psi_snapshots
times = out.result.times

for (si, snap) in enumerate(snapshots)
    t_ms = round(times[si] * EU_t_unit * 1e3, digits=2)
    println("t = $t_ms ms")
    println("  m_F   pop     dom_n  weight   top modes")
    println("  " * "-"^55)

    for c in 1:n_comp
        m_F = sys.F - (c - 1)
        psi_c = view(snap, :, :, :, c)
        pop = sum(abs2, psi_c) * dV
        pop < 1e-5 && continue

        # P_n = Σ_z Σ_ρ |f_n(ρ,z)|² ρ Δρ Δz
        P_n = zeros(N_n)

        for iz in 1:nz
            for (rb, pts) in enumerate(bin_lists)
                isempty(pts) && continue
                rho_mid = (rb - 0.5) * dr
                n_pts_bin = length(pts)

                for (ni, n) in enumerate(n_range)
                    coeff = zero(ComplexF64)
                    for (ix, iy) in pts
                        coeff += psi_c[ix, iy, iz] * cis(-n * phi_xy[ix, iy])
                    end
                    coeff /= n_pts_bin
                    P_n[ni] += abs2(coeff) * rho_mid * dr * grid.dx[3]
                end
            end
        end

        total = sum(P_n)
        total < 1e-30 && continue
        P_n ./= total

        si_sorted = sortperm(P_n; rev=true)
        dom_n = n_range[si_sorted[1]]
        dom_w = P_n[si_sorted[1]]

        top = String[]
        for k in 1:min(5, length(si_sorted))
            w = P_n[si_sorted[k]]
            w < 0.005 && break
            push!(top, "n=$(n_range[si_sorted[k]])($(round(w * 100; digits=1))%)")
        end

        println("  $(lpad(m_F, 3))  $(lpad(round(pop; digits=4), 7))  n=$(lpad(dom_n, 3))  $(lpad(round(dom_w * 100; digits=1), 5))%   $(join(top, ", "))")
    end
    println()
end

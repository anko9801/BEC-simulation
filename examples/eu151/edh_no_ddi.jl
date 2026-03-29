include(joinpath(@__DIR__, "eu151_params.jl"))
using Printf, Random, FFTW, JLD2

function run_edh_no_ddi()
    println("=" ^ 70)
    println("  Eu151 EdH — No DDI (c₁-only dynamics)")
    println("=" ^ 70)

    N_GRID = 64
    BOX = 20.0
    C1_RATIO = 1.0 / 36.0

    grid = make_grid(GridConfig(ntuple(_ -> N_GRID, 3), ntuple(_ -> BOX, 3)))
    atom = AtomSpecies("Eu151", 1.0, 6, EU_a_s_dl, 0.0)
    trap = HarmonicTrap((1.0, 1.0, EU_λ_z))
    interactions = interaction_params_from_constraint(;
        c_total=EU_c_total, c1_ratio=C1_RATIO, F=6)

    D = 13
    dV = cell_volume(grid)
    sys = SpinSystem(6)

    @printf("c0=%.1f, c1=%.1f, c_dd=0 (disabled), p=%.4f\n",
        interactions.c0, interactions.c1, EU_p_weak)
    @printf("c1 × n_peak ~ %.2f ω → spin mixing timescale ~ %.0f μs\n",
        interactions.c1 * 4e-3, 1.0 / (interactions.c1 * 4e-3) * EU_t_unit * 1e6)

    cache_file = joinpath(@__DIR__, "cache_eu151_gs_3d_$(N_GRID).jld2")
    psi_gs = if isfile(cache_file)
        println("Loading cached ground state")
        load(cache_file, "psi")
    else
        println("Computing ground state...")
        gs = find_ground_state(;
            grid, atom,
            interactions=InteractionParams(EU_c_total, 0.0),
            zeeman=ZeemanParams(100.0, 0.0),
            potential=trap,
            dt=0.005, n_steps=20000, tol=1e-9,
            initial_state=:ferromagnetic,
            enable_ddi=false,
            fft_flags=FFTW.MEASURE,
        )
        save(cache_file, "psi", gs.workspace.state.psi)
        gs.workspace.state.psi
    end

    psi = copy(psi_gs)
    Random.seed!(42)
    SpinorBEC._add_noise!(psi, 0.001, D, 3, grid)

    dt = 0.005
    t_final_dimless = 27.6  # 40 ms
    n_total = round(Int, t_final_dimless / dt)

    snapshot_times_ms = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0]
    snapshot_steps = [max(0, round(Int, t * 1e-3 / (EU_t_unit * dt))) for t in snapshot_times_ms]

    @printf("dt=%.3f, %d steps, t_final=40ms (%.1f ω⁻¹)\n\n", dt, n_total, t_final_dimless)

    sp = SimParams(; dt, n_steps=1)
    ws = make_workspace(;
        grid, atom, interactions,
        zeeman=ZeemanParams(EU_p_weak, 0.0),
        potential=trap,
        sim_params=sp,
        psi_init=psi,
        enable_ddi=false,
        fft_flags=FFTW.MEASURE,
    )

    sm = ws.spin_matrices
    snapshots = Dict{Float64, Any}()
    cx = N_GRID ÷ 2 + 1

    function save_snapshot!(label_ms)
        psi_now = ws.state.psi
        n_tot = SpinorBEC.total_density(psi_now, 3)
        pops = [sum(abs2, @view(psi_now[:,:,:,c])) * dV for c in 1:D]
        Sz = magnetization(psi_now, grid, sys)
        fx, fy, fz = spin_density_vector(psi_now, sm, 3)

        comp_col_xy = [dropdims(sum(SpinorBEC.component_density(psi_now, 3, c), dims=3), dims=3) .* grid.dx[3] for c in 1:D]
        comp_n_x = [SpinorBEC.component_density(psi_now, 3, c)[:, cx, cx] for c in 1:D]

        # Magnetization column densities (integrated along z)
        mz_col_xy = dropdims(sum(fz, dims=3), dims=3) .* grid.dx[3]
        mx_col_xy = dropdims(sum(fx, dims=3), dims=3) .* grid.dx[3]

        snapshots[label_ms] = (
            n_xy = dropdims(sum(n_tot, dims=3), dims=3) .* grid.dx[3],
            n_xz = dropdims(sum(n_tot, dims=2), dims=2) .* grid.dx[2],
            slice_xy = n_tot[:, :, cx],
            slice_xz = n_tot[:, cx, :],
            n_x = n_tot[:, cx, cx],
            n_z = n_tot[cx, cx, :],
            pops = pops,
            Sz = Sz,
            norm = sum(abs2, psi_now) * dV,
            comp_col_xy = comp_col_xy,
            comp_n_x = comp_n_x,
            mz_col_xy = mz_col_xy,
            mx_col_xy = mx_col_xy,
        )
    end

    @printf("%6s | %6s | %6s %6s %6s %6s | %+7s\n",
        "step", "t(ms)", "P(+6)", "P(+5)", "P(+4)", "P(0)", "Sz")
    println("-" ^ 65)

    save_snapshot!(0.0)
    s = snapshots[0.0]
    @printf("%6d | %6.1f | %.4f %.4f %.4f %.4f | %+7.3f\n",
        0, 0.0, s.pops[1], s.pops[2], s.pops[3], s.pops[7], s.Sz)

    for _ in 1:3; split_step!(ws); end

    t0 = time()
    print_every = max(1, n_total ÷ 20)
    next_snap_idx = 2

    for step in 4:(n_total + 3)
        split_step!(ws)
        actual_step = step - 3

        if next_snap_idx <= length(snapshot_steps) && actual_step >= snapshot_steps[next_snap_idx]
            t_ms = snapshot_times_ms[next_snap_idx]
            save_snapshot!(t_ms)
            s = snapshots[t_ms]
            @printf("%6d | %6.1f | %.4f %.4f %.4f %.4f | %+7.3f\n",
                actual_step, t_ms, s.pops[1], s.pops[2], s.pops[3], s.pops[7], s.Sz)
            next_snap_idx += 1
        elseif actual_step % print_every == 0
            t_ms = actual_step * dt * EU_t_unit * 1e3
            p6 = sum(abs2, @view(ws.state.psi[:,:,:,1])) * dV
            Sz = magnetization(ws.state.psi, grid, sys)
            @printf("%6d | %6.1f |  %.4f                   | %+7.3f\n",
                actual_step, t_ms, p6, Sz)
        end
    end
    wall = time() - t0
    @printf("\n%d steps in %.0fs (%.2fs/step)\n", n_total, wall, wall / n_total)

    outfile = joinpath(@__DIR__, "edh_no_ddi_64.jld2")
    jldsave(outfile;
        snapshots=snapshots,
        snapshot_times_ms=snapshot_times_ms,
        x_coords=collect(grid.x[1]),
        z_coords=collect(grid.x[3]),
        grid_N=N_GRID, grid_box=BOX,
        c0=interactions.c0, c1=interactions.c1, c_dd=0.0,
    )
    @printf("Saved → %s\n", outfile)

    println("\nFinal populations:")
    last_t = maximum(keys(snapshots))
    sf = snapshots[last_t]
    for c in 1:D
        m = 6 - (c - 1)
        bar = repeat("█", round(Int, sf.pops[c] * 50))
        @printf("  m=%+3d: %.4f %s\n", m, sf.pops[c], bar)
    end
end

run_edh_no_ddi()

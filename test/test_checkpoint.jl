@testset "Checkpoint/Restart" begin
    F = 1
    grid = make_grid(GridConfig((16,), (10.0,)))
    sys = SpinSystem(F)
    atom = Rb87
    interactions = InteractionParams(1.0, 0.0)
    dt = 0.01

    @testset "Checkpoint creates files" begin
        dir = mktempdir()
        sp = SimParams(; dt=dt, n_steps=100, save_every=50)
        ws = make_workspace(; grid, atom, interactions, sim_params=sp)

        run_simulation_checkpointed!(ws;
            checkpoint_dir=dir, checkpoint_every=50)

        files = filter(f -> endswith(f, ".jld2"), readdir(dir))
        @test length(files) >= 1
    end

    @testset "Resume produces same result as continuous run" begin
        dir = mktempdir()

        # Full continuous run
        sp_full = SimParams(; dt=dt, n_steps=100, save_every=100)
        ws_full = make_workspace(; grid, atom, interactions, sim_params=sp_full)
        psi0 = copy(ws_full.state.psi)
        run_simulation!(ws_full)
        psi_full = copy(ws_full.state.psi)

        # Run first 50 steps with checkpoint
        sp1 = SimParams(; dt=dt, n_steps=50, save_every=50)
        ws1 = make_workspace(; grid, atom, interactions, sim_params=sp1,
                             psi_init=copy(psi0))
        run_simulation_checkpointed!(ws1;
            checkpoint_dir=dir, checkpoint_every=50)

        # Resume for remaining 50 steps
        sp2 = SimParams(; dt=dt, n_steps=100, save_every=50)
        ws2 = make_workspace(; grid, atom, interactions, sim_params=sp2,
                             psi_init=copy(psi0))
        # Manually set state to match checkpoint
        data = load_state(joinpath(dir, readdir(dir)[end]))
        copyto!(ws2.state.psi, data.psi)
        ws2.state.t = data.t
        ws2.state.step = data.step

        sp_remain = SimParams(; dt=dt, n_steps=50, save_every=50)
        ws2_remain = make_workspace(; grid, atom, interactions, sim_params=sp_remain,
                                    psi_init=copy(ws2.state.psi))
        ws2_remain.state.t = ws2.state.t
        run_simulation!(ws2_remain)

        @test maximum(abs, ws2_remain.state.psi .- psi_full) < 1e-10
    end

    @testset "Checkpoint with callback" begin
        dir = mktempdir()
        sp = SimParams(; dt=dt, n_steps=50, save_every=25)
        ws = make_workspace(; grid, atom, interactions, sim_params=sp)

        cb_count = Ref(0)
        run_simulation_checkpointed!(ws;
            checkpoint_dir=dir, checkpoint_every=25,
            callback=(_, _) -> cb_count[] += 1)

        @test cb_count[] == 2  # save_every=25, n_steps=50 → 2 callbacks
    end
end

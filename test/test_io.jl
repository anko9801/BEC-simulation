using Test
using SpinorBEC

@testset "save_state / load_state round-trip" begin
    gc = GridConfig((32,), (10.0,))
    grid = make_grid(gc)
    atom = AtomSpecies("test-io", 1.0, 1, 0.0, 0.0)
    ip = InteractionParams(100.0, -0.5, 0.1, [0.0, 0.0, 5.0])
    zee = ZeemanParams(0.3, 0.8)

    sp = SimParams(; dt=0.002, n_steps=100, imaginary_time=false, save_every=100)
    ws = make_workspace(;
        grid, atom,
        interactions=ip,
        zeeman=zee,
        potential=HarmonicTrap(1.0),
        sim_params=sp,
    )

    # Evolve a few steps so t > 0
    for _ in 1:10
        split_step!(ws)
    end

    fname = tempname() * ".jld2"
    try
        save_state(fname, ws)
        data = load_state(fname)

        @test data.atom_name == "test-io"
        @test data.grid_n_points == (32,)
        @test data.grid_box_size == (10.0,)
        @test data.t == ws.state.t
        @test data.step == ws.state.step
        @test data.psi ≈ ws.state.psi

        @test data.c0 == 100.0
        @test data.c1 == -0.5
        @test data.c_lhy == 0.1
        @test data.c_extra == [0.0, 0.0, 5.0]
        @test data.zeeman_p == 0.3
        @test data.zeeman_q == 0.8
        @test data.c_dd == 0.0
        @test data.dt == 0.002
        @test data.imaginary_time == false
    finally
        rm(fname; force=true)
    end
end

@testset "save_state with DDI" begin
    gc = GridConfig((32,), (10.0,))
    grid = make_grid(gc)
    atom = AtomSpecies("test-io-ddi", 1.0, 1, 0.0, 0.0)

    sp = SimParams(; dt=0.001, n_steps=10, imaginary_time=false, save_every=10)
    ws = make_workspace(;
        grid, atom,
        interactions=InteractionParams(10.0, 0.0),
        potential=NoPotential(),
        sim_params=sp,
        enable_ddi=true, c_dd=50.0,
    )

    fname = tempname() * ".jld2"
    try
        save_state(fname, ws)
        data = load_state(fname)
        @test data.c_dd == 50.0
    finally
        rm(fname; force=true)
    end
end

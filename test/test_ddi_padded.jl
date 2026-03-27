@testset "DDI Zero-Padding" begin
    @testset "make_ddi_padded creates correct shapes" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        ctx = make_ddi_padded(grid, Rb87; c_dd=1.0)
        @test ctx.padded_shape == (128,)
        @test size(ctx.Q_xx) == rfft_output_shape((128,))
        @test size(ctx.Fx_pad) == (128,)
    end

    @testset "make_ddi_padded 2D" begin
        config = GridConfig((32, 32), (10.0, 10.0))
        grid = make_grid(config)
        ctx = make_ddi_padded(grid, Rb87; c_dd=1.0)
        @test ctx.padded_shape == (64, 64)
        @test size(ctx.Q_xx) == rfft_output_shape((64, 64))
    end

    @testset "Padded DDI norm conservation (1D)" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sp = SimParams(; dt=0.01, n_steps=50)
        ws = make_workspace(;
            grid, atom=Rb87,
            interactions=InteractionParams(10.0, -0.5),
            potential=HarmonicTrap(1.0),
            sim_params=sp,
            enable_ddi=true, c_dd=1.0,
            ddi_padding=true,
        )
        @test ws.ddi_padded !== nothing
        @test ws.ddi_padded isa DDIPaddedContext

        N0 = total_norm(ws.state.psi, ws.grid)
        for _ in 1:50
            split_step!(ws)
        end
        @test total_norm(ws.state.psi, ws.grid) ≈ N0 rtol = 1e-5
    end

    @testset "Padded vs unpadded for localized state (1D)" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sp = SimParams(; dt=0.005, n_steps=20)

        ws_pad = make_workspace(;
            grid, atom=Rb87,
            interactions=InteractionParams(10.0, -0.5),
            potential=HarmonicTrap(1.0),
            sim_params=sp,
            enable_ddi=true, c_dd=1.0,
            ddi_padding=true,
        )

        ws_no = make_workspace(;
            grid, atom=Rb87,
            interactions=InteractionParams(10.0, -0.5),
            potential=HarmonicTrap(1.0),
            sim_params=sp,
            enable_ddi=true, c_dd=1.0,
            ddi_padding=false,
        )

        for _ in 1:20
            split_step!(ws_pad)
            split_step!(ws_no)
        end

        # Both should preserve norm
        @test total_norm(ws_pad.state.psi, ws_pad.grid) ≈ 1.0 rtol = 1e-4
        @test total_norm(ws_no.state.psi, ws_no.grid) ≈ 1.0 rtol = 1e-4

        # Results should be similar (DDI is weak here) but not identical
        diff = sum(abs2, ws_pad.state.psi .- ws_no.state.psi) / sum(abs2, ws_no.state.psi)
        @test diff < 0.1
    end

    @testset "ddi_padding=false gives nothing" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sp = SimParams(; dt=0.01, n_steps=10)
        ws = make_workspace(;
            grid, atom=Rb87,
            interactions=InteractionParams(1.0, 0.0),
            sim_params=sp,
            enable_ddi=true, c_dd=1.0,
            ddi_padding=false,
        )
        @test ws.ddi_padded === nothing
    end

    @testset "No DDI: padding has no effect" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sp = SimParams(; dt=0.01, n_steps=10)
        ws = make_workspace(;
            grid, atom=Rb87,
            interactions=InteractionParams(1.0, 0.0),
            sim_params=sp,
            ddi_padding=true,
        )
        @test ws.ddi === nothing
        @test ws.ddi_padded === nothing
    end
end

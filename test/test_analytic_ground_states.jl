using FFTW

@testset "F=1 analytic ground states" begin
    config = GridConfig(64, 10.0)
    grid = make_grid(config)
    trap = HarmonicTrap(1.0)

    @testset "ferromagnetic (c1 < 0) → |F,+F⟩ with Zeeman bias" begin
        interactions = InteractionParams(10.0, -0.5)
        result = find_ground_state(;
            grid, atom=Rb87, interactions, potential=trap,
            zeeman=ZeemanParams(1.0, 0.0),
            dt=0.005, n_steps=5000, initial_state=:ferromagnetic,
            fft_flags=FFTW.ESTIMATE,
        )
        psi = result.workspace.state.psi
        dV = cell_volume(grid)
        n1 = sum(abs2, psi[:, 1]) * dV
        @test n1 > 0.9
    end

    @testset "polar (c1 > 0) → |F,0⟩" begin
        interactions = InteractionParams(10.0, 0.5)
        result = find_ground_state(;
            grid, atom=Na23, interactions, potential=trap,
            dt=0.005, n_steps=5000, initial_state=:polar,
            fft_flags=FFTW.ESTIMATE,
        )
        psi = result.workspace.state.psi
        dV = cell_volume(grid)
        n2 = sum(abs2, psi[:, 2]) * dV
        @test n2 > 0.9
    end

    @testset "energy ordering: E_ferro < E_polar when c1 < 0" begin
        interactions = InteractionParams(10.0, -0.5)

        r_ferro = find_ground_state(;
            grid, atom=Rb87, interactions, potential=trap,
            dt=0.005, n_steps=3000, initial_state=:ferromagnetic,
            fft_flags=FFTW.ESTIMATE,
        )

        r_polar = find_ground_state(;
            grid, atom=Rb87, interactions, potential=trap,
            dt=0.005, n_steps=3000, initial_state=:polar,
            fft_flags=FFTW.ESTIMATE,
        )

        @test r_ferro.energy < r_polar.energy
    end

    @testset "energy ordering: E_polar < E_ferro when c1 > 0" begin
        interactions = InteractionParams(10.0, 0.5)

        r_ferro = find_ground_state(;
            grid, atom=Na23, interactions, potential=trap,
            dt=0.005, n_steps=3000, initial_state=:ferromagnetic,
            fft_flags=FFTW.ESTIMATE,
        )

        r_polar = find_ground_state(;
            grid, atom=Na23, interactions, potential=trap,
            dt=0.005, n_steps=3000, initial_state=:polar,
            fft_flags=FFTW.ESTIMATE,
        )

        @test r_polar.energy < r_ferro.energy
    end
end

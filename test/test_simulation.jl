using Test
using SpinorBEC
using FFTW

@testset "Simulation" begin
    @testset "Ground state: 87Rb ferromagnetic (c1 < 0)" begin
        config = GridConfig(128, 20.0)
        grid = make_grid(config)
        interactions = InteractionParams(10.0, -0.5)
        trap = HarmonicTrap(1.0)

        result = find_ground_state(;
            grid, atom=Rb87, interactions, potential=trap,
            dt=0.005, n_steps=5000, initial_state=:ferromagnetic,
        )

        psi = result.workspace.state.psi
        n1 = sum(abs2, psi[:, 1]) * cell_volume(grid)
        n2 = sum(abs2, psi[:, 2]) * cell_volume(grid)
        n3 = sum(abs2, psi[:, 3]) * cell_volume(grid)

        @test n1 > 0.9
        @test n2 < 0.05
        @test n3 < 0.05
        @test haskey(result, :dE)
        @test haskey(result, :dpsi)
    end

    @testset "Ground state: 23Na polar (c1 > 0)" begin
        config = GridConfig(128, 20.0)
        grid = make_grid(config)
        interactions = InteractionParams(10.0, 0.5)
        trap = HarmonicTrap(1.0)

        result = find_ground_state(;
            grid, atom=Na23, interactions, potential=trap,
            dt=0.005, n_steps=5000, initial_state=:polar,
        )

        psi = result.workspace.state.psi
        n1 = sum(abs2, psi[:, 1]) * cell_volume(grid)
        n2 = sum(abs2, psi[:, 2]) * cell_volume(grid)
        n3 = sum(abs2, psi[:, 3]) * cell_volume(grid)

        @test n2 > 0.9
        @test n1 < 0.05
        @test n3 < 0.05
    end

    @testset "seed_noise returns noisy copy preserving norm" begin
        config = GridConfig(64, 10.0)
        grid = make_grid(config)
        sys = SpinSystem(1)
        psi_gs = init_psi(grid, sys; state=:ferromagnetic)
        dV = cell_volume(grid)

        psi_noisy = seed_noise(psi_gs, sys.n_components, 1, grid)

        @test psi_noisy !== psi_gs
        @test psi_noisy != psi_gs
        norm_noisy = sum(abs2, psi_noisy) * dV
        @test abs(norm_noisy - 1.0) < 1e-10

        psi_noisy2 = seed_noise(psi_gs, sys.n_components, 1, grid)
        @test psi_noisy ≈ psi_noisy2
    end

    @testset "init_psi new states" begin
        grid = make_grid(GridConfig(64, 10.0))
        sys1 = SpinSystem(1)
        sys2 = SpinSystem(2)
        dV = cell_volume(grid)

        @testset "antiferromagnetic: unit norm" begin
            psi = init_psi(grid, sys1; state=:antiferromagnetic)
            @test total_norm(psi, grid) ≈ 1.0 atol = 1e-12
        end

        @testset "antiferromagnetic: zero magnetization for F=1" begin
            psi = init_psi(grid, sys1; state=:antiferromagnetic)
            Mz = magnetization(psi, grid, sys1)
            @test abs(Mz) < 1e-12
        end

        @testset "random: unit norm and seed reproducibility" begin
            psi1 = init_psi(grid, sys1; state=:random, seed=123)
            psi2 = init_psi(grid, sys1; state=:random, seed=123)
            psi3 = init_psi(grid, sys1; state=:random, seed=456)
            @test total_norm(psi1, grid) ≈ 1.0 atol = 1e-12
            @test psi1 ≈ psi2
            @test !(psi1 ≈ psi3)
        end

        @testset "spin_helix: unit norm" begin
            psi = init_psi(grid, sys1; state=:spin_helix, helix_k=(1.0,))
            @test total_norm(psi, grid) ≈ 1.0 atol = 1e-12
        end

        @testset "F=2 antiferromagnetic: unit norm" begin
            grid2 = make_grid(GridConfig(32, 10.0))
            psi = init_psi(grid2, sys2; state=:antiferromagnetic)
            @test total_norm(psi, grid2) ≈ 1.0 atol = 1e-12
        end
    end

    @testset "find_ground_state_multistart" begin
        config = GridConfig(64, 10.0)
        grid = make_grid(config)
        trap = HarmonicTrap(1.0)

        @testset "ferromagnetic (c1 < 0) picks ferro" begin
            interactions = InteractionParams(10.0, -0.5)
            result = find_ground_state_multistart(;
                grid, atom=Rb87, interactions, potential=trap,
                dt=0.005, n_steps=2000,
                initial_states=[:polar, :ferromagnetic],
                fft_flags=FFTW.ESTIMATE,
            )
            @test result.initial_state == :ferromagnetic
            @test result.converged || result.energy < 10.0
            @test length(result.all_results) == 2
        end
    end

    @testset "magnetization-constrained ITP" begin
        config = GridConfig(64, 10.0)
        grid = make_grid(config)
        trap = HarmonicTrap(1.0)
        interactions = InteractionParams(10.0, -0.5)

        @testset "target Mz=0 for ferromagnetic c1<0 → polar-like" begin
            result = find_ground_state(;
                grid, atom=Rb87, interactions, potential=trap,
                dt=0.005, n_steps=3000, initial_state=:uniform,
                target_magnetization=0.0,
                fft_flags=FFTW.ESTIMATE,
            )
            sys = SpinSystem(1)
            Mz = magnetization(result.workspace.state.psi, grid, sys)
            @test abs(Mz) < 0.05
        end

        @testset "target Mz=1 for ferromagnetic c1<0 → ferromagnetic" begin
            result = find_ground_state(;
                grid, atom=Rb87, interactions, potential=trap,
                dt=0.005, n_steps=3000, initial_state=:ferromagnetic,
                target_magnetization=1.0,
                fft_flags=FFTW.ESTIMATE,
            )
            psi = result.workspace.state.psi
            n1 = sum(abs2, psi[:, 1]) * cell_volume(grid)
            @test n1 > 0.9
        end
    end

    @testset "run_simulation! returns result" begin
        config = GridConfig(64, 10.0)
        grid = make_grid(config)
        interactions = InteractionParams(1.0, 0.1)
        sp = SimParams(; dt=0.01, n_steps=50, imaginary_time=false, save_every=10)

        ws = make_workspace(;
            grid, atom=Rb87, interactions, sim_params=sp,
        )

        result = run_simulation!(ws)
        @test length(result.times) == 6   # initial + 5 saves
        @test length(result.energies) == 6
        @test all(n -> abs(n - 1.0) < 1e-8, result.norms)
    end
end

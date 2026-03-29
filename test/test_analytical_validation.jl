using Test
using FFTW
using Random
using LinearAlgebra

@testset "Analytical Validation" begin

    # --- Test 1: F=1 Uniform Interaction Energy ---
    @testset "F=1 uniform interaction energy" begin
        config = GridConfig(64, 10.0)
        grid = make_grid(config)
        dV = cell_volume(grid)
        sm = spin_matrices(1)
        n0 = 1.0 / grid.config.box_size[1]
        c0, c1 = 100.0, 10.0

        @testset "ferromagnetic (1,0,0)" begin
            psi = zeros(ComplexF64, 64, 3)
            psi[:, 1] .= sqrt(n0)
            E_c0 = SpinorBEC._density_interaction_energy(psi, c0, 3, 1, (64,), dV)
            E_c1 = SpinorBEC._spin_interaction_energy(psi, sm, c1, 3, 1, (64,), dV)
            V = grid.config.box_size[1]
            @test E_c0 ≈ 0.5 * c0 * n0^2 * V rtol = 1e-10
            @test E_c1 ≈ 0.5 * c1 * n0^2 * V rtol = 1e-10
        end

        @testset "polar (0,1,0)" begin
            psi = zeros(ComplexF64, 64, 3)
            psi[:, 2] .= sqrt(n0)
            E_c0 = SpinorBEC._density_interaction_energy(psi, c0, 3, 1, (64,), dV)
            E_c1 = SpinorBEC._spin_interaction_energy(psi, sm, c1, 3, 1, (64,), dV)
            V = grid.config.box_size[1]
            @test E_c0 ≈ 0.5 * c0 * n0^2 * V rtol = 1e-10
            @test E_c1 ≈ 0.0 atol = 1e-12
        end

        @testset "antiferromagnetic (1/√2, 0, -1/√2)" begin
            psi = zeros(ComplexF64, 64, 3)
            psi[:, 1] .= sqrt(n0 / 2)
            psi[:, 3] .= -sqrt(n0 / 2)
            E_c0 = SpinorBEC._density_interaction_energy(psi, c0, 3, 1, (64,), dV)
            E_c1 = SpinorBEC._spin_interaction_energy(psi, sm, c1, 3, 1, (64,), dV)
            V = grid.config.box_size[1]
            @test E_c0 ≈ 0.5 * c0 * n0^2 * V rtol = 1e-10
            @test E_c1 ≈ 0.0 atol = 1e-12
        end
    end

    # --- Test 2: DDI Energy = 0 for Uniform Polarized BEC ---
    @testset "DDI energy zero for uniform m=+F" begin
        config = GridConfig{3}((16, 16, 16), (10.0, 10.0, 10.0))
        grid = make_grid(config)
        dV = cell_volume(grid)
        n0 = 1.0 / prod(grid.config.box_size)

        psi = zeros(ComplexF64, 16, 16, 16, 3)
        psi[:, :, :, 1] .= sqrt(n0)

        sp = SimParams(; dt=0.01, n_steps=1, save_every=1)
        ws = make_workspace(;
            grid, atom=Rb87,
            interactions=InteractionParams(0.0, 0.0),
            sim_params=sp,
            psi_init=psi,
            enable_ddi=true, c_dd=1.0,
            fft_flags=FFTW.ESTIMATE,
        )

        sm = ws.spin_matrices
        n_pts = (16, 16, 16)
        E_ddi = SpinorBEC._ddi_energy(psi, sm, ws.ddi, ws.ddi_bufs, 3, 3, n_pts, dV)
        @test abs(E_ddi) < 1e-12
    end

    # --- Test 3: F=1 Ground State Energy Decomposition ---
    @testset "F=1 ground state energy decomposition" begin
        config = GridConfig(128, 20.0)
        grid = make_grid(config)
        dV = cell_volume(grid)
        trap = HarmonicTrap(1.0)

        @testset "ferromagnetic: E_c1 ≈ 0.5*c1*∫n²" begin
            c0, c1 = 50.0, -5.0
            interactions = InteractionParams(c0, c1)
            result = find_ground_state(;
                grid, atom=Rb87, interactions, potential=trap,
                zeeman=ZeemanParams(1.0, 0.0),
                dt=0.002, n_steps=10000, initial_state=:ferromagnetic,
                fft_flags=FFTW.ESTIMATE,
            )
            psi = result.workspace.state.psi
            sm = result.workspace.spin_matrices
            E_c1 = SpinorBEC._spin_interaction_energy(psi, sm, c1, 3, 1, (128,), dV)
            n = SpinorBEC.total_density(psi, 1)
            expected_c1 = 0.5 * c1 * sum(n .^ 2) * dV
            @test E_c1 ≈ expected_c1 rtol = 1e-2
        end

        @testset "polar: E_c1 ≈ 0" begin
            c0, c1 = 50.0, 5.0
            interactions = InteractionParams(c0, c1)
            result = find_ground_state(;
                grid, atom=Na23, interactions, potential=trap,
                dt=0.002, n_steps=10000, initial_state=:polar,
                fft_flags=FFTW.ESTIMATE,
            )
            psi = result.workspace.state.psi
            sm = result.workspace.spin_matrices
            E_c1 = SpinorBEC._spin_interaction_energy(psi, sm, c1, 3, 1, (128,), dV)
            @test abs(E_c1) < 1e-6
        end
    end

    # --- Test 4: F=2 Known Phases (Ciobanu et al. 2000) ---
    @testset "F=2 phase diagram (Ciobanu)" begin
        config = GridConfig(64, 15.0)
        grid = make_grid(config)
        dV = cell_volume(grid)
        trap = HarmonicTrap(1.0)
        c0 = 50.0

        test_atom = AtomSpecies("test_F2", 1.0, 2, 0.0, 0.0, 0.0, 0.0)

        @testset "ferromagnetic: c1<0, c2 < 4|c1|" begin
            c1_val, c2_val = -5.0, 10.0
            interactions = InteractionParams(c0, c1_val, 0.0, [c2_val])
            result = find_ground_state(;
                grid, atom=test_atom, interactions, potential=trap,
                zeeman=ZeemanParams(0.5, 0.0),
                dt=0.002, n_steps=15000, initial_state=:ferromagnetic,
                fft_flags=FFTW.ESTIMATE,
            )
            psi = result.workspace.state.psi
            pop_p2 = sum(abs2, view(psi, :, 1)) * dV
            @test pop_p2 > 0.9
        end

        @testset "uniaxial nematic: c1>0" begin
            c1_val, c2_val = 5.0, 10.0
            interactions = InteractionParams(c0, c1_val, 0.0, [c2_val])
            result = find_ground_state(;
                grid, atom=test_atom, interactions, potential=trap,
                dt=0.002, n_steps=15000, initial_state=:polar,
                fft_flags=FFTW.ESTIMATE,
            )
            psi = result.workspace.state.psi
            pop_0 = sum(abs2, view(psi, :, 3)) * dV  # m=0 is component 3 for F=2
            @test pop_0 > 0.9
        end

        @testset "cyclic state observables" begin
            # Cyclic state for F=2: ψ = √(n/2)(1, 0, 0, 0, i) [m=+2, m=-2]
            # Properties: |F|²=0 (zero spin density), A₀₀=in/√5 (nonzero singlet)
            sm_f2 = spin_matrices(2)
            n0 = 1.0 / grid.config.box_size[1]
            psi_cyclic = zeros(ComplexF64, 64, 5)
            psi_cyclic[:, 1] .= sqrt(n0 / 2)
            psi_cyclic[:, 5] .= 1im * sqrt(n0 / 2)

            # Zero spin density
            fx, fy, fz = spin_density_vector(psi_cyclic, sm_f2, 1)
            @test maximum(abs, fx) < 1e-12
            @test maximum(abs, fy) < 1e-12
            @test maximum(abs, fz) < 1e-12

            # Nonzero singlet pair amplitude: |A₀₀|² = n²/5
            A00 = singlet_pair_amplitude(psi_cyclic, 2, 1)
            @test all(x -> isapprox(abs2(x), n0^2 / 5; rtol=1e-10), A00)

            # Spin interaction energy is zero (since |F|²=0)
            E_c1 = SpinorBEC._spin_interaction_energy(psi_cyclic, sm_f2, -5.0, 5, 1, (64,), dV)
            @test abs(E_c1) < 1e-12
        end

        @testset "F=2 energy ordering: ferro < polar when c1<0" begin
            sm_f2 = spin_matrices(2)
            n0 = 1.0 / grid.config.box_size[1]
            c1_val = -5.0

            psi_ferro = zeros(ComplexF64, 64, 5)
            psi_ferro[:, 1] .= sqrt(n0)  # m=+2, |F|²=4n²
            E_ferro = SpinorBEC._spin_interaction_energy(psi_ferro, sm_f2, c1_val, 5, 1, (64,), dV)

            psi_polar = zeros(ComplexF64, 64, 5)
            psi_polar[:, 3] .= sqrt(n0)  # m=0, |F|²=0
            E_polar = SpinorBEC._spin_interaction_energy(psi_polar, sm_f2, c1_val, 5, 1, (64,), dV)

            # Ferro has E_c1 = 0.5*c1*4*n²*V < 0, polar has E_c1 = 0
            @test E_ferro < E_polar
            V = grid.config.box_size[1]
            @test E_ferro ≈ 0.5 * c1_val * 4 * n0^2 * V rtol = 1e-10
            @test abs(E_polar) < 1e-12
        end
    end

    # --- Test 5: Tensor vs Spin Mixing Equivalence (F=1) ---
    @testset "tensor vs spin mixing equivalence (F=1)" begin
        config = GridConfig(64, 10.0)
        grid = make_grid(config)
        dV = cell_volume(grid)
        sm = spin_matrices(1)
        c0, c1 = 100.0, 10.0

        rng = MersenneTwister(42)
        psi_base = randn(rng, ComplexF64, 64, 3)
        norm_val = sqrt(sum(abs2, psi_base) * dV)
        psi_base ./= norm_val

        @testset "energy equivalence" begin
            psi = copy(psi_base)
            E_c0 = SpinorBEC._density_interaction_energy(psi, c0, 3, 1, (64,), dV)
            E_c1 = SpinorBEC._spin_interaction_energy(psi, sm, c1, 3, 1, (64,), dV)
            E_standard = E_c0 + E_c1

            g_dict = SpinorBEC._c0c1_to_gS(1, c0, c1)
            cache = SpinorBEC._make_tensor_cache_from_channels(1, g_dict)
            E_tensor = SpinorBEC._tensor_interaction_energy(psi, cache, 1, (64,), dV)

            @test E_standard ≈ E_tensor rtol = 1e-10
        end

        @testset "propagation equivalence" begin
            dt = 0.001
            n_propagation_steps = 10

            psi_sm = copy(psi_base)
            for _ in 1:n_propagation_steps
                apply_spin_mixing_step!(psi_sm, sm, c1, dt, 1)
            end

            g_dict_c1 = SpinorBEC._c0c1_to_gS(1, 0.0, c1)
            cache_c1 = SpinorBEC._make_tensor_cache_from_channels(1, g_dict_c1)
            psi_tensor = copy(psi_base)
            for _ in 1:n_propagation_steps
                SpinorBEC.apply_tensor_interaction_step!(psi_tensor, cache_c1, sm, dt, 1)
            end

            @test psi_sm ≈ psi_tensor rtol = 1e-4
        end
    end

    # --- Test 6: Pair Amplitude Exact Values (F=1) ---
    @testset "F=1 pair amplitude exact values" begin
        @testset "ferromagnetic: A00 = 0" begin
            psi = zeros(ComplexF64, 16, 3)
            psi[:, 1] .= 1.0
            A = singlet_pair_amplitude(psi, 1, 1)
            @test all(x -> abs(x) < 1e-12, A)
        end

        @testset "polar: A00 = -1/√3" begin
            psi = zeros(ComplexF64, 16, 3)
            psi[:, 2] .= 1.0
            A = singlet_pair_amplitude(psi, 1, 1)
            expected = -1.0 / sqrt(3.0)
            @test all(x -> isapprox(x, expected; rtol=1e-12), A)
        end
    end

    @testset "ITP overflow detection" begin
        @testset "contact interaction overflow throws" begin
            huge = InteractionParams(1e6, 0.0)
            @test_throws ArgumentError SpinorBEC._validate_itp_interactions(huge, 1, 0.01)
        end

        @testset "safe contact interaction passes" begin
            safe = InteractionParams(100.0, 10.0)
            @test SpinorBEC._validate_itp_interactions(safe, 1, 0.001) === nothing
        end

        @testset "DDI overflow throws" begin
            small_contact = InteractionParams(1.0, 0.0)
            @test_throws ArgumentError SpinorBEC._validate_itp_interactions(
                small_contact, 6, 0.01; c_dd=1e5)
        end

        @testset "DDI safe passes" begin
            ip = InteractionParams(100.0, 10.0)
            @test SpinorBEC._validate_itp_interactions(ip, 1, 0.001; c_dd=10.0) === nothing
        end

        @testset "DDI check disabled when c_dd=0" begin
            ip = InteractionParams(100.0, 0.0)
            @test SpinorBEC._validate_itp_interactions(ip, 6, 0.01; c_dd=0.0) === nothing
        end

        @testset "runtime NaN detection" begin
            grid = make_grid(GridConfig((16,), (10.0,)))
            sys = SpinSystem(1)
            psi_nan = zeros(ComplexF64, 16, 3)
            psi_nan[1, 1] = NaN
            sp = SimParams(; dt=0.001, n_steps=1, imaginary_time=true)
            ws = make_workspace(; grid, atom=Rb87,
                interactions=InteractionParams(100.0, 0.0),
                zeeman=ZeemanParams(), potential=NoPotential(),
                sim_params=sp, psi_init=psi_nan)
            @test_throws ArgumentError SpinorBEC._check_itp_overflow(ws, 1)
        end
    end

    # --- Test 7: Constrained Magnetization Newton Convergence ---
    @testset "constrained magnetization Jacobian" begin
        config = GridConfig(64, 10.0)
        grid = make_grid(config)
        dV = cell_volume(grid)
        n0 = 1.0 / grid.config.box_size[1]

        @testset "F=1 target Mz=0.5 converges" begin
            psi = zeros(ComplexF64, 64, 3)
            psi[:, 1] .= sqrt(n0 * 0.7)
            psi[:, 2] .= sqrt(n0 * 0.2)
            psi[:, 3] .= sqrt(n0 * 0.1)
            SpinorBEC._normalize_psi_constrained!(psi, grid, 3, 1, 0.5, 1)
            Mz = sum((1 - (c - 1)) * sum(abs2, view(psi, :, c)) * dV for c in 1:3)
            @test Mz ≈ 0.5 atol = 1e-10
        end

        @testset "F=6 target Mz=3.0 converges" begin
            psi = zeros(ComplexF64, 64, 13)
            for c in 1:13
                psi[:, c] .= sqrt(n0 / 13)
            end
            SpinorBEC._normalize_psi_constrained!(psi, grid, 13, 1, 3.0, 6)
            Mz = sum((6 - (c - 1)) * sum(abs2, view(psi, :, c)) * dV for c in 1:13)
            @test Mz ≈ 3.0 atol = 1e-10
        end
    end
end

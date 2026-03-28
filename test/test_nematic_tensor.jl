@testset "Nematic Tensor Eigenvalues" begin
    using LinearAlgebra

    @testset "Ferromagnetic |F,+F> (F=1): uniaxial prolate" begin
        F = 1
        sys = SpinSystem(F)
        sm = spin_matrices(F)
        grid = make_grid(GridConfig((16,), (10.0,)))
        psi = init_psi(grid, sys; state=:ferromagnetic)

        l1, l2, l3 = nematic_tensor_eigenvalues(psi, sm, 1)
        beta = biaxiality_parameter(l1, l2, l3)

        mid = 8
        @test l1[mid] >= l2[mid] >= l3[mid]
        @test l1[mid] + l2[mid] + l3[mid] ≈ 0.0 atol=1e-10
        # |F=1,+1>: ⟨Fz²⟩=1, ⟨Fx²⟩=⟨Fy²⟩=1/2
        # N_zz = 1 - 2/3 = 1/3, N_xx = N_yy = 1/2 - 2/3 = -1/6
        @test l1[mid] ≈ 1.0 / 3.0 atol=1e-10
        @test l2[mid] ≈ -1.0 / 6.0 atol=1e-10
        @test l3[mid] ≈ -1.0 / 6.0 atol=1e-10
        # Prolate uniaxial: l2 ≈ l3 → β ≈ 0
        @test beta[mid] ≈ 0.0 atol=1e-8
    end

    @testset "Polar |F,0> (F=1): uniaxial oblate" begin
        F = 1
        sys = SpinSystem(F)
        sm = spin_matrices(F)
        grid = make_grid(GridConfig((16,), (10.0,)))
        psi = init_psi(grid, sys; state=:polar)

        l1, l2, l3 = nematic_tensor_eigenvalues(psi, sm, 1)

        mid = 8
        # |F=1,m=0>: ⟨Fz²⟩=0, ⟨Fx²⟩=⟨Fy²⟩=1
        # N_zz = 0 - 2/3 = -2/3, N_xx = N_yy = 1 - 2/3 = 1/3
        @test l1[mid] ≈ 1.0 / 3.0 atol=1e-10
        @test l2[mid] ≈ 1.0 / 3.0 atol=1e-10
        @test l3[mid] ≈ -2.0 / 3.0 atol=1e-10
        @test l1[mid] + l2[mid] + l3[mid] ≈ 0.0 atol=1e-10
    end

    @testset "Ferromagnetic |F,+F> (F=2)" begin
        F = 2
        sys = SpinSystem(F)
        sm = spin_matrices(F)
        grid = make_grid(GridConfig((16,), (10.0,)))
        psi = init_psi(grid, sys; state=:ferromagnetic)

        l1, l2, l3 = nematic_tensor_eigenvalues(psi, sm, 1)

        mid = 8
        @test l1[mid] + l2[mid] + l3[mid] ≈ 0.0 atol=1e-10
        # F=2: ⟨Fz²⟩=4, F(F+1)/3=2. N_zz = 4 - 2 = 2
        @test l1[mid] ≈ 2.0 atol=1e-10
    end

    @testset "Tracelessness for random state" begin
        F = 2
        sys = SpinSystem(F)
        sm = spin_matrices(F)
        grid = make_grid(GridConfig((16,), (10.0,)))
        psi = init_psi(grid, sys; state=:random, seed=123)

        l1, l2, l3 = nematic_tensor_eigenvalues(psi, sm, 1)

        for i in 1:16
            @test l1[i] + l2[i] + l3[i] ≈ 0.0 atol=1e-10
        end
    end

    @testset "2D grid" begin
        F = 1
        sys = SpinSystem(F)
        sm = spin_matrices(F)
        grid = make_grid(GridConfig((8, 8), (6.0, 6.0)))
        psi = init_psi(grid, sys; state=:polar)

        l1, l2, l3 = nematic_tensor_eigenvalues(psi, sm, 2)

        @test size(l1) == (8, 8)
        @test l1[4, 4] + l2[4, 4] + l3[4, 4] ≈ 0.0 atol=1e-10
    end

    @testset "Low density regions are zero" begin
        F = 1
        sys = SpinSystem(F)
        sm = spin_matrices(F)
        grid = make_grid(GridConfig((32,), (20.0,)))
        psi = init_psi(grid, sys; state=:polar)

        # Use high cutoff to force corners to be skipped
        max_n = maximum(sum(c -> abs2.(psi[:, c]), 1:3))
        l1, l2, l3 = nematic_tensor_eigenvalues(psi, sm, 1; density_cutoff=max_n * 0.5)

        # Corner points should be below cutoff and thus zero
        @test l1[1] == 0.0
        @test l2[1] == 0.0
        @test l3[1] == 0.0
    end

    @testset "Biaxiality in [0,1]" begin
        F = 2
        sys = SpinSystem(F)
        sm = spin_matrices(F)
        grid = make_grid(GridConfig((16,), (10.0,)))
        psi = init_psi(grid, sys; state=:random, seed=99)

        l1, l2, l3 = nematic_tensor_eigenvalues(psi, sm, 1)
        beta = biaxiality_parameter(l1, l2, l3)

        for i in 1:16
            n_local = sum(c -> abs2(psi[i, c]), 1:5)
            if n_local > 1e-10
                @test 0.0 <= beta[i] <= 1.0 + 1e-10
            end
        end
    end
end

using Test
using SpinorBEC
using LinearAlgebra

@testset "Spin Matrices" begin
    @testset "Spin-1 basics" begin
        sm = spin_matrices(1)
        @test sm.system.F == 1
        @test sm.system.n_components == 3
        @test sm.system.m_values == [1, 0, -1]
    end

    @testset "Hermiticity" begin
        for F in [1, 2, 3]
            sm = spin_matrices(F)
            @test sm.Fx ≈ sm.Fx' atol = 1e-14
            @test sm.Fy ≈ sm.Fy' atol = 1e-14
            @test sm.Fz ≈ sm.Fz' atol = 1e-14
        end
    end

    @testset "Commutation relations [Fi, Fj] = i εijk Fk" begin
        for F in [1, 2, 3]
            sm = spin_matrices(F)
            comm_xy = sm.Fx * sm.Fy - sm.Fy * sm.Fx
            @test comm_xy ≈ 1im * sm.Fz atol = 1e-12

            comm_yz = sm.Fy * sm.Fz - sm.Fz * sm.Fy
            @test comm_yz ≈ 1im * sm.Fx atol = 1e-12

            comm_zx = sm.Fz * sm.Fx - sm.Fx * sm.Fz
            @test comm_zx ≈ 1im * sm.Fy atol = 1e-12
        end
    end

    @testset "Casimir invariant F·F = F(F+1)I" begin
        for F in [1, 2, 3, 6]
            sm = spin_matrices(F)
            n = 2F + 1
            expected = F * (F + 1) * I(n)
            @test Matrix(sm.F_dot_F) ≈ expected atol = 1e-12
        end
    end

    @testset "Fz eigenvalues" begin
        sm = spin_matrices(1)
        @test real(diag(Matrix(sm.Fz))) ≈ [1.0, 0.0, -1.0]
    end

    @testset "Spin-6 (Eu)" begin
        sm = spin_matrices(6)
        @test sm.system.n_components == 13
        @test sm.system.m_values == collect(6:-1:-6)
    end
end

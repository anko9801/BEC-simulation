using Test
using SpinorBEC

@testset "Propagators" begin
    @testset "Free particle Gaussian spreading (1D)" begin
        config = GridConfig(256, 40.0)
        grid = make_grid(config)
        x = grid.x[1]
        dx = grid.dx[1]

        sigma0 = 2.0
        psi = zeros(ComplexF64, 256, 1)
        psi[:, 1] .= exp.(-x .^ 2 ./ (2 * sigma0^2))
        norm0 = sqrt(sum(abs2, psi) * dx)
        psi ./= norm0

        dt = 0.01
        n_steps = 100
        kinetic_phase = prepare_kinetic_phase(grid, dt)
        fft_buf = zeros(ComplexF64, 256)
        plans = make_fft_plans((256,))

        for _ in 1:n_steps
            apply_kinetic_step!(psi, fft_buf, kinetic_phase, plans, 1, 1)
        end

        norm_after = sqrt(sum(abs2, psi) * dx)
        @test norm_after ≈ 1.0 atol = 1e-10

        n = abs2.(psi[:, 1])
        width_after = sqrt(sum(n .* x .^ 2 .* dx) / sum(n .* dx))
        width_initial = sigma0 / sqrt(2)
        @test width_after > width_initial
    end

    @testset "Kinetic phase (imaginary time)" begin
        config = GridConfig(64, 10.0)
        grid = make_grid(config)
        phase = prepare_kinetic_phase(grid, 0.01; imaginary_time=true)
        @test all(real.(phase) .>= 0)
        @test all(real.(phase) .<= 1)
        @test all(imag.(phase) .≈ 0)
    end

    @testset "Diagonal potential preserves norm (zero c0)" begin
        config = GridConfig(128, 20.0)
        grid = make_grid(config)
        x = grid.x[1]
        dx = grid.dx[1]

        psi = zeros(ComplexF64, 128, 3)
        psi[:, 2] .= exp.(-x .^ 2 ./ 4) .+ 0im
        norm0 = sqrt(sum(abs2, psi) * dx)
        psi ./= norm0

        V = 0.5 .* x .^ 2
        zeeman = [0.0, 0.0, 0.0]

        apply_diagonal_potential_step!(psi, V, zeeman, 0.0, 0.1, 3, 1)
        norm_after = sqrt(sum(abs2, psi) * dx)
        @test norm_after ≈ 1.0 atol = 1e-12
    end
end

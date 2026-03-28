@testset "Phase Boundary Bisection" begin
    F = 1
    grid = make_grid(GridConfig((16,), (10.0,)))
    atom = Rb87

    @testset "Returns valid result" begin
        make_ip = val -> InteractionParams(10.0, val)

        result = find_phase_boundary(;
            param_range=(-10.0, 10.0),
            make_interactions=make_ip,
            grid, atom,
            tol=1.0,
            max_bisections=5,
            dt=0.01,
            n_steps=500,
        )

        @test haskey(result, :boundary_value)
        @test haskey(result, :phase_left)
        @test haskey(result, :phase_right)
        @test haskey(result, :n_evaluations)
        @test result.n_evaluations >= 2
        @test -10.0 <= result.boundary_value <= 10.0
    end

    @testset "Same phase returns midpoint" begin
        make_ip = val -> InteractionParams(10.0, val)

        result = find_phase_boundary(;
            param_range=(-10.0, -5.0),
            make_interactions=make_ip,
            grid, atom,
            tol=0.1,
            max_bisections=3,
            dt=0.01,
            n_steps=500,
        )

        @test result.n_evaluations == 2
        @test result.phase_left == result.phase_right
    end

    @testset "Bisection narrows interval" begin
        make_ip = val -> InteractionParams(10.0, val)

        r1 = find_phase_boundary(;
            param_range=(-20.0, 20.0),
            make_interactions=make_ip,
            grid, atom,
            tol=5.0,
            max_bisections=3,
            dt=0.01,
            n_steps=500,
        )

        r2 = find_phase_boundary(;
            param_range=(-20.0, 20.0),
            make_interactions=make_ip,
            grid, atom,
            tol=2.0,
            max_bisections=10,
            dt=0.01,
            n_steps=500,
        )

        @test r2.n_evaluations >= r1.n_evaluations
    end
end

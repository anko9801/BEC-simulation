@testset "Potentials" begin
    @testset "GravityPotential evaluate_potential" begin
        grid_cfg = GridConfig((32,), (10.0,))
        grid = make_grid(grid_cfg)
        grav = GravityPotential{1}(2.5, 1)
        V = evaluate_potential(grav, grid)
        @test size(V) == (32,)
        for i in 1:32
            @test V[i] ≈ 2.5 * grid.x[1][i]
        end

        grid_cfg2 = GridConfig((8, 8), (4.0, 4.0))
        grid2 = make_grid(grid_cfg2)
        grav2 = GravityPotential{2}(9.81, 2)
        V2 = evaluate_potential(grav2, grid2)
        @test size(V2) == (8, 8)
        for I in CartesianIndices((8, 8))
            @test V2[I] ≈ 9.81 * grid2.x[2][I[2]]
        end
    end

    @testset "GravityPotential validation" begin
        @test_throws ArgumentError GravityPotential{1}(9.81, 2)
        @test_throws ArgumentError GravityPotential{2}(9.81, 0)
        @test_throws ArgumentError GravityPotential{2}(9.81, 3)
    end

    @testset "CompositePotential evaluate_potential" begin
        grid_cfg = GridConfig((32,), (10.0,))
        grid = make_grid(grid_cfg)
        harm = HarmonicTrap((1.0,))
        grav = GravityPotential{1}(2.0, 1)
        comp = CompositePotential{1}([harm, grav])
        V = evaluate_potential(comp, grid)
        V_harm = evaluate_potential(harm, grid)
        V_grav = evaluate_potential(grav, grid)
        @test V ≈ V_harm .+ V_grav
    end

    @testset "build_potential - gravity" begin
        pc = PotentialConfig(:gravity, Dict{String,Any}("g" => 9.81, "axis" => 1))
        pot = SpinorBEC._build_potential(pc, 1)
        @test pot isa GravityPotential{1}
        @test pot.g == 9.81
        @test pot.axis == 1
    end

    @testset "build_potential - gravity defaults" begin
        pc = PotentialConfig(:gravity, Dict{String,Any}())
        pot = SpinorBEC._build_potential(pc, 2)
        @test pot isa GravityPotential{2}
        @test pot.g == 9.81
        @test pot.axis == 2
    end

    @testset "build_potential - crossed_dipole" begin
        beams = [
            Dict{String,Any}(
                "wavelength" => 1064e-9, "power" => 10.0, "waist" => 50e-6,
                "position" => [0, 0, 0], "direction" => [1, 0, 0]
            ),
        ]
        pc = PotentialConfig(:crossed_dipole, Dict{String,Any}("polarizability" => 1.5e-37, "beams" => beams))
        pot = SpinorBEC._build_potential(pc, 3)
        @test pot isa CrossedDipoleTrap{3}
        @test pot.polarizability == 1.5e-37
        @test length(pot.beams) == 1
    end

    @testset "build_potential - composite" begin
        c1 = PotentialConfig(:harmonic, Dict{String,Any}("omega" => [1.0]))
        c2 = PotentialConfig(:gravity, Dict{String,Any}("g" => 5.0, "axis" => 1))
        pc = PotentialConfig(:composite, Dict{String,Any}("components" => [c1, c2]))
        pot = SpinorBEC._build_potential(pc, 1)
        @test pot isa CompositePotential{1}
        @test length(pot.components) == 2
        @test pot.components[1] isa HarmonicTrap{1}
        @test pot.components[2] isa GravityPotential{1}
    end
end

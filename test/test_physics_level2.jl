using Test
using SpinorBEC

@testset "Level 2: Thomas-Fermi Ground State" begin

    @testset "1D Thomas-Fermi density profile" begin
        # ℏ = m = ω = 1, large c₀ → Thomas-Fermi regime
        # n_TF(x) = max(0, (μ - x²/2) / c₀)
        # Normalization: ∫n_TF dx = N = 1
        # μ_TF = (3c₀/(4√2))^{2/3}
        c0 = 200.0

        gc = GridConfig((256,), (20.0,))
        grid = make_grid(gc)
        atom = AtomSpecies("test", 1.0, 1, 0.0, 0.0)

        result = find_ground_state(;
            grid, atom,
            interactions=InteractionParams(c0, 0.0),
            potential=HarmonicTrap(1.0),
            dt=0.005, n_steps=10000, tol=1e-10,
            initial_state=:polar,
        )

        @test result.converged

        psi = result.workspace.state.psi
        dV = cell_volume(grid)
        x = grid.x[1]

        n_num = component_density(psi, 1, 2)

        μ_TF = (3 * c0 / (4 * √2))^(2 / 3)
        n_TF = max.(0.0, (μ_TF .- x .^ 2 ./ 2) ./ c0)
        n_TF ./= sum(n_TF) * dV

        # L1 norm error < 5%
        err = sum(abs.(vec(n_num) .- n_TF)) * dV
        @test err < 0.05

        # Thomas-Fermi radius check
        R_TF = √(2μ_TF)
        # Find numerical radius: outermost point where density > 1% of peak
        peak = maximum(n_num)
        numerical_R = maximum(abs(x[i]) for i in eachindex(x) if n_num[i] > 0.01 * peak)
        @test abs(numerical_R - R_TF) / R_TF < 0.1
    end

end

function evaluate_potential(::NoPotential, grid::Grid{N}) where {N}
    zeros(Float64, grid.config.n_points)
end

function evaluate_potential(trap::HarmonicTrap{1}, grid::Grid{1})
    x = grid.x[1]
    ox = trap.omega[1]
    @. 0.5 * ox^2 * x^2
end

function evaluate_potential(trap::HarmonicTrap{2}, grid::Grid{2})
    nx, ny = grid.config.n_points
    x, y = grid.x
    ox, oy = trap.omega
    V = zeros(Float64, nx, ny)
    for j in 1:ny, i in 1:nx
        V[i, j] = 0.5 * (ox^2 * x[i]^2 + oy^2 * y[j]^2)
    end
    V
end

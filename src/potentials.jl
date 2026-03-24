function evaluate_potential(::NoPotential, grid::Grid{N}) where {N}
    zeros(Float64, grid.config.n_points)
end

function evaluate_potential(trap::HarmonicTrap{N}, grid::Grid{N}) where {N}
    V = zeros(Float64, grid.config.n_points)
    @inbounds for I in CartesianIndices(grid.config.n_points)
        s = 0.0
        for d in 1:N
            s += trap.omega[d]^2 * grid.x[d][I[d]]^2
        end
        V[I] = 0.5 * s
    end
    V
end

function evaluate_potential(grav::GravityPotential{N}, grid::Grid{N}) where {N}
    V = zeros(Float64, grid.config.n_points)
    ax = grav.axis
    @inbounds for I in CartesianIndices(grid.config.n_points)
        V[I] = grav.g * grid.x[ax][I[ax]]
    end
    V
end

function evaluate_potential(comp::CompositePotential{N}, grid::Grid{N}) where {N}
    V = zeros(Float64, grid.config.n_points)
    for pot in comp.components
        V .+= evaluate_potential(pot, grid)
    end
    V
end

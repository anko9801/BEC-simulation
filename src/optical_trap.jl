struct GaussianBeam
    wavelength::Float64     # m
    power::Float64          # W
    waist::Float64          # m (1/e² radius)
    position::NTuple{3,Float64}   # m (focus position)
    direction::NTuple{3,Float64}  # unit vector (propagation axis)
end

struct CrossedDipoleTrap{N} <: AbstractPotential
    beams::Vector{GaussianBeam}
    polarizability::Float64  # J/(W/m²), scalar polarizability α
end

CrossedDipoleTrap(beams, pol; dims::Int=3) = CrossedDipoleTrap{dims}(beams, pol)

function evaluate_potential(trap::CrossedDipoleTrap{N}, grid::Grid{N}) where {N}
    V = zeros(Float64, grid.config.n_points)
    for beam in trap.beams
        _add_beam_potential!(V, beam, grid, trap.polarizability)
    end
    V
end

function _add_beam_potential!(V::Array{Float64,N}, beam::GaussianBeam, grid::Grid{N}, alpha::Float64) where {N}
    w0 = beam.waist
    P = beam.power
    I0 = 2 * P / (π * w0^2)

    d = beam.direction
    pos = beam.position

    @inbounds for I in CartesianIndices(size(V))
        coords = ntuple(N) do dim
            grid.x[dim][I[dim]] - pos[dim]
        end

        axial = sum(ntuple(dim -> coords[dim] * d[dim], N))
        r_perp_sq = sum(ntuple(dim -> coords[dim]^2, N)) - axial^2

        intensity = I0 * exp(-2 * r_perp_sq / w0^2)
        V[I] += -alpha * intensity
    end
    nothing
end

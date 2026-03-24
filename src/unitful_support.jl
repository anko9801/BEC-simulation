using Unitful: Quantity, ustrip, @u_str

# --- OpticalBeam: positional constructor with Quantity ---

function OpticalBeam(
    wavelength::Quantity,
    power::Quantity,
    waist::Quantity;
    M2::Float64=1.0,
)
    OpticalBeam(;
        wavelength=Float64(ustrip(u"m", wavelength)),
        power=Float64(ustrip(u"W", power)),
        waist=Float64(ustrip(u"m", waist)),
        M2,
    )
end

# --- GridConfig ---

GridConfig(n_points::Int, box_size::Quantity) =
    GridConfig(n_points, Float64(ustrip(u"m", box_size)))

function GridConfig(n_points::NTuple{N,Int}, box_size::NTuple{N,Quantity}) where {N}
    box_m = ntuple(i -> Float64(ustrip(u"m", box_size[i])), N)
    GridConfig(n_points, box_m)
end

# --- HarmonicTrap: angular frequency from Hz ---

HarmonicTrap(omega::Quantity) =
    HarmonicTrap(Float64(ustrip(u"Hz", omega)) * 2π)

HarmonicTrap(ox::Quantity, oy::Quantity) =
    HarmonicTrap(Float64(ustrip(u"Hz", ox)) * 2π,
                 Float64(ustrip(u"Hz", oy)) * 2π)

HarmonicTrap(ox::Quantity, oy::Quantity, oz::Quantity) =
    HarmonicTrap(Float64(ustrip(u"Hz", ox)) * 2π,
                 Float64(ustrip(u"Hz", oy)) * 2π,
                 Float64(ustrip(u"Hz", oz)) * 2π)

# --- LaserBeamPotential ---

function LaserBeamPotential(
    beam::OpticalBeam,
    polarizability::Float64,
    position::NTuple{N,Quantity},
    direction::NTuple{N,Float64},
) where {N}
    pos_m = ntuple(i -> Float64(ustrip(u"m", position[i])), N)
    LaserBeamPotential(beam, polarizability, pos_m, direction)
end

# --- RamanCoupling ---

function RamanCoupling{N}(
    Omega_R::Quantity,
    delta::Quantity,
    k_eff::NTuple{N,Quantity},
) where {N}
    RamanCoupling{N}(
        Float64(ustrip(u"Hz", Omega_R)) * 2π,
        Float64(ustrip(u"Hz", delta)) * 2π,
        ntuple(i -> Float64(ustrip(u"m^-1", k_eff[i])), N),
    )
end

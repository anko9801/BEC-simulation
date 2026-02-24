module Units

const HBAR = 1.054571817e-34      # J·s
const AMU = 1.66053906660e-27     # kg
const BOHR_RADIUS = 5.29177210903e-11  # m
const BOHR_MAGNETON = 9.2740100783e-24 # J/T
const KB = 1.380649e-23           # J/K

struct DimensionlessScales
    length_scale::Float64   # meters per dimensionless unit
    time_scale::Float64     # seconds per dimensionless unit
    energy_scale::Float64   # Joules per dimensionless unit
end

function harmonic_scales(mass::Float64, omega::Float64)
    l = sqrt(HBAR / (mass * omega))
    t = 1.0 / omega
    e = HBAR * omega
    DimensionlessScales(l, t, e)
end

function to_dimensionless_length(x_si::Float64, scales::DimensionlessScales)
    x_si / scales.length_scale
end

function to_si_length(x::Float64, scales::DimensionlessScales)
    x * scales.length_scale
end

function to_dimensionless_time(t_si::Float64, scales::DimensionlessScales)
    t_si / scales.time_scale
end

function to_si_time(t::Float64, scales::DimensionlessScales)
    t * scales.time_scale
end

function to_dimensionless_energy(e_si::Float64, scales::DimensionlessScales)
    e_si / scales.energy_scale
end

function to_si_energy(e::Float64, scales::DimensionlessScales)
    e * scales.energy_scale
end

end # module

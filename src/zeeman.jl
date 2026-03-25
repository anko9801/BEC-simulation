"""
Return Zeeman energy shifts as a vector indexed by m = F, F-1, ..., -F.

Linear Zeeman: E_m = -p * m
Quadratic Zeeman: E_m = q * m²
"""
function zeeman_energies(z::ZeemanParams, sys::SpinSystem)
    [(-z.p * m + z.q * m^2) for m in sys.m_values]
end

function zeeman_diagonal(z::ZeemanParams, sys::SpinSystem)
    SVector{sys.n_components,Float64}(zeeman_energies(z, sys))
end

function zeeman_diagonal(z::ZeemanParams, sm::SpinMatrices{D}) where {D}
    F = sm.system.F
    SVector{D,Float64}(ntuple(c -> -z.p * (F - (c - 1)) + z.q * (F - (c - 1))^2, Val(D)))
end

zeeman_at(z::ZeemanParams, ::Float64) = z
zeeman_at(z::TimeDependentZeeman, t::Float64) = z.B_func(t)

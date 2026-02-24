"""
Return Zeeman energy shifts as a vector indexed by m = F, F-1, ..., -F.

Linear Zeeman: E_m = -p * m
Quadratic Zeeman: E_m = q * m²
"""
function zeeman_energies(z::ZeemanParams, sys::SpinSystem)
    [(-z.p * m + z.q * m^2) for m in sys.m_values]
end

"""
Return diagonal Zeeman matrix (StaticVector) for spin-1.
"""
function zeeman_diagonal(z::ZeemanParams, sys::SpinSystem)
    SVector{sys.n_components,Float64}(zeeman_energies(z, sys))
end

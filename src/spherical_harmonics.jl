"""
    spherical_harmonic(l, m, theta, phi) → ComplexF64

Spherical harmonic Y_{lm}(θ,φ) in the Condon-Shortley convention.
Uses associated Legendre recurrence for numerical stability.
"""
function spherical_harmonic(l::Int, m::Int, theta::Float64, phi::Float64)::ComplexF64
    abs(m) > l && return zero(ComplexF64)

    am = abs(m)
    ct = cos(theta)
    st = sin(theta)

    plm = _associated_legendre(l, am, ct, st)

    norm = sqrt((2l + 1) / (4π) *
        exp(_log_factorial(l - am) - _log_factorial(l + am)))

    # Y_{l,|m|} with positive m phase
    ylm_pos = norm * plm * cis(am * phi)

    if m >= 0
        return ylm_pos
    else
        # Y_{l,-|m|} = (-1)^|m| conj(Y_{l,|m|})
        return (iseven(am) ? 1.0 : -1.0) * conj(ylm_pos)
    end
end

"""
Associated Legendre polynomial P_l^m(cos θ) computed via upward recurrence.
Includes the Condon-Shortley phase (-1)^m.
"""
function _associated_legendre(l::Int, m::Int, ct::Float64, st::Float64)
    pmm = 1.0
    if m > 0
        fact = 1.0
        for i in 1:m
            pmm *= -fact * st
            fact += 2.0
        end
    end

    m == l && return pmm

    pmm1 = ct * (2m + 1) * pmm
    (m + 1) == l && return pmm1

    plm = 0.0
    for ll in (m + 2):l
        plm = ((2ll - 1) * ct * pmm1 - (ll + m - 1) * pmm) / (ll - m)
        pmm = pmm1
        pmm1 = plm
    end
    plm
end

"""
    spinor_angular_density(spinor, F; n_theta=64, n_phi=128) → (theta, phi, rho)

Angular probability density ρ(θ,φ) = |Σ_m Y_{Fm}(θ,φ) ζ_m|² for a normalized spinor.
Returns theta ∈ [0,π], phi ∈ [0,2π), and rho matrix (n_theta × n_phi).
"""
function spinor_angular_density(spinor::AbstractVector{<:Number}, F::Int;
                                 n_theta::Int=64, n_phi::Int=128)
    D = 2F + 1
    length(spinor) == D || throw(DimensionMismatch(
        "spinor length $(length(spinor)) != 2F+1 = $D"))

    theta = range(0, π, length=n_theta)
    phi = range(0, 2π * (1 - 1 / n_phi), length=n_phi)
    rho = zeros(Float64, n_theta, n_phi)

    for (it, th) in enumerate(theta)
        for (ip, ph) in enumerate(phi)
            val = zero(ComplexF64)
            for c in 1:D
                m = F - (c - 1)
                val += spherical_harmonic(F, m, th, ph) * spinor[c]
            end
            rho[it, ip] = abs2(val)
        end
    end

    (theta=collect(theta), phi=collect(phi), rho=rho)
end

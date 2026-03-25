"""
Majorana polynomial coefficients for a spin-F spinor.
P(z) = Σ_{k=0}^{2F} (-1)^k √C(2F,k) ψ_{F-k} z^k
where ψ_{F-k} is the component with m = F-k (index k+1).
"""
function _majorana_polynomial(spinor::AbstractVector{ComplexF64}, F::Int)
    n = 2F + 1
    coeffs = Vector{ComplexF64}(undef, n)
    for k in 0:2F
        coeffs[k+1] = (-1)^k * sqrt(binomial(2F, k)) * spinor[k+1]
    end
    coeffs
end

"""
Find 2F Majorana stars (roots of the Majorana polynomial) via companion matrix.
Returns `Vector{ComplexF64}` of length 2F.
Roots at infinity (when leading coefficients vanish) are represented as `complex(Inf)`.
"""
function majorana_stars(spinor::AbstractVector{ComplexF64}, F::Int)
    n = 2F
    n == 0 && return ComplexF64[]
    coeffs = _majorana_polynomial(spinor, F)

    deg = n
    while deg >= 1 && abs(coeffs[deg+1]) < 1e-14
        deg -= 1
    end
    deg == 0 && return fill(complex(Inf), n)

    if deg == 1
        roots = [-coeffs[1] / coeffs[2]]
    else
        c = coeffs[1:deg+1]
        companion = zeros(ComplexF64, deg, deg)
        for i in 1:deg-1
            companion[i+1, i] = 1.0
        end
        for i in 1:deg
            companion[i, deg] = -c[i] / c[deg+1]
        end
        roots = eigvals(companion)
    end

    n_inf = n - deg
    if n_inf > 0
        append!(roots, fill(complex(Inf), n_inf))
    end
    roots
end

"""
Stereographic projection: complex plane → unit sphere.
z → (2Re(z), 2Im(z), |z|²-1) / (|z|²+1)
z = Inf maps to south pole (0, 0, -1).
"""
function _stereo_to_sphere(z::ComplexF64)
    if !isfinite(z)
        return (0.0, 0.0, -1.0)
    end
    r2 = abs2(z)
    inv_denom = 1.0 / (r2 + 1.0)
    (2.0 * real(z) * inv_denom, 2.0 * imag(z) * inv_denom, (r2 - 1.0) * inv_denom)
end

"""
Legendre polynomial P₆(x) = (231x⁶ - 315x⁴ + 105x² - 5) / 16.
"""
function _legendre_p6(x::Float64)
    x2 = x * x
    x4 = x2 * x2
    x6 = x4 * x2
    (231.0 * x6 - 315.0 * x4 + 105.0 * x2 - 5.0) / 16.0
end

"""
Steinhardt Q₆ bond-orientational order parameter, normalized so Q₆ = 1
for a perfect icosahedron.

Q₆_raw = √(4π/13 · (1/N²) Σ_{i,j} P₆(n̂_i · n̂_j))
Normalized: Q₆ = Q₆_raw / Q₆_icosa where Q₆_icosa ≈ 0.6633.
"""
function _steinhardt_q6(points::Vector{NTuple{3,Float64}})
    N = length(points)
    N == 0 && return 0.0

    s = 0.0
    @inbounds for i in 1:N
        for j in 1:N
            costh = points[i][1] * points[j][1] +
                    points[i][2] * points[j][2] +
                    points[i][3] * points[j][3]
            costh = clamp(costh, -1.0, 1.0)
            s += _legendre_p6(costh)
        end
    end

    q6_raw = sqrt(4π / 13.0 * s / N^2)
    q6_raw / 0.6633
end

"""
Local icosahedral order parameter at each spatial point.
At each point: spinor → Majorana stars → sphere points → Steinhardt Q₆.
Returns `Array{Float64,N}` (0 everywhere for F < 6).
"""
function icosahedral_order_parameter(psi::AbstractArray{ComplexF64}, grid::Grid{N},
                                     sm::SpinMatrices{D};
                                     density_cutoff::Float64=1e-10) where {D,N}
    F = sm.system.F
    n_comp = sm.system.n_components
    n_pts = ntuple(d -> size(psi, d), N)
    result = zeros(Float64, n_pts)
    F >= 6 || return result

    n = _total_density(psi, n_comp, N, n_pts)

    @inbounds for I in CartesianIndices(n_pts)
        n[I] > density_cutoff || continue
        spinor = _get_spinor(psi, I, Val(D))
        inv_norm = 1.0 / sqrt(real(dot(spinor, spinor)))
        spinor_normed = SVector{D,ComplexF64}(spinor .* inv_norm)

        stars = majorana_stars(Vector{ComplexF64}(spinor_normed), F)
        points = [_stereo_to_sphere(z) for z in stars]
        result[I] = _steinhardt_q6(points)
    end
    result
end

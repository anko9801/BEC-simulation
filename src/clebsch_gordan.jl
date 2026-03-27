"""
Log-space factorial for numerical stability (handles n up to ~170).
"""
function _log_factorial(n::Int)
    n < 0 && return -Inf
    n <= 1 && return 0.0
    s = 0.0
    for i in 2:n
        s += log(i)
    end
    s
end

"""
Triangle coefficient Δ(a,b,c) used in Wigner 3j and 6j symbols.
Returns log(Δ²) = log((a+b-c)!(a-b+c)!(-a+b+c)! / (a+b+c+1)!).
"""
function _log_triangle_coeff(a::Int, b::Int, c::Int)
    ab = a + b - c
    ac = a - b + c
    bc = -a + b + c
    (ab < 0 || ac < 0 || bc < 0) && return -Inf
    _log_factorial(ab) + _log_factorial(ac) + _log_factorial(bc) - _log_factorial(a + b + c + 1)
end

"""
    wigner_3j(j1, j2, j3, m1, m2, m3)

Wigner 3-j symbol via the Racah formula in log-space.
All arguments are integers (not half-integers); for half-integer spins, pass 2j values.
For this codebase, F is always integer so we use integer arithmetic throughout.
"""
function wigner_3j(j1::Int, j2::Int, j3::Int, m1::Int, m2::Int, m3::Int)
    m1 + m2 + m3 != 0 && return 0.0

    log_tri = _log_triangle_coeff(j1, j2, j3)
    log_tri == -Inf && return 0.0

    log_num = (_log_factorial(j1 + m1) + _log_factorial(j1 - m1) +
               _log_factorial(j2 + m2) + _log_factorial(j2 - m2) +
               _log_factorial(j3 + m3) + _log_factorial(j3 - m3))

    t_min = max(0, j2 - j3 - m1, j1 - j3 + m2)
    t_max = min(j1 + j2 - j3, j1 - m1, j2 + m2)
    t_min > t_max && return 0.0

    s = 0.0
    for t in t_min:t_max
        log_den = (_log_factorial(t) +
                   _log_factorial(j1 + j2 - j3 - t) +
                   _log_factorial(j1 - m1 - t) +
                   _log_factorial(j2 + m2 - t) +
                   _log_factorial(j3 - j2 + m1 + t) +
                   _log_factorial(j3 - j1 - m2 + t))
        sign = iseven(t) ? 1.0 : -1.0
        s += sign * exp(-log_den)
    end

    phase = iseven(j1 - j2 - m3) ? 1.0 : -1.0
    phase * exp(0.5 * (log_tri + log_num)) * s
end

"""
    clebsch_gordan(j1, m1, j2, m2, J, M)

Clebsch-Gordan coefficient ⟨j1,m1;j2,m2|J,M⟩ via the relation to Wigner 3-j:
    CG = (-1)^(j1-j2+M) √(2J+1) × 3j(j1,j2,J, m1,m2,-M)
"""
function clebsch_gordan(j1::Int, m1::Int, j2::Int, m2::Int, J::Int, M::Int)
    m1 + m2 != M && return 0.0
    phase = iseven(j1 - j2 + M) ? 1.0 : -1.0
    phase * sqrt(2J + 1) * wigner_3j(j1, j2, J, m1, m2, -M)
end

"""
    wigner_6j(j1, j2, j3, j4, j5, j6)

Wigner 6-j symbol {j1 j2 j3; j4 j5 j6} via the Racah formula in log-space.
"""
function wigner_6j(j1::Int, j2::Int, j3::Int, j4::Int, j5::Int, j6::Int)
    log_t1 = _log_triangle_coeff(j1, j2, j3)
    log_t2 = _log_triangle_coeff(j1, j5, j6)
    log_t3 = _log_triangle_coeff(j4, j2, j6)
    log_t4 = _log_triangle_coeff(j4, j5, j3)

    (log_t1 == -Inf || log_t2 == -Inf || log_t3 == -Inf || log_t4 == -Inf) && return 0.0

    log_tri_sum = 0.5 * (log_t1 + log_t2 + log_t3 + log_t4)

    t_min = max(j1 + j2 + j3, j1 + j5 + j6, j4 + j2 + j6, j4 + j5 + j3)
    t_max = min(j1 + j2 + j4 + j5, j2 + j3 + j5 + j6, j1 + j3 + j4 + j6)
    t_min > t_max && return 0.0

    s = 0.0
    for t in t_min:t_max
        log_num = _log_factorial(t + 1)
        log_den = (_log_factorial(t - j1 - j2 - j3) +
                   _log_factorial(t - j1 - j5 - j6) +
                   _log_factorial(t - j4 - j2 - j6) +
                   _log_factorial(t - j4 - j5 - j3) +
                   _log_factorial(j1 + j2 + j4 + j5 - t) +
                   _log_factorial(j2 + j3 + j5 + j6 - t) +
                   _log_factorial(j1 + j3 + j4 + j6 - t))
        sign = iseven(t) ? 1.0 : -1.0
        s += sign * exp(log_num - log_den)
    end

    exp(log_tri_sum) * s
end

"""
    precompute_cg_table(F)

Precompute CG coefficients ⟨F,m1;F,m2|l,M⟩ for all valid quantum numbers.
Returns `Dict{NTuple{4,Int},Float64}` mapping `(l, M, m1, m2)` to CG value.

Only even l channels are populated (bosonic symmetry: odd l channels vanish
for identical bosons in the interaction Hamiltonian).
"""
function precompute_cg_table(F::Int)
    table = Dict{NTuple{4,Int},Float64}()
    for l in 0:2:2F
        for M in -l:l
            for m1 in -F:F
                m2 = M - m1
                abs(m2) > F && continue
                val = clebsch_gordan(F, m1, F, m2, l, M)
                abs(val) < 1e-15 && continue
                table[(l, M, m1, m2)] = val
            end
        end
    end
    table
end

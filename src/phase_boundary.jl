"""
    find_phase_boundary(; param_range, make_interactions, grid, atom, tol, max_bisections, kwargs...)

Find a phase boundary by bisection within `param_range`.
Uses `find_ground_state` + `classify_phase` at each probe point.

Returns `(boundary_value, phase_left, phase_right, n_evaluations)`.
"""
function find_phase_boundary(;
    param_range::Tuple{Float64,Float64},
    make_interactions::Function,
    grid,
    atom,
    tol::Float64=0.01,
    max_bisections::Int=20,
    kwargs...,
)
    sm = spin_matrices(atom.F)

    function _classify_at(val)
        interactions = make_interactions(val)
        r = find_ground_state(; grid, atom, interactions, kwargs...)
        classify_phase(r.workspace.state.psi, atom.F, grid, sm).phase
    end

    a, b = param_range
    phase_a = _classify_at(a)
    phase_b = _classify_at(b)
    n_eval = 2

    phase_a != phase_b || return (
        boundary_value=(a + b) / 2,
        phase_left=phase_a,
        phase_right=phase_b,
        n_evaluations=n_eval,
    )

    for _ in 1:max_bisections
        abs(b - a) < tol && break
        mid = (a + b) / 2
        phase_mid = _classify_at(mid)
        n_eval += 1

        if phase_mid == phase_a
            a = mid
        else
            b = mid
            phase_b = phase_mid
        end
    end

    (
        boundary_value=(a + b) / 2,
        phase_left=phase_a,
        phase_right=phase_b,
        n_evaluations=n_eval,
    )
end

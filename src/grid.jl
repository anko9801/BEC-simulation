function make_grid(config::GridConfig{N}) where {N}
    x = ntuple(N) do d
        n = config.n_points[d]
        L = config.box_size[d]
        dx = L / n
        collect(range(-L / 2 + dx / 2, L / 2 - dx / 2, length=n))
    end

    dx = ntuple(d -> config.box_size[d] / config.n_points[d], N)

    k = ntuple(N) do d
        n = config.n_points[d]
        L = config.box_size[d]
        dk = 2π / L
        collect(fftfreq(n, n * dk))
    end

    dk = ntuple(d -> 2π / config.box_size[d], N)

    k_squared = _compute_k_squared(k, config.n_points)

    Grid{N}(config, x, dx, k, dk, k_squared)
end

function _compute_k_squared(k::NTuple{1,Vector{Float64}}, ::NTuple{1,Int})
    k[1] .^ 2
end

function _compute_k_squared(k::NTuple{2,Vector{Float64}}, n_points::NTuple{2,Int})
    ksq = zeros(Float64, n_points)
    kx, ky = k
    for j in 1:n_points[2], i in 1:n_points[1]
        ksq[i, j] = kx[i]^2 + ky[j]^2
    end
    ksq
end

function make_fft_plans(spatial_shape::NTuple{N,Int}) where {N}
    buf = zeros(ComplexF64, spatial_shape)
    fwd = plan_fft!(buf)
    inv = plan_ifft!(buf)
    FFTPlans(fwd, inv)
end

function cell_volume(grid::Grid{N}) where {N}
    prod(grid.dx)
end

function n_spatial_points(grid::Grid{N}) where {N}
    prod(grid.config.n_points)
end

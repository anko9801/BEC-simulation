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

function _compute_k_squared(k::NTuple{N,Vector{Float64}}, n_points::NTuple{N,Int}) where {N}
    ksq = zeros(Float64, n_points)
    @inbounds for I in CartesianIndices(n_points)
        s = 0.0
        for d in 1:N
            s += k[d][I[d]]^2
        end
        ksq[I] = s
    end
    ksq
end

function make_fft_plans(spatial_shape::NTuple{N,Int}; flags=FFTW.MEASURE) where {N}
    buf = zeros(ComplexF64, spatial_shape)
    fwd = plan_fft!(buf; flags=flags)
    inv = plan_ifft!(buf; flags=flags)
    FFTPlans(fwd, inv)
end

rfft_output_shape(n_pts::NTuple{N,Int}) where {N} = (n_pts[1] ÷ 2 + 1, n_pts[2:end]...)

function make_rfft_plans(spatial_shape::NTuple{N,Int}; flags=FFTW.MEASURE) where {N}
    rk_shape = rfft_output_shape(spatial_shape)
    real_buf = zeros(Float64, spatial_shape)
    complex_buf = zeros(ComplexF64, rk_shape)
    fwd = plan_rfft(real_buf; flags=flags)
    inv = plan_irfft(complex_buf, spatial_shape[1]; flags=flags)
    RFFTPlans{N,typeof(fwd),typeof(inv)}(fwd, inv, rk_shape)
end

function cell_volume(grid::Grid{N}) where {N}
    prod(grid.dx)
end

function n_spatial_points(grid::Grid{N}) where {N}
    prod(grid.config.n_points)
end

function load_fftw_wisdom(path::AbstractString)
    isfile(path) && FFTW.import_wisdom(path)
end

function save_fftw_wisdom(path::AbstractString)
    FFTW.export_wisdom(path)
end

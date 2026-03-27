"""
FFT-based partial derivative ∂f/∂x_dim of an N-dim real array.
"""
function _fft_partial_derivative(f::AbstractArray{Float64,N}, grid::Grid{N},
                                 plans::FFTPlans, dim::Int) where {N}
    n_pts = size(f)
    buf = Array{ComplexF64,N}(undef, n_pts)
    buf .= f
    plans.forward * buf
    @inbounds for I in CartesianIndices(n_pts)
        buf[I] = im * grid.k[dim][I[dim]] * buf[I]
    end
    plans.inverse * buf
    result = zeros(Float64, n_pts)
    @inbounds for I in CartesianIndices(n_pts)
        result[I] = real(buf[I])
    end
    result
end

"""
FFT-based gradient of an N-dim real array.
Returns `NTuple{N, Array{Float64,N}}`. Reuses a single forward FFT.
"""
function _fft_gradient(f::AbstractArray{Float64,N}, grid::Grid{N},
                       plans::FFTPlans) where {N}
    n_pts = size(f)
    f_k = Array{ComplexF64,N}(undef, n_pts)
    f_k .= f
    plans.forward * f_k

    buf = Array{ComplexF64,N}(undef, n_pts)
    ntuple(N) do dim
        @inbounds for I in CartesianIndices(n_pts)
            buf[I] = im * grid.k[dim][I[dim]] * f_k[I]
        end
        plans.inverse * buf
        result = zeros(Float64, n_pts)
        @inbounds for I in CartesianIndices(n_pts)
            result[I] = real(buf[I])
        end
        result
    end
end

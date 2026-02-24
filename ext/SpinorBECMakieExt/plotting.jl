const COMPONENT_COLORS = [:blue, :green, :red, :orange, :purple, :cyan, :magenta]
const COMPONENT_LABELS = ["m=+F", "m=+F-1", "m=0", "m=-F+1", "m=-F"]

function _component_label(sys::SpinorBEC.SpinSystem, c::Int)
    "m=$(sys.m_values[c])"
end

function SpinorBEC.plot_density(
    grid::SpinorBEC.Grid{1},
    psi::AbstractArray{ComplexF64};
    components::Bool=true,
    title::String="Density",
)
    x = grid.x[1]
    ndim = 1
    nc = size(psi, ndim + 1)

    fig = Figure(size=(800, 500))
    ax = Axis(fig[1, 1]; xlabel="x", ylabel="n(x)", title)

    n_total = SpinorBEC.total_density(psi, ndim)
    lines!(ax, x, n_total; color=:black, linewidth=2, label="total")

    if components
        for c in 1:nc
            nc_density = SpinorBEC.component_density(psi, ndim, c)
            color = COMPONENT_COLORS[mod1(c, length(COMPONENT_COLORS))]
            lines!(ax, x, nc_density; color, linewidth=1.5, linestyle=:dash, label="m=$(nc ÷ 2 + 1 - c)")
        end
    end

    axislegend(ax; position=:rt)
    fig
end

function SpinorBEC.plot_density(
    grid::SpinorBEC.Grid{2},
    psi::AbstractArray{ComplexF64};
    components::Bool=false,
    title::String="Density",
)
    x, y = grid.x
    ndim = 2
    nc = size(psi, ndim + 1)

    n_total = SpinorBEC.total_density(psi, ndim)

    if components
        fig = Figure(size=(300 * (nc + 1), 400))
        ax = Axis(fig[1, 1]; xlabel="x", ylabel="y", title="Total", aspect=DataAspect())
        heatmap!(ax, x, y, n_total)
        for c in 1:nc
            nc_density = SpinorBEC.component_density(psi, ndim, c)
            ax_c = Axis(fig[1, c + 1]; xlabel="x", ylabel="y",
                title="m=$(nc ÷ 2 + 1 - c)", aspect=DataAspect())
            heatmap!(ax_c, x, y, nc_density)
        end
    else
        fig = Figure(size=(600, 500))
        ax = Axis(fig[1, 1]; xlabel="x", ylabel="y", title, aspect=DataAspect())
        hm = heatmap!(ax, x, y, n_total)
        Colorbar(fig[1, 2], hm)
    end

    fig
end

function SpinorBEC.plot_spinor(
    grid::SpinorBEC.Grid{1},
    psi::AbstractArray{ComplexF64};
    title::String="Spinor Components",
)
    x = grid.x[1]
    nc = size(psi, 2)

    fig = Figure(size=(800, 600))

    ax_amp = Axis(fig[1, 1]; xlabel="x", ylabel="|ψ|", title="$title - Amplitude")
    ax_phase = Axis(fig[2, 1]; xlabel="x", ylabel="arg(ψ)")

    for c in 1:nc
        color = COMPONENT_COLORS[mod1(c, length(COMPONENT_COLORS))]
        label = "m=$(nc ÷ 2 + 1 - c)"
        lines!(ax_amp, x, abs.(psi[:, c]); color, linewidth=1.5, label)
        lines!(ax_phase, x, angle.(psi[:, c]); color, linewidth=1.5, label)
    end

    axislegend(ax_amp; position=:rt)
    fig
end

function SpinorBEC.plot_spin_texture(
    grid::SpinorBEC.Grid{1},
    psi::AbstractArray{ComplexF64},
    sm::SpinorBEC.SpinMatrices;
    title::String="Spin Texture",
)
    x = grid.x[1]
    fx, fy, fz = SpinorBEC.spin_density_vector(psi, sm, 1)
    n = SpinorBEC.total_density(psi, 1)
    threshold = maximum(n) * 1e-6

    fx_norm = [ni > threshold ? fi / ni : 0.0 for (fi, ni) in zip(fx, n)]
    fy_norm = [ni > threshold ? fi / ni : 0.0 for (fi, ni) in zip(fy, n)]
    fz_norm = [ni > threshold ? fi / ni : 0.0 for (fi, ni) in zip(fz, n)]

    fig = Figure(size=(800, 500))
    ax = Axis(fig[1, 1]; xlabel="x", ylabel="⟨F_α⟩/n", title)

    lines!(ax, x, fx_norm; color=:red, linewidth=1.5, label="Fx/n")
    lines!(ax, x, fy_norm; color=:green, linewidth=1.5, label="Fy/n")
    lines!(ax, x, fz_norm; color=:blue, linewidth=1.5, label="Fz/n")

    axislegend(ax; position=:rt)
    fig
end

function SpinorBEC.plot_spin_texture(
    grid::SpinorBEC.Grid{2},
    psi::AbstractArray{ComplexF64},
    sm::SpinorBEC.SpinMatrices;
    title::String="Spin Texture",
)
    x, y = grid.x
    fx, fy, fz = SpinorBEC.spin_density_vector(psi, sm, 2)
    n = SpinorBEC.total_density(psi, 2)
    threshold = maximum(n) * 1e-6

    fz_norm = @. ifelse(n > threshold, fz / n, 0.0)

    fig = Figure(size=(600, 500))
    ax = Axis(fig[1, 1]; xlabel="x", ylabel="y", title="$title: Fz/n", aspect=DataAspect())
    hm = heatmap!(ax, x, y, fz_norm; colorrange=(-1, 1), colormap=:RdBu)
    Colorbar(fig[1, 2], hm)
    fig
end

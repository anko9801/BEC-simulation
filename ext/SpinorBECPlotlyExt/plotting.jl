const COMPONENT_COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#9467bd", "#17becf", "#e377c2"]

function SpinorBEC.plot_density(
    grid::SpinorBEC.Grid{1},
    psi::AbstractArray{ComplexF64};
    components::Bool=true,
    title::String="Density",
)
    x = grid.x[1]
    ndim = 1
    nc = size(psi, ndim + 1)

    n_total = SpinorBEC.total_density(psi, ndim)

    traces = GenericTrace[
        scatter(
            x=x, y=collect(n_total),
            mode="lines", name="total",
            line=attr(color="black", width=2),
        )
    ]

    if components
        for c in 1:nc
            nc_density = SpinorBEC.component_density(psi, ndim, c)
            m_val = nc ÷ 2 + 1 - c
            color = COMPONENT_COLORS[mod1(c, length(COMPONENT_COLORS))]
            push!(traces, scatter(
                x=x, y=collect(nc_density),
                mode="lines", name="m=$m_val",
                line=attr(color=color, width=1.5, dash="dash"),
            ))
        end
    end

    layout = Layout(
        title=title,
        xaxis_title="x",
        yaxis_title="n(x)",
        legend=attr(x=0.02, y=0.98),
    )

    plot(traces, layout)
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
        p = make_subplots(
            rows=1, cols=nc + 1,
            subplot_titles=vcat(["Total"], ["m=$(nc ÷ 2 + 1 - c)" for c in 1:nc]),
        )
        add_trace!(p, heatmap(x=x, y=y, z=collect(n_total'), colorscale="Viridis"), row=1, col=1)
        for c in 1:nc
            nc_density = SpinorBEC.component_density(psi, ndim, c)
            add_trace!(p, heatmap(x=x, y=y, z=collect(nc_density'), colorscale="Viridis", showscale=false), row=1, col=c + 1)
        end
        relayout!(p, title_text=title, height=400, width=300 * (nc + 1))
        return p
    end

    trace = heatmap(
        x=x, y=y, z=collect(n_total'),
        colorscale="Viridis",
        colorbar=attr(title="n"),
    )

    layout = Layout(
        title=title,
        xaxis_title="x", yaxis_title="y",
        yaxis_scaleanchor="x",
        width=600, height=500,
    )

    plot(trace, layout)
end

function SpinorBEC.plot_spinor(
    grid::SpinorBEC.Grid{1},
    psi::AbstractArray{ComplexF64};
    title::String="Spinor Components",
)
    x = grid.x[1]
    nc = size(psi, 2)

    p = make_subplots(rows=2, cols=1, shared_xaxes=true, vertical_spacing=0.08,
        subplot_titles=["Amplitude" "Phase"])

    for c in 1:nc
        m_val = nc ÷ 2 + 1 - c
        color = COMPONENT_COLORS[mod1(c, length(COMPONENT_COLORS))]

        add_trace!(p, scatter(
            x=x, y=abs.(psi[:, c]),
            mode="lines", name="m=$m_val",
            line=attr(color=color, width=1.5),
            legendgroup="m$m_val",
        ), row=1, col=1)

        add_trace!(p, scatter(
            x=x, y=angle.(psi[:, c]),
            mode="lines", name="m=$m_val (phase)",
            line=attr(color=color, width=1.5, dash="dot"),
            legendgroup="m$m_val", showlegend=false,
        ), row=2, col=1)
    end

    relayout!(p, title_text=title, height=600, width=800,
        xaxis2_title="x", yaxis_title="|ψ|", yaxis2_title="arg(ψ)")
    p
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

    fx_n = [ni > threshold ? fi / ni : 0.0 for (fi, ni) in zip(fx, n)]
    fy_n = [ni > threshold ? fi / ni : 0.0 for (fi, ni) in zip(fy, n)]
    fz_n = [ni > threshold ? fi / ni : 0.0 for (fi, ni) in zip(fz, n)]

    traces = [
        scatter(x=x, y=fx_n, mode="lines", name="Fx/n", line=attr(color="red", width=1.5)),
        scatter(x=x, y=fy_n, mode="lines", name="Fy/n", line=attr(color="green", width=1.5)),
        scatter(x=x, y=fz_n, mode="lines", name="Fz/n", line=attr(color="blue", width=1.5)),
    ]

    layout = Layout(
        title=title,
        xaxis_title="x", yaxis_title="⟨Fα⟩/n",
        legend=attr(x=0.02, y=0.98),
    )

    plot(traces, layout)
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

    fz_n = @. ifelse(n > threshold, fz / n, 0.0)

    trace = heatmap(
        x=x, y=y, z=collect(fz_n'),
        colorscale="RdBu", zmin=-1, zmax=1,
        colorbar=attr(title="Fz/n"),
    )

    layout = Layout(
        title="$title: Fz/n",
        xaxis_title="x", yaxis_title="y",
        yaxis_scaleanchor="x",
        width=600, height=500,
    )

    plot(trace, layout)
end

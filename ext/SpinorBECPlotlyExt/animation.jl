function SpinorBEC.animate_dynamics(
    grid::SpinorBEC.Grid{1},
    result::SpinorBEC.SimulationResult;
    title::String="Dynamics",
    filename::Union{Nothing,String}=nothing,
)
    x = grid.x[1]
    snapshots = result.psi_snapshots
    times = result.times
    n_frames = length(snapshots)
    nc = size(snapshots[1], 2)

    n_max = maximum(maximum(SpinorBEC.total_density(s, 1)) for s in snapshots)

    initial_traces = GenericTrace[]

    n0 = SpinorBEC.total_density(snapshots[1], 1)
    push!(initial_traces, scatter(
        x=x, y=collect(n0),
        mode="lines", name="total",
        line=attr(color="black", width=2),
    ))

    for c in 1:nc
        nc0 = SpinorBEC.component_density(snapshots[1], 1, c)
        m_val = nc ÷ 2 + 1 - c
        color = COMPONENT_COLORS[mod1(c, length(COMPONENT_COLORS))]
        push!(initial_traces, scatter(
            x=x, y=collect(nc0),
            mode="lines", name="m=$m_val",
            line=attr(color=color, width=1.5, dash="dash"),
        ))
    end

    frames = PlotlyFrame[]
    for k in 1:n_frames
        n_k = SpinorBEC.total_density(snapshots[k], 1)
        frame_data = [attr(y=collect(n_k))]

        for c in 1:nc
            nc_k = SpinorBEC.component_density(snapshots[k], 1, c)
            push!(frame_data, attr(y=collect(nc_k)))
        end

        push!(frames, frame(
            data=frame_data,
            name="frame_$k",
            traces=collect(0:nc),
        ))
    end

    slider_steps = [
        attr(
            label="$(round(times[k], digits=4))",
            method="animate",
            args=[
                ["frame_$k"],
                attr(mode="immediate", frame=attr(duration=50, redraw=true), transition=attr(duration=0)),
            ],
        )
        for k in 1:n_frames
    ]

    updatemenus = [attr(
        type="buttons", showactive=false,
        xanchor="center", yanchor="top", x=0.5, y=-0.12,
        buttons=[
            attr(label="▶ Play", method="animate", args=[
                nothing,
                attr(fromcurrent=true, frame=attr(duration=50, redraw=true), transition=attr(duration=0)),
            ]),
            attr(label="⏸ Pause", method="animate", args=[
                [nothing],
                attr(mode="immediate", frame=attr(duration=0, redraw=false), transition=attr(duration=0)),
            ]),
        ],
    )]

    layout = Layout(
        title=title,
        xaxis_title="x", yaxis_title="n(x)",
        yaxis_range=[0, n_max * 1.1],
        sliders=[attr(
            active=0, steps=slider_steps,
            currentvalue=attr(prefix="t = "),
            pad=attr(t=60),
        )],
        updatemenus=updatemenus,
        legend=attr(x=0.02, y=0.98),
    )

    p = Plot(initial_traces, layout, frames)

    if filename !== nothing
        savefig(p, filename)
    end

    p
end

function SpinorBEC.animate_dynamics(
    grid::SpinorBEC.Grid{2},
    result::SpinorBEC.SimulationResult;
    title::String="Dynamics",
    filename::Union{Nothing,String}=nothing,
)
    x, y = grid.x
    snapshots = result.psi_snapshots
    times = result.times
    n_frames = length(snapshots)

    n_max = maximum(maximum(SpinorBEC.total_density(s, 2)) for s in snapshots)

    n0 = SpinorBEC.total_density(snapshots[1], 2)
    initial_trace = heatmap(
        x=x, y=y, z=collect(n0'),
        colorscale="Viridis", zmin=0, zmax=n_max,
        colorbar=attr(title="n"),
    )

    frames = PlotlyFrame[]
    for k in 1:n_frames
        n_k = SpinorBEC.total_density(snapshots[k], 2)
        push!(frames, frame(
            data=[attr(z=[collect(n_k')])],
            name="frame_$k",
            traces=[0],
        ))
    end

    slider_steps = [
        attr(
            label="$(round(times[k], digits=4))",
            method="animate",
            args=[
                ["frame_$k"],
                attr(mode="immediate", frame=attr(duration=50, redraw=true), transition=attr(duration=0)),
            ],
        )
        for k in 1:n_frames
    ]

    updatemenus = [attr(
        type="buttons", showactive=false,
        xanchor="center", yanchor="top", x=0.5, y=-0.12,
        buttons=[
            attr(label="▶ Play", method="animate", args=[
                nothing,
                attr(fromcurrent=true, frame=attr(duration=50, redraw=true), transition=attr(duration=0)),
            ]),
            attr(label="⏸ Pause", method="animate", args=[
                [nothing],
                attr(mode="immediate", frame=attr(duration=0, redraw=false), transition=attr(duration=0)),
            ]),
        ],
    )]

    layout = Layout(
        title=title,
        xaxis_title="x", yaxis_title="y",
        yaxis_scaleanchor="x",
        width=600, height=550,
        sliders=[attr(
            active=0, steps=slider_steps,
            currentvalue=attr(prefix="t = "),
            pad=attr(t=60),
        )],
        updatemenus=updatemenus,
    )

    p = Plot(initial_trace, layout, frames)

    if filename !== nothing
        savefig(p, filename)
    end

    p
end

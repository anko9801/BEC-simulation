function SpinorBEC.animate_dynamics(
    grid::SpinorBEC.Grid{1},
    result::SpinorBEC.SimulationResult;
    title::String="Dynamics",
    fps::Int=30,
    filename::Union{Nothing,String}=nothing,
)
    x = grid.x[1]
    snapshots = result.psi_snapshots
    times = result.times
    n_frames = length(snapshots)
    nc = size(snapshots[1], 2)

    n_max = maximum(maximum(SpinorBEC.total_density(s, 1)) for s in snapshots)

    fig = Figure(size=(800, 500))
    ax = Axis(fig[1, 1]; xlabel="x", ylabel="n(x)", title,
        limits=(nothing, (0, n_max * 1.1)))

    time_label = Label(fig[0, 1], "t = $(round(times[1], digits=4))")

    n_total = Observable(SpinorBEC.total_density(snapshots[1], 1))
    comp_densities = [Observable(SpinorBEC.component_density(snapshots[1], 1, c)) for c in 1:nc]

    lines!(ax, x, n_total; color=:black, linewidth=2, label="total")
    for c in 1:nc
        color = COMPONENT_COLORS[mod1(c, length(COMPONENT_COLORS))]
        lines!(ax, x, comp_densities[c]; color, linewidth=1.5, linestyle=:dash,
            label="m=$(nc ÷ 2 + 1 - c)")
    end
    axislegend(ax; position=:rt)

    if filename !== nothing
        record(fig, filename, 1:n_frames; framerate=fps) do frame
            n_total[] = SpinorBEC.total_density(snapshots[frame], 1)
            for c in 1:nc
                comp_densities[c][] = SpinorBEC.component_density(snapshots[frame], 1, c)
            end
            time_label.text[] = "t = $(round(times[frame], digits=4))"
        end
    else
        frame_idx = Observable(1)
        on(frame_idx) do i
            n_total[] = SpinorBEC.total_density(snapshots[i], 1)
            for c in 1:nc
                comp_densities[c][] = SpinorBEC.component_density(snapshots[i], 1, c)
            end
            time_label.text[] = "t = $(round(times[i], digits=4))"
        end

        sl = Slider(fig[2, 1]; range=1:n_frames, startvalue=1)
        connect!(frame_idx, sl.value)
    end

    fig
end

function SpinorBEC.animate_dynamics(
    grid::SpinorBEC.Grid{2},
    result::SpinorBEC.SimulationResult;
    title::String="Dynamics",
    fps::Int=30,
    filename::Union{Nothing,String}=nothing,
)
    x, y = grid.x
    snapshots = result.psi_snapshots
    times = result.times
    n_frames = length(snapshots)

    n_max = maximum(maximum(SpinorBEC.total_density(s, 2)) for s in snapshots)

    fig = Figure(size=(600, 550))
    ax = Axis(fig[1, 1]; xlabel="x", ylabel="y", title, aspect=DataAspect())
    time_label = Label(fig[0, 1], "t = $(round(times[1], digits=4))")

    n_obs = Observable(SpinorBEC.total_density(snapshots[1], 2))
    hm = heatmap!(ax, x, y, n_obs; colorrange=(0, n_max))
    Colorbar(fig[1, 2], hm)

    if filename !== nothing
        record(fig, filename, 1:n_frames; framerate=fps) do frame
            n_obs[] = SpinorBEC.total_density(snapshots[frame], 2)
            time_label.text[] = "t = $(round(times[frame], digits=4))"
        end
    else
        frame_idx = Observable(1)
        on(frame_idx) do i
            n_obs[] = SpinorBEC.total_density(snapshots[i], 2)
            time_label.text[] = "t = $(round(times[i], digits=4))"
        end

        sl = Slider(fig[2, 1]; range=1:n_frames, startvalue=1)
        connect!(frame_idx, sl.value)
    end

    fig
end

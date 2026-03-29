using JLD2

data = load("examples/eu151/edh_full_64.jld2")
snaps = data["snapshots"]
times_ms = sort(collect(keys(snaps)))
x = data["x_coords"]
z = data["z_coords"]
D = 13

function arr_json(io, arr)
    print(io, "[")
    for (i, v) in enumerate(arr)
        i > 1 && print(io, ",")
        isnan(v) || isinf(v) ? print(io, "0") : print(io, v)
    end
    print(io, "]")
end

mat_flat(m) = [m[i, j] for j in 1:size(m, 2) for i in 1:size(m, 1)]

open("examples/eu151/edh_full_64.json", "w") do f
    print(f, "{")
    print(f, "\"x\":"); arr_json(f, x); print(f, ",")
    print(f, "\"z\":"); arr_json(f, z); print(f, ",")
    print(f, "\"N\":64,\"D\":13,")
    print(f, "\"times\":"); arr_json(f, times_ms); print(f, ",")

    print(f, "\"frames\":[")
    for (ti, t) in enumerate(times_ms)
        ti > 1 && print(f, ",")
        s = snaps[t]
        print(f, "{")
        print(f, "\"t_ms\":", t, ",")
        print(f, "\"Sz\":", s.Sz, ",")
        print(f, "\"norm\":", s.norm, ",")
        print(f, "\"pops\":"); arr_json(f, s.pops); print(f, ",")
        print(f, "\"n_x\":"); arr_json(f, s.n_x); print(f, ",")
        print(f, "\"n_z\":"); arr_json(f, s.n_z); print(f, ",")
        print(f, "\"slice_xy\":"); arr_json(f, mat_flat(s.slice_xy)); print(f, ",")
        print(f, "\"slice_xz\":"); arr_json(f, mat_flat(s.slice_xz)); print(f, ",")
        print(f, "\"n_xy\":"); arr_json(f, mat_flat(s.n_xy)); print(f, ",")
        print(f, "\"n_xz\":"); arr_json(f, mat_flat(s.n_xz)); print(f, ",")

        # Per-component column densities (xy view)
        print(f, "\"comp_col_xy\":[")
        for c in 1:D
            c > 1 && print(f, ",")
            arr_json(f, mat_flat(s.comp_col_xy[c]))
        end
        print(f, "],")

        # Per-component 1D profiles (x-axis)
        print(f, "\"comp_n_x\":[")
        for c in 1:D
            c > 1 && print(f, ",")
            arr_json(f, s.comp_n_x[c])
        end
        print(f, "]")

        print(f, "}")
    end
    print(f, "]}")
end
println("Exported edh_full_64.json")

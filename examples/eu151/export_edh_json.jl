using JLD2

data = load("examples/eu151/edh_full_64.jld2")
snaps = data["snapshots"]
times_ms = sort(collect(keys(snaps)))
x = data["x_coords"]
z = data["z_coords"]

function arr_to_json(io, arr)
    print(io, "[")
    for (i, v) in enumerate(arr)
        i > 1 && print(io, ",")
        isnan(v) || isinf(v) ? print(io, "0") : print(io, v)
    end
    print(io, "]")
end

function mat_to_flat(m)
    [m[i, j] for j in 1:size(m, 2) for i in 1:size(m, 1)]
end

open("examples/eu151/edh_full_64.json", "w") do f
    print(f, "{")
    print(f, "\"x\":"); arr_to_json(f, x); print(f, ",")
    print(f, "\"z\":"); arr_to_json(f, z); print(f, ",")
    print(f, "\"N\":64,\"box\":20.0,")

    print(f, "\"times\":")
    arr_to_json(f, times_ms)
    print(f, ",")

    print(f, "\"frames\":[")
    for (ti, t) in enumerate(times_ms)
        ti > 1 && print(f, ",")
        s = snaps[t]
        print(f, "{")
        print(f, "\"t_ms\":", t, ",")
        print(f, "\"Sz\":", s.Sz, ",")
        print(f, "\"norm\":", s.norm, ",")
        print(f, "\"pops\":"); arr_to_json(f, s.pops); print(f, ",")
        print(f, "\"n_x\":"); arr_to_json(f, s.n_x); print(f, ",")
        print(f, "\"n_z\":"); arr_to_json(f, s.n_z); print(f, ",")
        print(f, "\"slice_xy\":"); arr_to_json(f, mat_to_flat(s.slice_xy)); print(f, ",")
        print(f, "\"slice_xz\":"); arr_to_json(f, mat_to_flat(s.slice_xz)); print(f, ",")
        print(f, "\"n_xy\":"); arr_to_json(f, mat_to_flat(s.n_xy)); print(f, ",")
        print(f, "\"n_xz\":"); arr_to_json(f, mat_to_flat(s.n_xz))
        print(f, "}")
    end
    print(f, "]}")
end
println("Exported edh_full_64.json")

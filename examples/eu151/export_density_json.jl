using JLD2

data = load("examples/eu151/density_64.jld2")

function write_json_array(io, arr)
    print(io, "[")
    for (i, v) in enumerate(arr)
        i > 1 && print(io, ",")
        if isnan(v) || isinf(v)
            print(io, "0")
        else
            print(io, v)
        end
    end
    print(io, "]")
end

open("examples/eu151/density_64.json", "w") do f
    print(f, "{")

    # 1D coords and profiles
    for (key, val) in [
        ("x", collect(data["x_coords"])),
        ("y", collect(data["y_coords"])),
        ("z", collect(data["z_coords"])),
        ("n_x", collect(data["n_x"])),
        ("n_y", collect(data["n_y"])),
        ("n_z", collect(data["n_z"])),
        ("comp_pops", collect(data["comp_pops"])),
    ]
        print(f, "\"$key\":")
        write_json_array(f, val)
        print(f, ",")
    end

    print(f, "\"n_peak\":", maximum(data["n_total"]), ",")

    # 2D slices through center (row-major for JS)
    n_total = data["n_total"]
    cx = size(n_total, 1) ÷ 2 + 1

    for (key, slice) in [
        ("slice_xy", n_total[:, :, cx]),
        ("slice_xz", n_total[:, cx, :]),
        ("slice_yz", n_total[cx, :, :]),
        ("n_xy", data["n_xy"]),
        ("n_xz", data["n_xz"]),
    ]
        flat = [slice[i, j] for j in 1:size(slice, 2) for i in 1:size(slice, 1)]
        print(f, "\"$key\":")
        write_json_array(f, flat)
        print(f, ",\"$(key)_shape\":[$(size(slice,2)),$(size(slice,1))],")
    end

    print(f, "\"N\":64,\"box\":20.0}")
end
println("Exported to examples/eu151/density_64.json")

#!/usr/bin/env julia
using DelimitedFiles
using Plots
gr()

println("Loading data...")

# データ読み込み
xy_data = readdlm("density_xy.dat")
xz_data = readdlm("density_xz.dat")
hist_data = readdlm("energy_history.dat")

# XY平面の密度をリシェイプ
N = Int(sqrt(size(xy_data, 1)))
x = xy_data[1:N, 1]
y = unique(xy_data[:, 2])
density_xy = reshape(xy_data[:, 3], N, N)

# XZ平面
Nx = length(unique(xz_data[:, 1]))
Nz = length(unique(xz_data[:, 2]))
x_xz = unique(xz_data[:, 1])
z_xz = unique(xz_data[:, 2])
density_xz = reshape(xz_data[:, 3], Nx, Nz)

println("Creating plots...")

# 密度プロット
p1 = heatmap(x, y, density_xy',
             xlabel="x", ylabel="y",
             title="Density (z=0 plane)",
             color=:viridis, aspect_ratio=1)

p2 = heatmap(x_xz, z_xz, density_xz',
             xlabel="x", ylabel="z",
             title="Density (y=0 plane)",
             color=:viridis, aspect_ratio=:auto)

# エネルギー履歴
p3 = plot(hist_data[:,1], hist_data[:,2],
          xlabel="Step", ylabel="E/N",
          title="Energy Convergence", lw=2, legend=false,
          color=:blue)

p4 = plot(hist_data[:,1], hist_data[:,3],
          xlabel="Step", ylabel="E_dd/N",
          title="DDI Energy", lw=2, legend=false,
          color=:red)

# 結合プロット
combined = plot(p1, p2, p3, p4, layout=(2,2), size=(1000,800))
savefig(combined, "bec_results.png")
println("Saved: bec_results.png")

# 3D surface plot
p3d = surface(x, y, density_xy',
              xlabel="x", ylabel="y", zlabel="n",
              title="3D Density (z=0)",
              color=:viridis, camera=(30,30))
savefig(p3d, "density_3d.png")
println("Saved: density_3d.png")

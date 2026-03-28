include(joinpath(@__DIR__, "eu151_setup.jl"))
using Printf, JLD2

const N_GRID = 64
const BOX = 20.0

println("=" ^ 60)
@printf("  Eu151 Density Distribution (%d³, box=%.0f)\n", N_GRID, BOX)
println("=" ^ 60)

grid = make_grid(GridConfig(ntuple(_ -> N_GRID, 3), ntuple(_ -> BOX, 3)))
trap = HarmonicTrap((1.0, 1.0, EU_λ_z))
psi = load_or_compute_gs(grid; trap)

dV = cell_volume(grid)
norm = sum(abs2, psi) * dV
@printf("  norm = %.8f\n", norm)

# Total density n(r) = Σ_c |ψ_c(r)|²
n_total = SpinorBEC.total_density(psi, 3)
@printf("  n_peak = %.4e\n", maximum(n_total))
@printf("  n_total integral = %.8f\n", sum(n_total) * dV)

# Per-component densities
D = 13
comp_dens = [SpinorBEC.component_density(psi, 3, c) for c in 1:D]
for c in 1:D
    m = 6 - (c - 1)
    pop = sum(comp_dens[c]) * dV
    peak = maximum(comp_dens[c])
    @printf("  m=%+3d: pop=%.6f, peak=%.4e\n", m, pop, peak)
end

# Column-integrated densities (absorption images)
# xy-plane (integrate along z)
n_xy = dropdims(sum(n_total, dims=3), dims=3) .* grid.dx[3]
# xz-plane (integrate along y)
n_xz = dropdims(sum(n_total, dims=2), dims=2) .* grid.dx[2]

# 1D cuts through center
cx = N_GRID ÷ 2 + 1
n_x = n_total[:, cx, cx]
n_y = n_total[cx, :, cx]
n_z = n_total[cx, cx, :]

x_coords = grid.x[1]
y_coords = grid.x[2]
z_coords = grid.x[3]

# Save results
outfile = joinpath(@__DIR__, "density_64.jld2")
jldsave(outfile;
    n_total, n_xy, n_xz,
    n_x, n_y, n_z,
    x_coords, y_coords, z_coords,
    comp_dens_peak=[maximum(comp_dens[c]) for c in 1:D],
    comp_pops=[sum(comp_dens[c]) * dV for c in 1:D],
    grid_info=(N=N_GRID, box=BOX, dx=grid.dx),
)
@printf("\nSaved to %s\n", outfile)

# Print summary: Thomas-Fermi radii
println("\n--- Spatial extent (FWHM from 1D cuts) ---")
for (label, coords, profile) in [("x", x_coords, n_x), ("y", y_coords, n_y), ("z", z_coords, n_z)]
    half_max = maximum(profile) / 2
    above = findall(p -> p > half_max, profile)
    if !isempty(above)
        fwhm = coords[above[end]] - coords[above[1]]
        @printf("  %s: FWHM = %.3f a_ho (%.3f μm)\n", label, fwhm, fwhm * 0.780)
    end
end

println("\nDone.")

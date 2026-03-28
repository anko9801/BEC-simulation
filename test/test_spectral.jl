@testset "Spectral Analysis" begin
    @testset "Single frequency detection" begin
        f0 = 5.0
        dt = 0.01
        N = 1000
        times = collect(range(0, step=dt, length=N))
        signal = sin.(2π * f0 .* times)

        result = power_spectrum(times, signal; window=:hanning)

        peak_idx = argmax(result.power)
        peak_freq = result.frequencies[peak_idx]
        @test abs(peak_freq - f0) < 1.0 / (N * dt)
    end

    @testset "DC signal → peak at f=0" begin
        dt = 0.01
        N = 256
        times = collect(range(0, step=dt, length=N))
        signal = ones(Float64, N)

        result = power_spectrum(times, signal; window=:none)
        @test argmax(result.power) == 1
    end

    @testset "Two frequencies" begin
        f1, f2 = 3.0, 7.0
        dt = 0.005
        N = 2000
        times = collect(range(0, step=dt, length=N))
        signal = sin.(2π * f1 .* times) .+ 0.5 * sin.(2π * f2 .* times)

        result = power_spectrum(times, signal; window=:hanning, pad_factor=2)

        # Find top 2 peaks (exclude DC)
        power_no_dc = copy(result.power)
        power_no_dc[1] = 0.0
        idx1 = argmax(power_no_dc)
        power_no_dc[idx1] = 0.0
        # Clear neighbors
        for di in -3:3
            ii = idx1 + di
            1 <= ii <= length(power_no_dc) && (power_no_dc[ii] = 0.0)
        end
        idx2 = argmax(power_no_dc)

        freqs_found = sort([result.frequencies[idx1], result.frequencies[idx2]])
        @test abs(freqs_found[1] - f1) < 0.5
        @test abs(freqs_found[2] - f2) < 0.5
    end

    @testset "Hamming window" begin
        f0 = 10.0
        dt = 0.01
        N = 500
        times = collect(range(0, step=dt, length=N))
        signal = cos.(2π * f0 .* times)

        result = power_spectrum(times, signal; window=:hamming)
        peak_idx = argmax(result.power)
        @test abs(result.frequencies[peak_idx] - f0) < 0.5
    end

    @testset "Padding increases frequency resolution" begin
        dt = 0.01
        N = 100
        times = collect(range(0, step=dt, length=N))
        signal = sin.(2π * 5.0 .* times)

        r1 = power_spectrum(times, signal; pad_factor=1)
        r4 = power_spectrum(times, signal; pad_factor=4)

        @test length(r4.frequencies) > length(r1.frequencies)
        df1 = r1.frequencies[2] - r1.frequencies[1]
        df4 = r4.frequencies[2] - r4.frequencies[1]
        @test df4 < df1
    end

    @testset "Error handling" begin
        @test_throws DimensionMismatch power_spectrum([1.0, 2.0], [1.0])
        @test_throws ArgumentError power_spectrum([1.0], [1.0])
        @test_throws ArgumentError power_spectrum([1.0, 2.0], [1.0, 2.0]; window=:invalid)
        @test_throws ArgumentError power_spectrum([1.0, 2.0], [1.0, 2.0]; pad_factor=0)
    end
end

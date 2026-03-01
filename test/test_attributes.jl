using Test
using NeatEvolution
using Random

@testset "Attribute Tests" begin

    @testset "FloatAttribute" begin
        @testset "Construction from config dict" begin
            config = Dict(
                :weight_init_mean => 0.0,
                :weight_init_stdev => 1.0,
                :weight_init_type => "gaussian",
                :weight_replace_rate => 0.1,
                :weight_mutate_rate => 0.8,
                :weight_mutate_power => 0.5,
                :weight_min_value => -5.0,
                :weight_max_value => 5.0
            )
            attr = NeatEvolution.FloatAttribute(:weight, config)

            @test attr.name == :weight
            @test attr.init_mean == 0.0
            @test attr.init_stdev == 1.0
            @test attr.init_type == :gaussian
            @test attr.replace_rate == 0.1
            @test attr.mutate_rate == 0.8
            @test attr.mutate_power == 0.5
            @test attr.min_value == -5.0
            @test attr.max_value == 5.0
        end

        @testset "Construction with defaults" begin
            attr = NeatEvolution.FloatAttribute(:bias, Dict{Symbol,Any}())
            @test attr.name == :bias
            @test attr.init_mean == 0.0
            @test attr.init_stdev == 1.0
            @test attr.init_type == :gaussian
            @test attr.min_value == -30.0
            @test attr.max_value == 30.0
        end

        @testset "Uniform init_type parsing" begin
            config = Dict(:bias_init_type => "Uniform")
            attr = NeatEvolution.FloatAttribute(:bias, config)
            @test attr.init_type == :uniform
        end

        @testset "clamp_value" begin
            attr = NeatEvolution.FloatAttribute(:w, 0.0, 1.0, :gaussian, 0.0, 0.0, 0.0, -2.0, 2.0)
            @test NeatEvolution.clamp_value(attr, 0.0) == 0.0
            @test NeatEvolution.clamp_value(attr, 3.0) == 2.0
            @test NeatEvolution.clamp_value(attr, -3.0) == -2.0
            @test NeatEvolution.clamp_value(attr, 1.5) == 1.5
        end

        @testset "init_value - gaussian" begin
            attr = NeatEvolution.FloatAttribute(:w, 0.0, 1.0, :gaussian, 0.0, 0.0, 0.0, -5.0, 5.0)
            rng = MersenneTwister(42)
            values = [NeatEvolution.init_value(attr, rng) for _ in 1:1000]

            # All values should be within bounds
            @test all(v -> -5.0 <= v <= 5.0, values)

            # Mean should be close to init_mean
            m = sum(values) / length(values)
            @test abs(m) < 0.15  # Should be near 0.0

            # Should have variation (stdev ~ 1.0)
            variance = sum((v - m)^2 for v in values) / length(values)
            @test variance > 0.5
            println("  Gaussian init: mean=$(round(m, digits=4)), var=$(round(variance, digits=4))")
        end

        @testset "init_value - uniform" begin
            attr = NeatEvolution.FloatAttribute(:w, 0.0, 1.0, :uniform, 0.0, 0.0, 0.0, -5.0, 5.0)
            rng = MersenneTwister(42)
            values = [NeatEvolution.init_value(attr, rng) for _ in 1:1000]

            # Uniform range should be [init_mean - 2*stdev, init_mean + 2*stdev]
            # = [-2.0, 2.0]
            @test all(v -> -2.0 <= v <= 2.0, values)

            # Should be roughly uniform
            m = sum(values) / length(values)
            @test abs(m) < 0.15
            println("  Uniform init: mean=$(round(m, digits=4)), range=[$(round(minimum(values), digits=4)), $(round(maximum(values), digits=4))]")
        end

        @testset "init_value - unknown type errors" begin
            attr = NeatEvolution.FloatAttribute(:w, 0.0, 1.0, :unknown, 0.0, 0.0, 0.0, -5.0, 5.0)
            @test_throws ErrorException NeatEvolution.init_value(attr)
        end

        @testset "mutate_value - mutation" begin
            # High mutate_rate, zero replace_rate
            attr = NeatEvolution.FloatAttribute(:w, 0.0, 1.0, :gaussian, 0.0, 1.0, 0.5, -5.0, 5.0)
            rng = MersenneTwister(42)

            original = 1.0
            mutated_values = [NeatEvolution.mutate_value(attr, original, rng) for _ in 1:100]

            # Most values should differ from original due to 100% mutate_rate
            changed = count(v -> v != original, mutated_values)
            @test changed == 100  # 100% mutate rate means all should change

            # All should be clamped within bounds
            @test all(v -> -5.0 <= v <= 5.0, mutated_values)
        end

        @testset "mutate_value - replace" begin
            # Zero mutate_rate, high replace_rate
            attr = NeatEvolution.FloatAttribute(:w, 0.0, 1.0, :gaussian, 1.0, 0.0, 0.5, -5.0, 5.0)
            rng = MersenneTwister(42)

            original = 3.0
            replaced_values = [NeatEvolution.mutate_value(attr, original, rng) for _ in 1:100]

            # All should be replaced (since mutate_rate=0, replace_rate=1.0, r will always be
            # in the [0, 1] range, so r < 0 + 1.0 is always true after skipping mutate)
            changed = count(v -> v != original, replaced_values)
            @test changed == 100
        end

        @testset "mutate_value - no mutation" begin
            # Zero mutate and replace rates
            attr = NeatEvolution.FloatAttribute(:w, 0.0, 1.0, :gaussian, 0.0, 0.0, 0.5, -5.0, 5.0)
            rng = MersenneTwister(42)

            original = 2.5
            values = [NeatEvolution.mutate_value(attr, original, rng) for _ in 1:100]

            # No mutations should occur
            @test all(v -> v == original, values)
        end

        @testset "validate" begin
            # Valid attribute
            attr = NeatEvolution.FloatAttribute(:w, 0.0, 1.0, :gaussian, 0.0, 0.0, 0.0, -5.0, 5.0)
            NeatEvolution.validate(attr)  # Should not throw

            # Invalid: max < min
            bad_attr = NeatEvolution.FloatAttribute(:w, 0.0, 1.0, :gaussian, 0.0, 0.0, 0.0, 5.0, -5.0)
            @test_throws ErrorException NeatEvolution.validate(bad_attr)
        end

        @testset "Deterministic with same seed" begin
            attr = NeatEvolution.FloatAttribute(:w, 0.0, 1.0, :gaussian, 0.1, 0.8, 0.5, -5.0, 5.0)

            rng1 = MersenneTwister(123)
            rng2 = MersenneTwister(123)

            v1 = NeatEvolution.init_value(attr, rng1)
            v2 = NeatEvolution.init_value(attr, rng2)
            @test v1 == v2

            m1 = NeatEvolution.mutate_value(attr, 1.0, rng1)
            m2 = NeatEvolution.mutate_value(attr, 1.0, rng2)
            @test m1 == m2
        end
    end

    @testset "BoolAttribute" begin
        @testset "Construction from config dict - true default" begin
            config = Dict(
                :enabled_default => "true",
                :enabled_mutate_rate => 0.01,
                :enabled_rate_to_true_add => 0.0,
                :enabled_rate_to_false_add => 0.0
            )
            attr = NeatEvolution.BoolAttribute(:enabled, config)

            @test attr.name == :enabled
            @test attr.default == Symbol("true")
            @test attr.mutate_rate == 0.01
        end

        @testset "Construction from config dict - false default" begin
            config = Dict(:enabled_default => "false")
            attr = NeatEvolution.BoolAttribute(:enabled, config)
            @test attr.default == Symbol("false")
        end

        @testset "Construction from config dict - random default" begin
            config = Dict(:enabled_default => "random")
            attr = NeatEvolution.BoolAttribute(:enabled, config)
            @test attr.default == Symbol("random")
        end

        @testset "Construction with Bool values from TOML" begin
            # TOML parses true/false as Bool, not String
            config = Dict(:enabled_default => true)
            attr = NeatEvolution.BoolAttribute(:enabled, config)
            @test attr.default == Symbol("true")

            config2 = Dict(:enabled_default => false)
            attr2 = NeatEvolution.BoolAttribute(:enabled, config2)
            @test attr2.default == Symbol("false")
        end

        @testset "Construction with various string synonyms" begin
            for (input, expected) in [
                ("1", Symbol("true")),
                ("on", Symbol("true")),
                ("yes", Symbol("true")),
                ("0", Symbol("false")),
                ("off", Symbol("false")),
                ("no", Symbol("false")),
                ("none", Symbol("random")),
            ]
                config = Dict(:enabled_default => input)
                attr = NeatEvolution.BoolAttribute(:enabled, config)
                @test attr.default == expected
            end
        end

        @testset "Construction with unknown default errors" begin
            config = Dict(:enabled_default => "maybe")
            @test_throws ErrorException NeatEvolution.BoolAttribute(:enabled, config)
        end

        @testset "init_value - true default" begin
            attr = NeatEvolution.BoolAttribute(:enabled, Symbol("true"), 0.0, 0.0, 0.0)
            @test NeatEvolution.init_value(attr) == true
        end

        @testset "init_value - false default" begin
            attr = NeatEvolution.BoolAttribute(:enabled, Symbol("false"), 0.0, 0.0, 0.0)
            @test NeatEvolution.init_value(attr) == false
        end

        @testset "init_value - random default" begin
            attr = NeatEvolution.BoolAttribute(:enabled, Symbol("random"), 0.0, 0.0, 0.0)
            rng = MersenneTwister(42)
            values = [NeatEvolution.init_value(attr, rng) for _ in 1:1000]

            true_count = count(values)
            rate = true_count / 1000
            @test 0.4 < rate < 0.6  # Should be roughly 50/50
            println("  Random init: $(round(rate * 100, digits=1))% true (expected ~50%)")
        end

        @testset "mutate_value - basic mutation" begin
            # High mutation rate
            attr = NeatEvolution.BoolAttribute(:enabled, Symbol("true"), 1.0, 0.0, 0.0)
            rng = MersenneTwister(42)

            # With 100% mutate rate, result is random bool
            values = [NeatEvolution.mutate_value(attr, true, rng) for _ in 1:1000]
            false_count = count(v -> !v, values)
            rate = false_count / 1000
            @test 0.4 < rate < 0.6  # Random bool, so ~50% false
        end

        @testset "mutate_value - directional bias" begin
            # Test rate_to_true_add: when value is false, extra push to flip
            attr = NeatEvolution.BoolAttribute(:enabled, Symbol("true"), 0.0, 0.5, 0.0)
            rng = MersenneTwister(42)

            # Starting from false, rate_to_true_add adds 0.5 to mutate_rate
            flip_count = count(_ -> NeatEvolution.mutate_value(attr, false, rng) != false, 1:1000)
            # Effective rate is 0 + 0.5 = 0.5, then random bool
            println("  rate_to_true_add: flipped $(flip_count)/1000 from false")
            @test flip_count > 100  # Should flip at least some

            # Starting from true, rate_to_true_add doesn't apply
            rng2 = MersenneTwister(42)
            flip_count2 = count(_ -> NeatEvolution.mutate_value(attr, true, rng2) != true, 1:1000)
            # Effective rate is 0 + 0 = 0, so no flips
            @test flip_count2 == 0
        end

        @testset "mutate_value - no mutation when rate is zero" begin
            attr = NeatEvolution.BoolAttribute(:enabled, Symbol("true"), 0.0, 0.0, 0.0)
            rng = MersenneTwister(42)

            values = [NeatEvolution.mutate_value(attr, true, rng) for _ in 1:100]
            @test all(v -> v == true, values)

            values2 = [NeatEvolution.mutate_value(attr, false, rng) for _ in 1:100]
            @test all(v -> v == false, values2)
        end
    end

    @testset "StringAttribute" begin
        @testset "Construction from config dict" begin
            config = Dict(
                :activation_default => "sigmoid",
                :activation_options => ["sigmoid", "tanh", "relu"],
                :activation_mutate_rate => 0.1
            )
            attr = NeatEvolution.StringAttribute(:activation, config)

            @test attr.name == :activation
            @test attr.default == :sigmoid
            @test attr.options == [:sigmoid, :tanh, :relu]
            @test attr.mutate_rate == 0.1
        end

        @testset "Construction with random default" begin
            config = Dict(
                :activation_default => "random",
                :activation_options => ["sigmoid", "tanh"]
            )
            attr = NeatEvolution.StringAttribute(:activation, config)
            @test attr.default === nothing
        end

        @testset "Construction with none default" begin
            config = Dict(
                :activation_default => "none",
                :activation_options => ["sigmoid"]
            )
            attr = NeatEvolution.StringAttribute(:activation, config)
            @test attr.default === nothing
        end

        @testset "Construction with space-separated string options" begin
            config = Dict(
                :activation_default => "sigmoid",
                :activation_options => "sigmoid tanh relu"
            )
            attr = NeatEvolution.StringAttribute(:activation, config)
            @test attr.options == [:sigmoid, :tanh, :relu]
        end

        @testset "init_value - with default" begin
            attr = NeatEvolution.StringAttribute(:act, :sigmoid, [:sigmoid, :tanh, :relu], 0.0)
            @test NeatEvolution.init_value(attr) == :sigmoid
        end

        @testset "init_value - random (no default)" begin
            attr = NeatEvolution.StringAttribute(:act, nothing, [:sigmoid, :tanh, :relu], 0.0)
            rng = MersenneTwister(42)
            values = Set(NeatEvolution.init_value(attr, rng) for _ in 1:100)

            # Should pick from options
            @test values ⊆ Set([:sigmoid, :tanh, :relu])
            # With 100 samples, should hit all three
            @test length(values) == 3
        end

        @testset "mutate_value - with mutation" begin
            attr = NeatEvolution.StringAttribute(:act, :sigmoid, [:sigmoid, :tanh, :relu], 1.0)
            rng = MersenneTwister(42)

            values = Set(NeatEvolution.mutate_value(attr, :sigmoid, rng) for _ in 1:100)
            # With 100% mutate rate, should see other options too
            @test length(values) > 1
        end

        @testset "mutate_value - no mutation" begin
            attr = NeatEvolution.StringAttribute(:act, :sigmoid, [:sigmoid, :tanh, :relu], 0.0)
            rng = MersenneTwister(42)

            values = [NeatEvolution.mutate_value(attr, :sigmoid, rng) for _ in 1:100]
            @test all(v -> v == :sigmoid, values)
        end

        @testset "validate" begin
            # Valid: default is in options
            attr = NeatEvolution.StringAttribute(:act, :sigmoid, [:sigmoid, :tanh], 0.0)
            NeatEvolution.validate(attr)  # Should not throw

            # Invalid: default not in options
            bad_attr = NeatEvolution.StringAttribute(:act, :relu, [:sigmoid, :tanh], 0.0)
            @test_throws ErrorException NeatEvolution.validate(bad_attr)

            # Valid: no default (nothing)
            attr2 = NeatEvolution.StringAttribute(:act, nothing, [:sigmoid, :tanh], 0.0)
            NeatEvolution.validate(attr2)  # Should not throw
        end
    end
end

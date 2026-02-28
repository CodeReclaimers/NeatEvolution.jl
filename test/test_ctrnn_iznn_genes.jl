"""
Tests for CTRNN and Izhikevich NodeGene extensions.

Verifies that the new per-node fields (time_constant, iz_a/b/c/d) work
correctly through init, mutate, crossover, distance, and copy operations.
"""

using Test
using NeatEvolution
using Random

"""
Create a GenomeConfig with CTRNN time_constant configured.
"""
function make_ctrnn_config(; num_inputs=1, num_outputs=1)
    params = Dict{Symbol, Any}(
        :num_inputs => num_inputs,
        :num_outputs => num_outputs,
        :feed_forward => false,
        :initial_connection => "unconnected",
        :activation_default => "identity",
        :activation_options => ["identity"],
        :activation_mutate_rate => 0.0,
        :aggregation_default => "sum",
        :aggregation_options => ["sum"],
        :aggregation_mutate_rate => 0.0,
        # CTRNN time_constant
        :time_constant_init_mean => 0.05,
        :time_constant_init_stdev => 0.01,
        :time_constant_min_value => 0.001,
        :time_constant_max_value => 1.0,
        :time_constant_mutate_rate => 0.5,
        :time_constant_mutate_power => 0.01,
        :time_constant_replace_rate => 0.1,
    )
    return GenomeConfig(params)
end

"""
Create a GenomeConfig with Izhikevich parameters configured.
"""
function make_iznn_config(; num_inputs=1, num_outputs=1)
    params = Dict{Symbol, Any}(
        :num_inputs => num_inputs,
        :num_outputs => num_outputs,
        :feed_forward => false,
        :initial_connection => "unconnected",
        :activation_default => "identity",
        :activation_options => ["identity"],
        :activation_mutate_rate => 0.0,
        :aggregation_default => "sum",
        :aggregation_options => ["sum"],
        :aggregation_mutate_rate => 0.0,
        # Izhikevich parameters (regular spiking defaults)
        :iz_a_init_mean => 0.02, :iz_a_init_stdev => 0.001,
        :iz_a_min_value => 0.001, :iz_a_max_value => 0.2,
        :iz_a_mutate_rate => 0.5, :iz_a_mutate_power => 0.005,
        :iz_a_replace_rate => 0.1,
        :iz_b_init_mean => 0.20, :iz_b_init_stdev => 0.01,
        :iz_b_min_value => 0.01, :iz_b_max_value => 0.3,
        :iz_b_mutate_rate => 0.5, :iz_b_mutate_power => 0.01,
        :iz_b_replace_rate => 0.1,
        :iz_c_init_mean => -65.0, :iz_c_init_stdev => 5.0,
        :iz_c_min_value => -80.0, :iz_c_max_value => -40.0,
        :iz_c_mutate_rate => 0.5, :iz_c_mutate_power => 1.0,
        :iz_c_replace_rate => 0.1,
        :iz_d_init_mean => 8.0, :iz_d_init_stdev => 1.0,
        :iz_d_min_value => 0.05, :iz_d_max_value => 10.0,
        :iz_d_mutate_rate => 0.5, :iz_d_mutate_power => 0.5,
        :iz_d_replace_rate => 0.1,
    )
    return GenomeConfig(params)
end

@testset "CTRNN/IZNN Gene Extensions" begin

    @testset "NaN defaults for standard config" begin
        config_path = joinpath(dirname(@__DIR__), "examples", "xor", "config.toml")
        config = load_config(config_path)

        @test config.genome_config.time_constant_attr === nothing
        @test config.genome_config.iz_a_attr === nothing

        # Nodes created with standard config should have NaN for new fields
        node = NodeGene(1)
        NeatEvolution.init_attributes!(node, config.genome_config, MersenneTwister(42))
        @test isnan(node.time_constant)
        @test isnan(node.iz_a)
        @test isnan(node.iz_b)
        @test isnan(node.iz_c)
        @test isnan(node.iz_d)
        println("  Standard config: all CTRNN/IZNN fields remain NaN")
    end

    @testset "CTRNN time_constant init and mutate" begin
        config = make_ctrnn_config()
        @test config.time_constant_attr !== nothing
        @test config.iz_a_attr === nothing

        rng = MersenneTwister(42)
        node = NodeGene(1)
        NeatEvolution.init_attributes!(node, config, rng)

        @test !isnan(node.time_constant)
        @test node.time_constant >= 0.001  # min_value
        @test node.time_constant <= 1.0    # max_value
        @test isnan(node.iz_a)  # Not configured
        println("  CTRNN init: time_constant=$(node.time_constant)")

        # Determinism
        rng2 = MersenneTwister(42)
        node2 = NodeGene(1)
        NeatEvolution.init_attributes!(node2, config, rng2)
        @test node2.time_constant == node.time_constant

        # Mutation should eventually change the value
        original_tc = node.time_constant
        changed = false
        for _ in 1:50
            mutate!(node, config, rng)
            if node.time_constant != original_tc
                changed = true
                break
            end
        end
        @test changed
        @test node.time_constant >= 0.001
        @test node.time_constant <= 1.0
        println("  CTRNN mutate: time_constant changed to $(node.time_constant)")
    end

    @testset "IZNN parameter init and mutate" begin
        config = make_iznn_config()
        @test config.iz_a_attr !== nothing
        @test config.iz_b_attr !== nothing
        @test config.iz_c_attr !== nothing
        @test config.iz_d_attr !== nothing

        rng = MersenneTwister(42)
        node = NodeGene(1)
        NeatEvolution.init_attributes!(node, config, rng)

        @test !isnan(node.iz_a)
        @test !isnan(node.iz_b)
        @test !isnan(node.iz_c)
        @test !isnan(node.iz_d)
        println("  IZNN init: a=$(node.iz_a), b=$(node.iz_b), c=$(node.iz_c), d=$(node.iz_d)")

        # Bounds
        @test 0.001 <= node.iz_a <= 0.2
        @test 0.01 <= node.iz_b <= 0.3
        @test -80.0 <= node.iz_c <= -40.0
        @test 0.05 <= node.iz_d <= 10.0

        # Mutation
        original_a = node.iz_a
        changed = false
        for _ in 1:50
            mutate!(node, config, rng)
            if node.iz_a != original_a
                changed = true
                break
            end
        end
        @test changed
        println("  IZNN mutate: iz_a changed to $(node.iz_a)")
    end

    @testset "Copy preserves CTRNN/IZNN fields" begin
        node = NodeGene(5)
        node.time_constant = 0.05
        node.iz_a = 0.02
        node.iz_b = 0.20
        node.iz_c = -65.0
        node.iz_d = 8.0

        c = copy(node)
        @test c.time_constant == 0.05
        @test c.iz_a == 0.02
        @test c.iz_b == 0.20
        @test c.iz_c == -65.0
        @test c.iz_d == 8.0

        # Verify independence
        c.time_constant = 0.1
        @test node.time_constant == 0.05
        println("  Copy: all 5 new fields preserved and independent")
    end

    @testset "Crossover with CTRNN fields" begin
        rng = MersenneTwister(123)

        n1 = NodeGene(10)
        n1.bias = 1.0; n1.response = 1.0
        n1.activation = :identity; n1.aggregation = :sum
        n1.time_constant = 0.01

        n2 = NodeGene(10)
        n2.bias = 2.0; n2.response = 2.0
        n2.activation = :identity; n2.aggregation = :sum
        n2.time_constant = 1.0

        # Over many trials, offspring should inherit from both parents
        from_1 = 0
        trials = 1000
        for _ in 1:trials
            child = NeatEvolution.crossover(n1, n2, rng)
            @test child.time_constant == 0.01 || child.time_constant == 1.0
            from_1 += (child.time_constant == 0.01)
        end
        @test 0.4 < from_1 / trials < 0.6
        println("  Crossover CTRNN: time_constant from parent1 $(from_1/trials*100)% of trials")
    end

    @testset "Crossover with NaN fields passes through NaN" begin
        rng = MersenneTwister(456)

        n1 = NodeGene(10)
        n1.bias = 1.0; n1.response = 1.0
        n1.activation = :identity; n1.aggregation = :sum
        # time_constant and iz_* left as NaN

        n2 = NodeGene(10)
        n2.bias = 2.0; n2.response = 2.0
        n2.activation = :identity; n2.aggregation = :sum

        child = NeatEvolution.crossover(n1, n2, rng)
        @test isnan(child.time_constant)
        @test isnan(child.iz_a)
        println("  Crossover NaN: NaN fields propagate correctly")
    end

    @testset "Distance with CTRNN fields" begin
        config_path = joinpath(dirname(@__DIR__), "examples", "xor", "config.toml")
        config = load_config(config_path)
        wc = config.genome_config.compatibility_weight_coefficient

        # Both NaN: no contribution
        n1 = NodeGene(1)
        n1.bias = 0.0; n1.response = 1.0
        n1.activation = :sigmoid; n1.aggregation = :sum

        n2 = copy(n1)
        d_baseline = NeatEvolution.distance(n1, n2, config.genome_config)
        @test d_baseline == 0.0

        # Both have time_constant: contributes to distance
        n1.time_constant = 0.01
        n2.time_constant = 0.05
        d_with_tc = NeatEvolution.distance(n1, n2, config.genome_config)
        expected_extra = abs(0.01 - 0.05) * wc
        @test d_with_tc ≈ expected_extra atol=1e-10
        println("  Distance CTRNN: Δtc contribution = $(d_with_tc)")

        # One NaN, one not: no contribution (asymmetric)
        n2.time_constant = NaN
        d_one_nan = NeatEvolution.distance(n1, n2, config.genome_config)
        @test d_one_nan == 0.0  # time_constant doesn't contribute
        println("  Distance CTRNN: one NaN → no contribution")
    end

    @testset "Distance with IZNN fields" begin
        config_path = joinpath(dirname(@__DIR__), "examples", "xor", "config.toml")
        config = load_config(config_path)
        wc = config.genome_config.compatibility_weight_coefficient

        n1 = NodeGene(1)
        n1.bias = 0.0; n1.response = 1.0
        n1.activation = :sigmoid; n1.aggregation = :sum
        n1.iz_a = 0.02; n1.iz_b = 0.20; n1.iz_c = -65.0; n1.iz_d = 8.0

        n2 = copy(n1)
        n2.iz_a = 0.10; n2.iz_b = 0.25; n2.iz_c = -50.0; n2.iz_d = 2.0

        d = NeatEvolution.distance(n1, n2, config.genome_config)
        expected = (abs(0.02 - 0.10) + abs(0.20 - 0.25) + abs(-65.0 - (-50.0)) + abs(8.0 - 2.0)) * wc
        @test d ≈ expected atol=1e-10
        println("  Distance IZNN: $(d) (expected $(expected))")
    end
end

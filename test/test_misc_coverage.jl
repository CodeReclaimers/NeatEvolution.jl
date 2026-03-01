using Test
using NeatEvolution
using Random
using Statistics

@testset "Miscellaneous Coverage Tests" begin

    @testset "Missing Activation Functions" begin
        @testset "elu_activation" begin
            @test elu_activation(1.0) == 1.0
            @test elu_activation(0.0) == 0.0
            @test elu_activation(-1.0) ≈ exp(-1.0) - 1.0
            @test elu_activation(5.0) == 5.0
            # ELU is continuous at 0
            @test elu_activation(1e-10) ≈ 1e-10 atol=1e-9
            @test elu_activation(-1e-10) ≈ exp(-1e-10) - 1.0 atol=1e-9
        end

        @testset "lelu_activation" begin
            @test lelu_activation(1.0) == 1.0
            @test lelu_activation(0.0) == 0.0
            @test lelu_activation(-1.0) == 0.005 * -1.0
            @test lelu_activation(-100.0) == 0.005 * -100.0
            @test lelu_activation(5.0) == 5.0
            # Leaky: negative values are small but non-zero
            @test lelu_activation(-2.0) < 0.0
            @test lelu_activation(-2.0) > -1.0
        end

        @testset "selu_activation" begin
            lam = 1.0507009873554804934193349852946
            alpha = 1.6732632423543772848170429916717

            @test selu_activation(1.0) ≈ lam * 1.0
            @test selu_activation(0.0) == 0.0
            @test selu_activation(-1.0) ≈ lam * alpha * (exp(-1.0) - 1.0)
            # SELU preserves self-normalizing property
            @test selu_activation(2.0) ≈ lam * 2.0
        end
    end

    @testset "Custom Function Registration" begin
        @testset "add_activation_function!" begin
            # Register a custom activation
            custom_fn(z::Float64) = z * z + 1.0
            add_activation_function!(:test_custom_act, custom_fn)

            retrieved = get_activation_function(:test_custom_act)
            @test retrieved(2.0) == 5.0
            @test retrieved(0.0) == 1.0

            # Clean up
            delete!(NeatEvolution.ACTIVATION_FUNCTIONS, :test_custom_act)
        end

        @testset "add_aggregation_function!" begin
            # Register a custom aggregation
            custom_agg(x::Vector{Float64}) = sum(x .^ 2)
            add_aggregation_function!(:test_custom_agg, custom_agg)

            retrieved = get_aggregation_function(:test_custom_agg)
            @test retrieved([1.0, 2.0, 3.0]) == 14.0

            # Clean up
            delete!(NeatEvolution.AGGREGATION_FUNCTIONS, :test_custom_agg)
        end

        @testset "get_activation_function - unknown errors" begin
            @test_throws ErrorException get_activation_function(:nonexistent_function)
        end

        @testset "get_aggregation_function - unknown errors" begin
            @test_throws ErrorException get_aggregation_function(:nonexistent_function)
        end
    end

    @testset "Utility Functions" begin
        @testset "stat function wrappers" begin
            vals = [1.0, 2.0, 3.0, 4.0, 5.0]

            @test NeatEvolution.mean_stat(vals) ≈ 3.0
            @test NeatEvolution.median_stat(vals) ≈ 3.0
            @test NeatEvolution.stdev_stat(vals) ≈ std(vals)
            @test NeatEvolution.variance_stat(vals) ≈ var(vals)
        end

        @testset "tmean_stat wrapper" begin
            vals = [1.0, 2.0, 3.0, 4.0, 5.0]
            @test NeatEvolution.tmean_stat(vals) isa Float64
        end

        @testset "get_stat_function" begin
            for name in [:min, :max, :mean, :median, :stdev, :variance, :tmean]
                fn = NeatEvolution.get_stat_function(name)
                @test fn isa Function
            end

            @test_throws ErrorException NeatEvolution.get_stat_function(:nonexistent)
        end

        @testset "get_stat_function correctness" begin
            vals = [1.0, 5.0, 3.0, 2.0, 4.0]
            @test NeatEvolution.get_stat_function(:min)(vals) == 1.0
            @test NeatEvolution.get_stat_function(:max)(vals) == 5.0
            @test NeatEvolution.get_stat_function(:mean)(vals) == 3.0
            @test NeatEvolution.get_stat_function(:median)(vals) == 3.0
        end

        @testset "tmean edge cases" begin
            # Single element
            @test tmean([5.0]) == 5.0

            # Two elements with trim
            @test tmean([1.0, 2.0]; trim=0.1) ≈ 1.5

            # High trim that trims everything
            @test tmean([1.0, 2.0, 3.0, 4.0, 5.0]; trim=0.5) isa Float64

            # Empty collection
            @test_throws ErrorException tmean(Float64[])
        end

        @testset "softmax" begin
            # Basic functionality
            result = softmax([1.0, 2.0, 3.0])
            @test sum(result) ≈ 1.0 atol=1e-10

            # All equal inputs -> uniform distribution
            result2 = softmax([1.0, 1.0, 1.0])
            @test all(r -> r ≈ 1/3, result2)

            # Larger input gets higher probability
            result3 = softmax([0.0, 10.0])
            @test result3[2] > result3[1]
            @test result3[2] > 0.99

            # Numerical stability: large inputs shouldn't overflow
            result4 = softmax([1000.0, 1001.0])
            @test all(isfinite, result4)
            @test sum(result4) ≈ 1.0 atol=1e-10
        end
    end

    @testset "Aggregation - empty input handling" begin
        @test sum_aggregation(Float64[]) == 0.0
        @test product_aggregation(Float64[]) == 1.0
        @test max_aggregation(Float64[]) == 0.0
        @test min_aggregation(Float64[]) == 0.0
        @test maxabs_aggregation(Float64[]) == 0.0
        @test median_aggregation(Float64[]) == 0.0
        @test mean_aggregation(Float64[]) == 0.0
    end

    @testset "AbstractNetwork interface" begin
        config_path = joinpath(dirname(@__DIR__), "examples", "xor", "config.toml")
        config = load_config(config_path)

        g = Genome(1)
        configure_new!(g, config.genome_config)

        net = FeedForwardNetwork(g, config.genome_config)

        @test net isa AbstractNetwork
        @test input_nodes(net) == config.genome_config.input_keys
        @test output_nodes(net) == config.genome_config.output_keys
        @test num_inputs(net) == config.genome_config.num_inputs
        @test num_outputs(net) == config.genome_config.num_outputs
    end

    @testset "Species utility functions" begin
        @testset "get_fitnesses" begin
            s = NeatEvolution.Species(1, 1)

            # Empty species
            @test isempty(NeatEvolution.get_fitnesses(s))

            # Add members with fitness
            g1 = Genome(1)
            g1.fitness = 3.0
            g2 = Genome(2)
            g2.fitness = 5.0
            g3 = Genome(3)
            g3.fitness = nothing  # No fitness yet

            s.members = Dict(1 => g1, 2 => g2, 3 => g3)

            fitnesses = NeatEvolution.get_fitnesses(s)
            @test length(fitnesses) == 2  # g3 excluded
            @test 3.0 in fitnesses
            @test 5.0 in fitnesses
        end

        @testset "get_species_id" begin
            species_config = NeatEvolution.SpeciesConfig(Dict(:compatibility_threshold => 3.0))
            species_set = NeatEvolution.SpeciesSet(species_config)
            species_set.genome_to_species[42] = 1

            @test NeatEvolution.get_species_id(species_set, 42) == 1
            @test_throws KeyError NeatEvolution.get_species_id(species_set, 999)
        end
    end

    @testset "Config construction" begin
        @testset "ReproductionConfig defaults" begin
            rc = NeatEvolution.ReproductionConfig(Dict{Symbol,Any}())
            @test rc.elitism == 0
            @test rc.survival_threshold == 0.2
            @test rc.min_species_size == 1
        end

        @testset "StagnationConfig defaults" begin
            sc = NeatEvolution.StagnationConfig(Dict{Symbol,Any}())
            @test sc.species_fitness_func == :mean
            @test sc.max_stagnation == 15
            @test sc.species_elitism == 0
        end

        @testset "SpeciesConfig defaults" begin
            sc = NeatEvolution.SpeciesConfig(Dict{Symbol,Any}())
            @test sc.compatibility_threshold == 3.0
        end

        @testset "Config from components" begin
            gc = GenomeConfig(Dict(:num_inputs => 3, :num_outputs => 2,
                                   :activation_default => "sigmoid",
                                   :aggregation_default => "sum",
                                   :activation_options => ["sigmoid"],
                                   :aggregation_options => ["sum"]))
            sc = NeatEvolution.SpeciesConfig(Dict{Symbol,Any}())
            stag = NeatEvolution.StagnationConfig(Dict{Symbol,Any}())
            rc = NeatEvolution.ReproductionConfig(Dict{Symbol,Any}())

            config = Config(
                Dict(:pop_size => 200, :fitness_criterion => :min, :fitness_threshold => 0.1),
                gc, sc, stag, rc
            )

            @test config.pop_size == 200
            @test config.fitness_criterion == :min
            @test config.fitness_threshold == 0.1
            @test config.genome_config.num_inputs == 3
            @test config.genome_config.num_outputs == 2
        end
    end

    @testset "GenomeConfig input/output key conventions" begin
        gc = GenomeConfig(Dict(
            :num_inputs => 3,
            :num_outputs => 2,
            :activation_default => "sigmoid",
            :aggregation_default => "sum",
            :activation_options => ["sigmoid"],
            :aggregation_options => ["sum"]
        ))

        # Input keys are negative: -3, -2, -1
        @test gc.input_keys == [-3, -2, -1]
        # Output keys are 0-indexed: 0, 1
        @test gc.output_keys == [0, 1]
    end
end

using Test
using NEAT

@testset "NEAT.jl" begin
    @testset "Config Loading" begin
        # Test that we can load a config file
        config_path = joinpath(dirname(@__DIR__), "examples", "xor", "config.toml")
        config = load_config(config_path)

        @test config.pop_size == 150
        @test config.fitness_criterion == :max
        @test config.genome_config.num_inputs == 2
        @test config.genome_config.num_outputs == 1
    end

    @testset "Activation Functions" begin
        @test sigmoid_activation(0.0) ≈ 0.5
        @test tanh_activation(0.0) ≈ 0.0
        @test relu_activation(-1.0) == 0.0
        @test relu_activation(1.0) == 1.0
    end

    @testset "Aggregation Functions" begin
        @test sum_aggregation([1.0, 2.0, 3.0]) == 6.0
        @test product_aggregation([2.0, 3.0]) == 6.0
        @test max_aggregation([1.0, 5.0, 3.0]) == 5.0
        @test min_aggregation([1.0, 5.0, 3.0]) == 1.0
    end

    @testset "Genome Creation" begin
        config_path = joinpath(dirname(@__DIR__), "examples", "xor", "config.toml")
        config = load_config(config_path)

        genome = Genome(1)
        configure_new!(genome, config.genome_config)

        @test !isempty(genome.nodes)
        @test haskey(genome.nodes, 0)  # Output node
    end

    @testset "XOR Evolution (short run)" begin
        config_path = joinpath(dirname(@__DIR__), "examples", "xor", "config.toml")
        config = load_config(config_path)

        # XOR test data
        xor_inputs = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ]

        xor_outputs = [
            [0.0],
            [1.0],
            [1.0],
            [0.0]
        ]

        # Fitness function
        function eval_genomes(genomes, cfg)
            for (genome_id, genome) in genomes
                genome.fitness = 4.0
                net = FeedForwardNetwork(genome, cfg.genome_config)

                for (xi, xo) in zip(xor_inputs, xor_outputs)
                    output = activate!(net, xi)
                    genome.fitness -= (output[1] - xo[1])^2
                end
            end
        end

        # Create population
        pop = Population(config)

        # Run for just 5 generations to verify it works
        winner = run!(pop, eval_genomes, 5)

        @test winner !== nothing
        @test winner.fitness !== nothing
    end
end

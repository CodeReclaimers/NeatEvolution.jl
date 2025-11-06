using Test
using NEAT
using Random

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
        # Basic tests
        @test sigmoid_activation(0.0) ≈ 0.5
        @test tanh_activation(0.0) ≈ 0.0
        @test sin_activation(0.0) ≈ 0.0
        @test gauss_activation(0.0) ≈ 1.0 atol=1e-6
        @test gauss_activation(-1.0) ≈ gauss_activation(1.0) atol=1e-6

        # ReLU tests
        @test relu_activation(-1.0) == 0.0
        @test relu_activation(0.0) == 0.0
        @test relu_activation(1.0) == 1.0

        # Softplus tests
        @test softplus_activation(-5.0) ≈ 0.0 atol=1e-3
        @test 0.0 < softplus_activation(0.0) < 0.25
        @test softplus_activation(5.0) ≈ 5.0 atol=1e-2

        # Identity tests
        @test identity_activation(-1.0) == -1.0
        @test identity_activation(0.0) == 0.0
        @test identity_activation(1.0) == 1.0

        # Clamped tests
        @test clamped_activation(-2.0) == -1.0
        @test clamped_activation(-1.0) == -1.0
        @test clamped_activation(0.0) == 0.0
        @test clamped_activation(1.0) == 1.0
        @test clamped_activation(2.0) == 1.0

        # Inverse tests
        @test inv_activation(1.0) == 1.0
        @test inv_activation(0.5) == 2.0
        @test inv_activation(2.0) == 0.5
        @test inv_activation(0.0) == 0.0

        # Log and Exp tests
        @test log_activation(1.0) == 0.0
        @test exp_activation(0.0) == 1.0

        # Abs tests
        @test abs_activation(-1.0) == 1.0
        @test abs_activation(0.0) == 0.0
        @test abs_activation(1.0) == 1.0

        # Hat tests
        @test hat_activation(-1.0) == 0.0
        @test hat_activation(0.0) == 1.0
        @test hat_activation(1.0) == 0.0

        # Square tests
        @test square_activation(-1.0) == 1.0
        @test square_activation(-0.5) == 0.25
        @test square_activation(0.0) == 0.0
        @test square_activation(0.5) == 0.25
        @test square_activation(1.0) == 1.0

        # Cube tests
        @test cube_activation(-1.0) == -1.0
        @test cube_activation(-0.5) == -0.125
        @test cube_activation(0.0) == 0.0
        @test cube_activation(0.5) == 0.125
        @test cube_activation(1.0) == 1.0

        # Test getting activation functions
        @test get_activation_function(:sigmoid) !== nothing
        @test get_activation_function(:tanh) !== nothing
        @test get_activation_function(:relu) !== nothing
    end

    @testset "Aggregation Functions" begin
        # Sum tests
        @test sum_aggregation([1.0, 2.0, 0.5]) == 3.5
        @test sum_aggregation([1.0, -1.0, 0.0]) == 0.0

        # Product tests
        @test product_aggregation([1.0, 2.0, 0.5]) == 1.0
        @test product_aggregation([1.0, 0.5, 0.0]) == 0.0

        # Max tests
        @test max_aggregation([0.0, 1.0, 2.0]) == 2.0
        @test max_aggregation([0.0, -1.0, -2.0]) == 0.0

        # Min tests
        @test min_aggregation([0.0, 1.0, 2.0]) == 0.0
        @test min_aggregation([0.0, -1.0, -2.0]) == -2.0

        # Maxabs tests
        @test maxabs_aggregation([0.0, 1.0, 2.0]) == 2.0
        @test maxabs_aggregation([0.0, -1.0, -2.0]) == -2.0

        # Median tests
        @test median_aggregation([0.0, 1.0, 2.0]) == 1.0
        @test median_aggregation([-10.0, 1.0, 3.0, 10.0]) == 2.0

        # Mean tests
        @test mean_aggregation([0.0, 1.0, 2.0]) == 1.0
        @test mean_aggregation([0.0, -1.0, -2.0]) == -1.0

        # Test getting aggregation functions
        @test get_aggregation_function(:sum) !== nothing
        @test get_aggregation_function(:product) !== nothing
        @test get_aggregation_function(:max) !== nothing
    end

    @testset "Genome Creation" begin
        config_path = joinpath(dirname(@__DIR__), "examples", "xor", "config.toml")
        config = load_config(config_path)

        genome = Genome(1)
        configure_new!(genome, config.genome_config)

        @test !isempty(genome.nodes)
        @test haskey(genome.nodes, 0)  # Output node
    end

    @testset "Graph Algorithms" begin
        # Import graph functions
        using NEAT: creates_cycle, required_for_output, feed_forward_layers

        @testset "creates_cycle" begin
            # Self-connection creates cycle
            @test creates_cycle([(0, 1), (1, 2), (2, 3)], (0, 0))

            # Test various cycle scenarios
            @test creates_cycle([(0, 1), (1, 2), (2, 3)], (1, 0))
            @test !creates_cycle([(0, 1), (1, 2), (2, 3)], (0, 1))

            @test creates_cycle([(0, 1), (1, 2), (2, 3)], (2, 0))
            @test !creates_cycle([(0, 1), (1, 2), (2, 3)], (0, 2))

            @test creates_cycle([(0, 1), (1, 2), (2, 3)], (3, 0))
            @test !creates_cycle([(0, 1), (1, 2), (2, 3)], (0, 3))

            # More complex network
            @test creates_cycle([(0, 2), (1, 3), (2, 3), (4, 2)], (3, 4))
            @test !creates_cycle([(0, 2), (1, 3), (2, 3), (4, 2)], (4, 3))
        end

        @testset "required_for_output" begin
            # Simple direct connection
            inputs = [0, 1]
            outputs = [2]
            connections = [(0, 2), (1, 2)]
            required = required_for_output(inputs, outputs, connections)
            @test required == Set([2])

            # With hidden nodes
            inputs = [0, 1]
            outputs = [2]
            connections = [(0, 3), (1, 4), (3, 2), (4, 2)]
            required = required_for_output(inputs, outputs, connections)
            @test required == Set([2, 3, 4])

            # Single hidden layer
            inputs = [0, 1]
            outputs = [3]
            connections = [(0, 2), (1, 2), (2, 3)]
            required = required_for_output(inputs, outputs, connections)
            @test required == Set([2, 3])

            # Multiple paths
            inputs = [0, 1]
            outputs = [4]
            connections = [(0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
            required = required_for_output(inputs, outputs, connections)
            @test required == Set([2, 3, 4])

            # With recurrent connection
            inputs = [0, 1]
            outputs = [4]
            connections = [(0, 2), (1, 3), (2, 3), (3, 4), (4, 2)]
            required = required_for_output(inputs, outputs, connections)
            @test required == Set([2, 3, 4])

            # With unused node
            inputs = [0, 1]
            outputs = [4]
            connections = [(0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (2, 5)]
            required = required_for_output(inputs, outputs, connections)
            @test required == Set([2, 3, 4])
        end

        @testset "feed_forward_layers" begin
            # Simple direct connection
            inputs = [0, 1]
            outputs = [2]
            connections = [(0, 2), (1, 2)]
            layers = feed_forward_layers(inputs, outputs, connections)
            @test layers == [Set([2])]

            # Single hidden layer
            inputs = [0, 1]
            outputs = [3]
            connections = [(0, 2), (1, 2), (2, 3)]
            layers = feed_forward_layers(inputs, outputs, connections)
            @test layers == [Set([2]), Set([3])]

            # Multiple layers
            inputs = [0, 1]
            outputs = [4]
            connections = [(0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
            layers = feed_forward_layers(inputs, outputs, connections)
            @test layers == [Set([2]), Set([3]), Set([4])]

            # Complex network
            inputs = [0, 1, 2, 3]
            outputs = [11, 12, 13]
            connections = [(0, 4), (1, 4), (1, 5), (2, 5), (2, 6), (3, 6), (3, 7),
                          (4, 8), (5, 8), (5, 9), (5, 10), (6, 10), (6, 7),
                          (8, 11), (8, 12), (8, 9), (9, 10), (7, 10),
                          (10, 12), (10, 13)]
            layers = feed_forward_layers(inputs, outputs, connections)
            @test layers == [Set([4, 5, 6]), Set([8, 7]), Set([9, 11]), Set([10]), Set([12, 13])]

            # With unused nodes
            inputs = [0, 1, 2, 3]
            outputs = [11, 12, 13]
            connections = [(0, 4), (1, 4), (1, 5), (2, 5), (2, 6), (3, 6), (3, 7),
                          (4, 8), (5, 8), (5, 9), (5, 10), (6, 10), (6, 7),
                          (8, 11), (8, 12), (8, 9), (9, 10), (7, 10),
                          (10, 12), (10, 13),
                          (3, 14), (14, 15), (5, 16), (10, 16)]
            layers = feed_forward_layers(inputs, outputs, connections)
            @test layers == [Set([4, 5, 6]), Set([8, 7]), Set([9, 11]), Set([10]), Set([12, 13])]
        end

        @testset "Fuzz test - required_for_output" begin
            # Simplified fuzz test (10 iterations instead of 1000)
            Random.seed!(42)
            for _ in 1:10
                n_hidden = rand(10:30)
                n_in = rand(1:5)
                n_out = rand(1:5)
                nodes = unique([rand(0:100) for _ in 1:(n_in + n_out + n_hidden)])
                shuffle!(nodes)

                inputs = nodes[1:min(n_in, length(nodes))]
                outputs = nodes[min(n_in + 1, length(nodes)):min(n_in + n_out, length(nodes))]

                if isempty(outputs)
                    continue
                end

                connections = Tuple{Int, Int}[]
                for _ in 1:(n_hidden * 2)
                    a = rand(nodes)
                    b = rand(nodes)
                    if a == b || (a in inputs && b in inputs) || (a in outputs && b in outputs)
                        continue
                    end
                    push!(connections, (a, b))
                end

                required = required_for_output(inputs, outputs, connections)
                for o in outputs
                    @test o in required
                end
            end
        end
    end

    @testset "Feed-Forward Network Evaluation" begin
        using NEAT: sigmoid_activation

        @testset "Unconnected network" begin
            # Network with no inputs and one output neuron
            input_nodes = Int[]
            output_nodes = [0]
            node_evals = [(0, sigmoid_activation, sum_aggregation, 0.0, 1.0, Tuple{Int, Float64}[])]
            values = Dict{Int, Float64}(0 => 0.0)

            net = FeedForwardNetwork(input_nodes, output_nodes, node_evals, values)

            @test net.values[0] == 0.0

            result = activate!(net, Float64[])
            @test result[1] ≈ 0.5 atol=0.001
            @test result[1] == net.values[0]

            # Activate again - should be same
            result = activate!(net, Float64[])
            @test result[1] ≈ 0.5 atol=0.001
        end

        @testset "Basic network" begin
            # Simple network with one connection of weight 1.0 to sigmoid output
            input_nodes = [-1]
            output_nodes = [0]
            node_evals = [(0, sigmoid_activation, sum_aggregation, 0.0, 1.0, [(-1, 1.0)])]
            values = Dict{Int, Float64}(-1 => 0.0, 0 => 0.0)

            net = FeedForwardNetwork(input_nodes, output_nodes, node_evals, values)

            @test net.values[0] == 0.0

            # First activation with input 0.2
            result = activate!(net, [0.2])
            @test net.values[-1] == 0.2
            @test result[1] ≈ 0.7311 atol=0.001
            @test result[1] == net.values[0]

            # Second activation with input 0.4
            result = activate!(net, [0.4])
            @test net.values[-1] == 0.4
            @test result[1] ≈ 0.8808 atol=0.001
            @test result[1] == net.values[0]
        end
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

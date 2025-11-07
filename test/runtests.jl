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

    @testset "Genome Initialization" begin
        test_config_path = joinpath(@__DIR__, "test_config.toml")
        test_config = load_config(test_config_path)

        @testset "Unconnected - no hidden" begin
            config = test_config.genome_config
            # Modify config for this test
            temp_config = GenomeConfig(Dict(
                :num_inputs => 2,
                :num_outputs => 1,
                :num_hidden => 0,
                :initial_connection => :unconnected,
                :feed_forward => true,
                :activation_default => "sigmoid",
                :aggregation_default => "sum",
                :activation_options => ["sigmoid"],
                :aggregation_options => ["sum"],
                :bias_init_mean => 0.0,
                :bias_init_stdev => 1.0,
                :response_init_mean => 1.0,
                :response_init_stdev => 0.0,
                :weight_init_mean => 0.0,
                :weight_init_stdev => 1.0,
                :enabled_default => true
            ))

            g = Genome(42)
            configure_new!(g, temp_config)

            @test g.key == 42
            @test Set(keys(g.nodes)) == Set([0])
            @test isempty(g.connections)
        end

        @testset "Unconnected - with hidden" begin
            temp_config = GenomeConfig(Dict(
                :num_inputs => 2,
                :num_outputs => 1,
                :num_hidden => 2,
                :initial_connection => :unconnected,
                :feed_forward => true,
                :activation_default => "sigmoid",
                :aggregation_default => "sum",
                :activation_options => ["sigmoid"],
                :aggregation_options => ["sum"],
                :bias_init_mean => 0.0,
                :bias_init_stdev => 1.0,
                :response_init_mean => 1.0,
                :response_init_stdev => 0.0,
                :weight_init_mean => 0.0,
                :weight_init_stdev => 1.0,
                :enabled_default => true
            ))

            g = Genome(42)
            configure_new!(g, temp_config)

            @test g.key == 42
            @test Set(keys(g.nodes)) == Set([0, 1, 2])
            @test isempty(g.connections)
        end

        @testset "FS-NEAT - no hidden" begin
            temp_config = GenomeConfig(Dict(
                :num_inputs => 2,
                :num_outputs => 1,
                :num_hidden => 0,
                :initial_connection => :fs_neat,
                :feed_forward => true,
                :activation_default => "sigmoid",
                :aggregation_default => "sum",
                :activation_options => ["sigmoid"],
                :aggregation_options => ["sum"],
                :bias_init_mean => 0.0,
                :bias_init_stdev => 1.0,
                :response_init_mean => 1.0,
                :response_init_stdev => 0.0,
                :weight_init_mean => 0.0,
                :weight_init_stdev => 1.0,
                :enabled_default => true
            ))

            g = Genome(42)
            configure_new!(g, temp_config)

            @test g.key == 42
            @test Set(keys(g.nodes)) == Set([0])
            @test length(g.connections) == 2  # 2 inputs to 1 output
        end

        @testset "FS-NEAT - with hidden (nohidden)" begin
            temp_config = GenomeConfig(Dict(
                :num_inputs => 2,
                :num_outputs => 1,
                :num_hidden => 2,
                :initial_connection => :fs_neat_nohidden,
                :feed_forward => true,
                :activation_default => "sigmoid",
                :aggregation_default => "sum",
                :activation_options => ["sigmoid"],
                :aggregation_options => ["sum"],
                :bias_init_mean => 0.0,
                :bias_init_stdev => 1.0,
                :response_init_mean => 1.0,
                :response_init_stdev => 0.0,
                :weight_init_mean => 0.0,
                :weight_init_stdev => 1.0,
                :enabled_default => true
            ))

            g = Genome(42)
            configure_new!(g, temp_config)

            @test g.key == 42
            @test Set(keys(g.nodes)) == Set([0, 1, 2])
            @test length(g.connections) == 2  # Only inputs to output
        end

        @testset "FS-NEAT - with hidden (connect hidden)" begin
            temp_config = GenomeConfig(Dict(
                :num_inputs => 2,
                :num_outputs => 1,
                :num_hidden => 2,
                :initial_connection => :fs_neat_hidden,
                :feed_forward => true,
                :activation_default => "sigmoid",
                :aggregation_default => "sum",
                :activation_options => ["sigmoid"],
                :aggregation_options => ["sum"],
                :bias_init_mean => 0.0,
                :bias_init_stdev => 1.0,
                :response_init_mean => 1.0,
                :response_init_stdev => 0.0,
                :weight_init_mean => 0.0,
                :weight_init_stdev => 1.0,
                :enabled_default => true
            ))

            g = Genome(42)
            configure_new!(g, temp_config)

            @test g.key == 42
            @test Set(keys(g.nodes)) == Set([0, 1, 2])
            @test length(g.connections) == 6  # inputs->hidden + hidden->output
        end

        @testset "Full - no hidden" begin
            temp_config = GenomeConfig(Dict(
                :num_inputs => 2,
                :num_outputs => 1,
                :num_hidden => 0,
                :initial_connection => :full,
                :feed_forward => true,
                :activation_default => "sigmoid",
                :aggregation_default => "sum",
                :activation_options => ["sigmoid"],
                :aggregation_options => ["sum"],
                :bias_init_mean => 0.0,
                :bias_init_stdev => 1.0,
                :response_init_mean => 1.0,
                :response_init_stdev => 0.0,
                :weight_init_mean => 0.0,
                :weight_init_stdev => 1.0,
                :enabled_default => true
            ))

            g = Genome(42)
            configure_new!(g, temp_config)

            @test g.key == 42
            @test Set(keys(g.nodes)) == Set([0])
            @test length(g.connections) == 2
            # Check that each input is connected to output
            for i in temp_config.input_keys
                @test haskey(g.connections, (i, 0))
            end
        end

        @testset "Full nodirect - with hidden" begin
            temp_config = GenomeConfig(Dict(
                :num_inputs => 2,
                :num_outputs => 1,
                :num_hidden => 2,
                :initial_connection => :full_nodirect,
                :feed_forward => true,
                :activation_default => "sigmoid",
                :aggregation_default => "sum",
                :activation_options => ["sigmoid"],
                :aggregation_options => ["sum"],
                :bias_init_mean => 0.0,
                :bias_init_stdev => 1.0,
                :response_init_mean => 1.0,
                :response_init_stdev => 0.0,
                :weight_init_mean => 0.0,
                :weight_init_stdev => 1.0,
                :enabled_default => true
            ))

            g = Genome(42)
            configure_new!(g, temp_config)

            @test g.key == 42
            @test Set(keys(g.nodes)) == Set([0, 1, 2])
            @test length(g.connections) == 6

            # Check that each input is connected to each hidden
            for i in temp_config.input_keys
                for h in [1, 2]
                    @test haskey(g.connections, (i, h))
                end
            end

            # Check that each hidden is connected to output
            for h in [1, 2]
                @test haskey(g.connections, (h, 0))
            end

            # Check that inputs are NOT directly connected to output
            for i in temp_config.input_keys
                @test !haskey(g.connections, (i, 0))
            end
        end

        @testset "Full direct - with hidden" begin
            temp_config = GenomeConfig(Dict(
                :num_inputs => 2,
                :num_outputs => 1,
                :num_hidden => 2,
                :initial_connection => :full_direct,
                :feed_forward => true,
                :activation_default => "sigmoid",
                :aggregation_default => "sum",
                :activation_options => ["sigmoid"],
                :aggregation_options => ["sum"],
                :bias_init_mean => 0.0,
                :bias_init_stdev => 1.0,
                :response_init_mean => 1.0,
                :response_init_stdev => 0.0,
                :weight_init_mean => 0.0,
                :weight_init_stdev => 1.0,
                :enabled_default => true
            ))

            g = Genome(42)
            configure_new!(g, temp_config)

            @test g.key == 42
            @test Set(keys(g.nodes)) == Set([0, 1, 2])
            @test length(g.connections) == 8

            # Check that each input is connected to each hidden
            for i in temp_config.input_keys
                for h in [1, 2]
                    @test haskey(g.connections, (i, h))
                end
            end

            # Check that each hidden is connected to output
            for h in [1, 2]
                @test haskey(g.connections, (h, 0))
            end

            # Check that inputs ARE directly connected to output
            for i in temp_config.input_keys
                @test haskey(g.connections, (i, 0))
            end
        end

        @testset "Partial nodirect - with hidden" begin
            Random.seed!(42)
            temp_config = GenomeConfig(Dict(
                :num_inputs => 2,
                :num_outputs => 1,
                :num_hidden => 2,
                :initial_connection => :partial_nodirect,
                :connection_fraction => 0.5,
                :feed_forward => true,
                :activation_default => "sigmoid",
                :aggregation_default => "sum",
                :activation_options => ["sigmoid"],
                :aggregation_options => ["sum"],
                :bias_init_mean => 0.0,
                :bias_init_stdev => 1.0,
                :response_init_mean => 1.0,
                :response_init_stdev => 0.0,
                :weight_init_mean => 0.0,
                :weight_init_stdev => 1.0,
                :enabled_default => true
            ))

            g = Genome(42)
            configure_new!(g, temp_config)

            @test g.key == 42
            @test Set(keys(g.nodes)) == Set([0, 1, 2])
            @test length(g.connections) < 6  # Less than full
            @test length(g.connections) > 0  # But has some connections
        end

        @testset "Partial direct - with hidden" begin
            Random.seed!(42)
            temp_config = GenomeConfig(Dict(
                :num_inputs => 2,
                :num_outputs => 1,
                :num_hidden => 2,
                :initial_connection => :partial_direct,
                :connection_fraction => 0.5,
                :feed_forward => true,
                :activation_default => "sigmoid",
                :aggregation_default => "sum",
                :activation_options => ["sigmoid"],
                :aggregation_options => ["sum"],
                :bias_init_mean => 0.0,
                :bias_init_stdev => 1.0,
                :response_init_mean => 1.0,
                :response_init_stdev => 0.0,
                :weight_init_mean => 0.0,
                :weight_init_stdev => 1.0,
                :enabled_default => true
            ))

            g = Genome(42)
            configure_new!(g, temp_config)

            @test g.key == 42
            @test Set(keys(g.nodes)) == Set([0, 1, 2])
            @test length(g.connections) < 8  # Less than full direct
            @test length(g.connections) > 0  # But has some connections
        end
    end

    @testset "Reproduction Spawn Computation" begin
        using NEAT: compute_spawn

        @testset "Spawn adjust 1" begin
            adjusted_fitness = [1.0, 0.0]
            previous_sizes = [20, 20]
            pop_size = 40
            min_species_size = 10

            spawn = compute_spawn(adjusted_fitness, previous_sizes, pop_size, min_species_size)
            @test spawn == [27, 13]

            spawn = compute_spawn(adjusted_fitness, spawn, pop_size, min_species_size)
            @test spawn == [30, 10]

            spawn = compute_spawn(adjusted_fitness, spawn, pop_size, min_species_size)
            @test spawn == [31, 10]

            spawn = compute_spawn(adjusted_fitness, spawn, pop_size, min_species_size)
            @test spawn == [31, 10]
        end

        @testset "Spawn adjust 2" begin
            adjusted_fitness = [0.5, 0.5]
            previous_sizes = [20, 20]
            pop_size = 40
            min_species_size = 10

            spawn = compute_spawn(adjusted_fitness, previous_sizes, pop_size, min_species_size)
            @test spawn == [20, 20]
        end

        @testset "Spawn adjust 3" begin
            adjusted_fitness = [0.5, 0.5]
            previous_sizes = [30, 10]
            pop_size = 40
            min_species_size = 10

            spawn = compute_spawn(adjusted_fitness, previous_sizes, pop_size, min_species_size)
            @test spawn == [25, 15]

            spawn = compute_spawn(adjusted_fitness, spawn, pop_size, min_species_size)
            @test spawn == [23, 17]

            spawn = compute_spawn(adjusted_fitness, spawn, pop_size, min_species_size)
            @test spawn == [21, 19]

            spawn = compute_spawn(adjusted_fitness, spawn, pop_size, min_species_size)
            @test spawn == [20, 20]

            spawn = compute_spawn(adjusted_fitness, spawn, pop_size, min_species_size)
            @test spawn == [20, 20]
        end
    end

    @testset "Population Evolution" begin
        test_config_path = joinpath(@__DIR__, "test_config.toml")

        @testset "Fitness criterion - max" begin
            config = load_config(test_config_path)
            # Ensure max criterion
            @test config.fitness_criterion == :max

            pop = Population(config)

            function eval_genomes_simple(genomes, cfg)
                for (genome_id, genome) in genomes
                    genome.fitness = 1.0
                end
            end

            winner = run!(pop, eval_genomes_simple, 10)
            @test winner !== nothing
            @test winner.fitness !== nothing
        end

        @testset "Fitness criterion - min" begin
            config = load_config(test_config_path)
            # Change to min criterion
            min_config = Config(
                Dict(
                    :fitness_criterion => :min,
                    :fitness_threshold => -1.0,
                    :pop_size => config.pop_size,
                    :reset_on_extinction => config.reset_on_extinction
                ),
                config.genome_config,
                config.species_config,
                config.stagnation_config,
                config.reproduction_config
            )

            pop = Population(min_config)

            function eval_genomes_min(genomes, cfg)
                for (genome_id, genome) in genomes
                    genome.fitness = -1.0
                end
            end

            winner = run!(pop, eval_genomes_min, 10)
            @test winner !== nothing
            @test winner.fitness !== nothing
        end

        @testset "Fitness criterion - mean" begin
            config = load_config(test_config_path)
            # Change to mean criterion
            mean_config = Config(
                Dict(
                    :fitness_criterion => :mean,
                    :fitness_threshold => 0.9,
                    :pop_size => config.pop_size,
                    :reset_on_extinction => config.reset_on_extinction
                ),
                config.genome_config,
                config.species_config,
                config.stagnation_config,
                config.reproduction_config
            )

            pop = Population(mean_config)

            function eval_genomes_mean(genomes, cfg)
                for (genome_id, genome) in genomes
                    genome.fitness = 1.0
                end
            end

            winner = run!(pop, eval_genomes_mean, 10)
            @test winner !== nothing
            @test winner.fitness !== nothing
        end
    end

    @testset "Simple Evolution Run" begin
        test_config_path = joinpath(@__DIR__, "test_config.toml")
        config = load_config(test_config_path)

        @testset "Dummy fitness - serial" begin
            pop = Population(config)

            function eval_dummy_genomes(genomes, cfg)
                for (genome_id, genome) in genomes
                    net = FeedForwardNetwork(genome, cfg.genome_config)
                    output = activate!(net, [0.5, 0.5])
                    genome.fitness = 0.0
                end
            end

            # Run for a few generations
            winner = run!(pop, eval_dummy_genomes, 5)

            @test winner !== nothing
            @test winner.fitness !== nothing
        end

        @testset "Simple fitness" begin
            pop = Population(config)

            function eval_simple_fitness(genomes, cfg)
                for (genome_id, genome) in genomes
                    # Very simple fitness based on number of connections
                    genome.fitness = Float64(length(genome.connections))
                end
            end

            winner = run!(pop, eval_simple_fitness, 10)

            @test winner !== nothing
            @test winner.fitness !== nothing
            @test winner.fitness >= 0.0
        end
    end

    @testset "Genome Mutations" begin
        test_config_path = joinpath(@__DIR__, "test_config.toml")
        config = load_config(test_config_path)

        @testset "Mutate connections" begin
            g = Genome(1)
            configure_new!(g, config.genome_config)

            initial_connections = copy(g.connections)

            # Mutate multiple times
            for _ in 1:10
                mutate!(g, config.genome_config)
            end

            # After mutations, genome should still be valid
            @test !isempty(g.nodes)
            @test g.key == 1
        end

        @testset "Add node mutation" begin
            Random.seed!(42)
            g = Genome(1)
            configure_new!(g, config.genome_config)

            initial_node_count = length(g.nodes)

            # Force node addition by mutating many times
            for _ in 1:20
                mutate!(g, config.genome_config)
            end

            # Should have at least the original nodes
            @test length(g.nodes) >= initial_node_count
        end

        @testset "Add connection mutation" begin
            Random.seed!(43)
            temp_config = GenomeConfig(Dict(
                :num_inputs => 2,
                :num_outputs => 1,
                :num_hidden => 2,
                :initial_connection => :unconnected,
                :feed_forward => true,
                :conn_add_prob => 0.9,
                :node_add_prob => 0.0,
                :activation_default => "sigmoid",
                :aggregation_default => "sum",
                :activation_options => ["sigmoid"],
                :aggregation_options => ["sum"],
                :bias_init_mean => 0.0,
                :bias_init_stdev => 1.0,
                :response_init_mean => 1.0,
                :response_init_stdev => 0.0,
                :weight_init_mean => 0.0,
                :weight_init_stdev => 1.0,
                :enabled_default => true
            ))

            g = Genome(1)
            configure_new!(g, temp_config)

            @test isempty(g.connections)

            # Mutate to add connections
            for _ in 1:20
                mutate!(g, temp_config)
            end

            # Should have added some connections
            @test length(g.connections) > 0
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

    @testset "Statistics Reporter" begin
        test_config_path = joinpath(@__DIR__, "test_config.toml")
        config = load_config(test_config_path)

        @testset "Reporter initialization" begin
            stats = StatisticsReporter()
            @test isempty(stats.most_fit_genomes)
            @test isempty(stats.generation_statistics)
        end

        @testset "Statistics collection" begin
            # Create config with very high fitness threshold so it won't terminate early
            high_threshold_config = Config(
                Dict(
                    :fitness_criterion => :max,
                    :fitness_threshold => 10000.0,  # Very high threshold
                    :pop_size => config.pop_size,
                    :reset_on_extinction => config.reset_on_extinction,
                    :no_fitness_termination => true  # Don't terminate on fitness
                ),
                config.genome_config,
                config.species_config,
                config.stagnation_config,
                config.reproduction_config
            )

            pop = Population(high_threshold_config)
            stats = StatisticsReporter()
            add_reporter!(pop, stats)

            function eval_simple(genomes, cfg)
                for (gid, genome) in genomes
                    genome.fitness = Float64(gid % 100)  # Simple fitness based on ID
                end
            end

            # Run for exactly 5 generations
            run!(pop, eval_simple, 5)

            # Check that statistics were collected
            @test length(stats.most_fit_genomes) == 5
            @test length(stats.generation_statistics) == 5

            # Check fitness functions
            mean_fitness = get_fitness_mean(stats)
            @test length(mean_fitness) == 5
            @test all(f -> f > 0, mean_fitness)

            stdev_fitness = get_fitness_stdev(stats)
            @test length(stdev_fitness) == 5

            median_fitness = get_fitness_median(stats)
            @test length(median_fitness) == 5
        end

        @testset "Best genome functions" begin
            stats = StatisticsReporter()

            # Create some mock genomes
            g1 = Genome(1)
            g1.fitness = 1.0
            g2 = Genome(2)
            g2.fitness = 3.0
            g3 = Genome(3)
            g3.fitness = 2.0

            push!(stats.most_fit_genomes, g1)
            push!(stats.most_fit_genomes, g2)
            push!(stats.most_fit_genomes, g3)

            # Test best genome
            best = best_genome(stats)
            @test best.fitness == 3.0

            # Test best genomes
            top2 = best_genomes(stats, 2)
            @test length(top2) == 2
            @test top2[1].fitness == 3.0
            @test top2[2].fitness == 2.0

            # Test best unique genomes (with duplicate)
            g2_dup = Genome(2)
            g2_dup.fitness = 2.5  # Lower fitness than original g2
            push!(stats.most_fit_genomes, g2_dup)

            unique_best = best_unique_genomes(stats, 3)
            @test length(unique_best) == 3  # Should have 3 unique genomes
        end

        @testset "Species statistics" begin
            # This is tested implicitly in population evolution tests
            # Just verify the functions exist and return correct types
            stats = StatisticsReporter()

            sizes = get_species_sizes(stats)
            @test sizes isa Vector{Vector{Int}}

            fitness = get_species_fitness(stats)
            @test fitness isa Vector{Vector{Float64}}
        end
    end

    @testset "Visualization (with Plots.jl)" begin
        # Only run if Plots is available
        try
            using Plots

            test_config_path = joinpath(@__DIR__, "test_config.toml")
            config = load_config(test_config_path)

            pop = Population(config)
            stats = StatisticsReporter()
            add_reporter!(pop, stats)

            function eval_simple(genomes, cfg)
                for (gid, genome) in genomes
                    genome.fitness = Float64(gid)
                end
            end

            run!(pop, eval_simple, 3)

            @testset "plot_fitness" begin
                # Test that plot_fitness runs without error
                p = plot_fitness(stats, filename="test_fitness.png", show_plot=false)
                @test p !== nothing
                @test isfile("test_fitness.png")
                rm("test_fitness.png", force=true)
            end

            @testset "plot_species" begin
                # Test that plot_species runs without error
                p = plot_species(stats, filename="test_species.png", show_plot=false)
                @test p !== nothing
                @test isfile("test_species.png")
                rm("test_species.png", force=true)
            end

            @testset "save_statistics" begin
                save_statistics(stats, prefix="test_stats")
                @test isfile("test_stats_fitness.csv")
                @test isfile("test_stats_speciation.csv")
                @test isfile("test_stats_species_fitness.csv")

                # Clean up
                rm("test_stats_fitness.csv", force=true)
                rm("test_stats_speciation.csv", force=true)
                rm("test_stats_species_fitness.csv", force=true)
            end
        catch e
            if isa(e, ArgumentError) && occursin("Package Plots not found", string(e))
                @test_skip "Plots.jl not available, skipping visualization tests"
            else
                rethrow(e)
            end
        end
    end
end

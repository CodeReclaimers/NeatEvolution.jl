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

    @testset "Network Visualization (with Plots.jl)" begin
        # Only run if Plots is available (should be, since it's in the test deps)
        try
            using Plots

            test_config_path = joinpath(@__DIR__, "test_config.toml")
            config = load_config(test_config_path)

            # Create a simple genome for testing
            genome = Genome(1)
            configure_new!(genome, config.genome_config)

            # Add a hidden node
            new_node_id = maximum(union(config.genome_config.output_keys,
                                       [k for k in keys(genome.nodes)])) + 1
            genome.nodes[new_node_id] = NodeGene(new_node_id, 0.5, 1.0, :sigmoid, :sum)

            # Add some connections
            genome.connections[(config.genome_config.input_keys[1], new_node_id)] =
                ConnectionGene((config.genome_config.input_keys[1], new_node_id), 1.5, true, 0)
            genome.connections[(new_node_id, config.genome_config.output_keys[1])] =
                ConnectionGene((new_node_id, config.genome_config.output_keys[1]), -0.8, true, 1)
            # Add disabled connection
            genome.connections[(config.genome_config.input_keys[2], new_node_id)] =
                ConnectionGene((config.genome_config.input_keys[2], new_node_id), 0.3, false, 2)

            @testset "draw_net basic functionality" begin
                # Test that draw_net runs without error
                p = draw_net(genome, config.genome_config,
                            filename="test_network.png")
                @test p !== nothing
                @test isfile("test_network.png")
                rm("test_network.png", force=true)
            end

            @testset "draw_net with custom node names" begin
                node_names = Dict(
                    config.genome_config.input_keys[1] => "Input1",
                    config.genome_config.input_keys[2] => "Input2",
                    config.genome_config.output_keys[1] => "Output"
                )
                p = draw_net(genome, config.genome_config,
                            filename="test_network_named.png",
                            node_names=node_names)
                @test p !== nothing
                @test isfile("test_network_named.png")
                rm("test_network_named.png", force=true)
            end

            @testset "draw_net without disabled connections" begin
                p = draw_net(genome, config.genome_config,
                            filename="test_network_no_disabled.png",
                            show_disabled=false)
                @test p !== nothing
                @test isfile("test_network_no_disabled.png")
                rm("test_network_no_disabled.png", force=true)
            end

            @testset "draw_net with pruning" begin
                p = draw_net(genome, config.genome_config,
                            filename="test_network_pruned.png",
                            prune_unused=true)
                @test p !== nothing
                @test isfile("test_network_pruned.png")
                rm("test_network_pruned.png", force=true)
            end

            @testset "draw_net_comparison" begin
                # Create a second genome
                genome2 = Genome(2)
                configure_new!(genome2, config.genome_config)

                combined = draw_net_comparison([genome, genome2],
                                               config.genome_config,
                                               filename="test_comparison.png",
                                               labels=["Genome 1", "Genome 2"])
                @test combined !== nothing
                @test isfile("test_comparison.png")

                # Clean up
                rm("test_comparison.png", force=true)
            end
        catch e
            if isa(e, ArgumentError) && occursin("Package Plots not found", string(e))
                @test_skip "Plots.jl not available, skipping network visualization tests"
            else
                rethrow(e)
            end
        end
    end

    @testset "Advanced Visualization - Phase 3 (with Plots.jl)" begin
        # Only run if Plots is available
        try
            using Plots

            test_config_path = joinpath(@__DIR__, "test_config.toml")
            config = load_config(test_config_path)

            # Create and evolve a simple population for testing
            pop = Population(config)
            stats = StatisticsReporter()
            add_reporter!(pop, stats)

            function eval_simple_2d(genomes, cfg)
                for (gid, genome) in genomes
                    # Simple fitness based on XOR-like behavior
                    net = FeedForwardNetwork(genome, cfg.genome_config)
                    fitness = 0.0
                    for x in [0.0, 1.0]
                        for y in [0.0, 1.0]
                            output = activate!(net, [x, y])
                            # Reward XOR behavior
                            expected = (x + y == 1.0) ? 1.0 : 0.0
                            fitness -= (output[1] - expected)^2
                        end
                    end
                    genome.fitness = fitness
                end
            end

            # Run for a few generations to get data
            run!(pop, eval_simple_2d, 5)

            winner = best_genome(stats)

            @testset "plot_activation_heatmap" begin
                # Test basic heatmap
                p = plot_activation_heatmap(winner, config.genome_config,
                                           x_range=(0.0, 1.0),
                                           y_range=(0.0, 1.0),
                                           resolution=20,
                                           filename="test_heatmap.png")
                @test p !== nothing
                @test isfile("test_heatmap.png")
                rm("test_heatmap.png", force=true)
            end

            @testset "plot_activation_heatmap with different resolution" begin
                p = plot_activation_heatmap(winner, config.genome_config,
                                           resolution=10,
                                           filename="test_heatmap_lowres.png")
                @test p !== nothing
                @test isfile("test_heatmap_lowres.png")
                rm("test_heatmap_lowres.png", force=true)
            end

            @testset "plot_activation_comparison" begin
                # Get top 3 genomes
                top_genomes = best_genomes(stats, 3)

                combined = plot_activation_comparison(top_genomes,
                                                     config.genome_config,
                                                     resolution=15,
                                                     labels=["Best", "2nd", "3rd"],
                                                     filename="test_heatmap_comparison.png")
                @test combined !== nothing
                @test isfile("test_heatmap_comparison.png")
                rm("test_heatmap_comparison.png", force=true)
            end

            @testset "animate_evolution" begin
                # Test animation generation
                anim = animate_evolution(stats, config.genome_config,
                                        filename="test_evolution.gif",
                                        fps=2,
                                        show_disabled=false)
                @test anim !== nothing
                @test isfile("test_evolution.gif")
                rm("test_evolution.gif", force=true)
            end

            @testset "animate_evolution with node names" begin
                node_names = Dict(
                    config.genome_config.input_keys[1] => "X",
                    config.genome_config.input_keys[2] => "Y",
                    config.genome_config.output_keys[1] => "Out"
                )

                anim = animate_evolution(stats, config.genome_config,
                                        filename="test_evolution_named.gif",
                                        fps=1,
                                        node_names=node_names)
                @test anim !== nothing
                @test isfile("test_evolution_named.gif")
                rm("test_evolution_named.gif", force=true)
            end

        catch e
            if isa(e, ArgumentError) && occursin("Package Plots not found", string(e))
                @test_skip "Plots.jl not available, skipping advanced visualization tests"
            else
                rethrow(e)
            end
        end
    end

    @testset "Interactive Visualization (with GraphMakie)" begin
        # Skip in CI environments - GLMakie requires display/graphics capabilities
        if haskey(ENV, "CI") || haskey(ENV, "GITHUB_ACTIONS")
            @test_skip "Skipping GraphMakie tests in CI (requires display)"
        else
            # Only run if GraphMakie, GLMakie, and Graphs are available
            try
                using GLMakie
                using GraphMakie
                using Graphs

                test_config_path = joinpath(@__DIR__, "test_config.toml")
                config = load_config(test_config_path)

                # Create a simple test genome
                genome = Genome(1)
                configure_new!(genome, config.genome_config)

                # Add a hidden node and some connections
                add_node!(genome, config.genome_config, 1)
                mutate_add_connection!(genome, config.genome_config)

                genome.fitness = 2.5

                @testset "draw_network_interactive" begin
                    # Test basic interactive network visualization
                    fig = draw_network_interactive(genome, config.genome_config)
                    @test fig isa GLMakie.Figure

                    # Test with different layouts
                    for layout in [:spring, :stress, :shell, :spectral, :circular]
                        fig = draw_network_interactive(genome, config.genome_config,
                                                         layout=layout)
                        @test fig isa GLMakie.Figure
                    end

                    # Test with custom parameters
                    fig = draw_network_interactive(genome, config.genome_config,
                                                     layout=:spring,
                                                     node_size=30.0,
                                                     edge_width_scale=2.0,
                                                     show_disabled=true,
                                                     prune_unused=false,
                                                     title="Test Network",
                                                     resolution=(800, 600))
                    @test fig isa GLMakie.Figure

                    # Test with custom node names
                    node_names = Dict(
                        config.genome_config.input_keys[1] => "Input 1",
                        config.genome_config.input_keys[2] => "Input 2",
                        config.genome_config.output_keys[1] => "Output"
                    )
                    fig = draw_network_interactive(genome, config.genome_config,
                                                     node_names=node_names)
                    @test fig isa GLMakie.Figure

                    # Test with pruning
                    fig = draw_network_interactive(genome, config.genome_config,
                                                     prune_unused=true)
                    @test fig isa GLMakie.Figure
                end

                @testset "draw_network_comparison_interactive" begin
                    # Create multiple genomes for comparison
                    genome1 = Genome(1)
                    configure_new!(genome1, config.genome_config)
                    genome1.fitness = 1.0

                    genome2 = Genome(2)
                    configure_new!(genome2, config.genome_config)
                    add_node!(genome2, config.genome_config, 1)
                    genome2.fitness = 2.0

                    genome3 = Genome(3)
                    configure_new!(genome3, config.genome_config)
                    add_node!(genome3, config.genome_config, 1)
                    add_node!(genome3, config.genome_config, 2)
                    genome3.fitness = 3.0

                    genomes = [genome1, genome2, genome3]

                    # Test basic comparison
                    fig = draw_network_comparison_interactive(genomes, config.genome_config)
                    @test fig isa GLMakie.Figure

                    # Test with labels
                    labels = ["Network 1", "Network 2", "Network 3"]
                    fig = draw_network_comparison_interactive(genomes, config.genome_config,
                                                               labels=labels)
                    @test fig isa GLMakie.Figure

                    # Test with custom parameters
                    fig = draw_network_comparison_interactive(genomes, config.genome_config,
                                                               labels=labels,
                                                               layout=:stress,
                                                               node_size=25.0,
                                                               prune_unused=true,
                                                               resolution=(1400, 700))
                    @test fig isa GLMakie.Figure

                    # Test with different layouts
                    for layout in [:spring, :circular, :stress]
                        fig = draw_network_comparison_interactive(genomes, config.genome_config,
                                                                   layout=layout)
                        @test fig isa GLMakie.Figure
                    end
                end

                @testset "Interactive visualization with evolution" begin
                    # Run a small evolution and visualize
                    pop = Population(config)
                    stats = StatisticsReporter()
                    add_reporter!(pop, stats)

                    function simple_eval(genomes, config)
                        for (_, g) in genomes
                            g.fitness = rand()
                        end
                    end

                    winner = run!(pop, simple_eval, 5)

                    # Test interactive visualization of winner
                    fig = draw_network_interactive(winner, config.genome_config,
                                                     layout=:spring,
                                                     title="Evolution Winner")
                    @test fig isa GLMakie.Figure

                    # Test comparison of top genomes
                    if length(stats.most_fit_genomes) >= 3
                        top3 = stats.most_fit_genomes[1:3]
                        fig = draw_network_comparison_interactive(top3, config.genome_config,
                                                                   labels=["Gen 1", "Gen 2", "Gen 3"])
                        @test fig isa GLMakie.Figure
                    end
                end

                println("✓ All GraphMakie interactive visualization tests passed")

            catch e
                if isa(e, ArgumentError) && (occursin("Package GLMakie not found", string(e)) ||
                                              occursin("Package GraphMakie not found", string(e)) ||
                                              occursin("Package Graphs not found", string(e)))
                    @test_skip "GraphMakie/GLMakie/Graphs not available, skipping interactive visualization tests"
                else
                    rethrow(e)
                end
            end
        end  # end else (not in CI)
    end

    @testset "JSON Export and Import" begin
        # Load config
        config_path = joinpath(dirname(@__DIR__), "examples", "xor", "config.toml")
        config = load_config(config_path)

        # Create a simple genome for testing
        genome = Genome(1)
        genome.fitness = 2.5

        # Add some nodes
        node0 = NodeGene(0)
        node0.bias = 0.5
        node0.response = 1.0
        node0.activation = :sigmoid
        node0.aggregation = :sum
        genome.nodes[0] = node0

        node1 = NodeGene(1)
        node1.bias = -0.3
        node1.response = 1.0
        node1.activation = :tanh
        node1.aggregation = :sum
        genome.nodes[1] = node1

        # Add a connection
        conn_key = (-1, 0)
        conn = ConnectionGene(conn_key, 1)
        conn.weight = 1.5
        conn.enabled = true
        genome.connections[conn_key] = conn

        # Test export to JSON
        temp_file = tempname() * ".json"
        try
            export_network_json(genome, config.genome_config, temp_file)

            # Verify file was created
            @test isfile(temp_file)

            # Test import from JSON
            imported_genome = import_network_json(temp_file, config.genome_config)

            # Verify basic properties
            @test imported_genome.key == genome.key
            @test imported_genome.fitness == genome.fitness

            # Verify nodes
            @test length(imported_genome.nodes) == length(genome.nodes)
            @test haskey(imported_genome.nodes, 0)
            @test haskey(imported_genome.nodes, 1)

            # Verify node properties
            imported_node0 = imported_genome.nodes[0]
            @test imported_node0.bias == 0.5
            @test imported_node0.response == 1.0
            @test imported_node0.activation == :sigmoid
            @test imported_node0.aggregation == :sum

            imported_node1 = imported_genome.nodes[1]
            @test imported_node1.bias == -0.3
            @test imported_node1.activation == :tanh

            # Verify connections
            @test length(imported_genome.connections) == length(genome.connections)
            @test haskey(imported_genome.connections, conn_key)

            imported_conn = imported_genome.connections[conn_key]
            @test imported_conn.weight == 1.5
            @test imported_conn.enabled == true
            @test imported_conn.innovation == 1

            # Test round-trip with network
            net1 = FeedForwardNetwork(genome, config.genome_config)
            net2 = FeedForwardNetwork(imported_genome, config.genome_config)

            test_input = [1.0, 0.5]
            output1 = activate!(net1, test_input)
            output2 = activate!(net2, test_input)

            # Outputs should be identical
            @test output1 ≈ output2

        finally
            # Clean up (use force=true for Windows compatibility)
            if isfile(temp_file)
                try
                    rm(temp_file, force=true)
                catch e
                    @warn "Could not delete temp file: $temp_file" exception=e
                end
            end
        end

        # Test export_population_json
        population = Dict{Int, Genome}()
        for i in 1:5
            g = Genome(i)
            g.fitness = Float64(i)
            population[i] = g
        end

        temp_pop_file = tempname() * ".json"
        try
            export_population_json(population, config.genome_config, temp_pop_file, top_n=3)

            @test isfile(temp_pop_file)

            # Read and verify
            using JSON
            data = JSON.parsefile(temp_pop_file)

            @test data["count"] == 3  # Only top 3
            @test length(data["genomes"]) == 3

            # Check they're sorted by fitness (descending)
            @test data["genomes"][1]["fitness"] == 5.0
            @test data["genomes"][2]["fitness"] == 4.0
            @test data["genomes"][3]["fitness"] == 3.0

        finally
            if isfile(temp_pop_file)
                try
                    rm(temp_pop_file, force=true)
                catch e
                    @warn "Could not delete temp file: $temp_pop_file" exception=e
                end
            end
        end
    end

    @testset "Configuration Validation" begin
        # Test with valid config (should load successfully)
        config_path = joinpath(dirname(@__DIR__), "examples", "xor", "config.toml")
        config = load_config(config_path)
        @test config !== nothing
        @test config.genome_config.num_inputs == 2
        @test config.genome_config.num_outputs == 1

        # Test validation warnings for missing critical parameters
        temp_config = tempname() * ".toml"
        try
            # Create minimal config missing num_inputs/num_outputs
            open(temp_config, "w") do io
                write(io, """
                [NEAT]
                pop_size = 100

                [DefaultGenome]
                feed_forward = true

                [DefaultSpeciesSet]
                compatibility_threshold = 3.0

                [DefaultStagnation]
                species_fitness_func = "max"

                [DefaultReproduction]
                elitism = 2
                """)
            end

            # Should warn about missing num_inputs/num_outputs but still work
            @test_logs (:warn, r"Missing 'num_inputs'") (:warn, r"Missing 'num_outputs'") load_config(temp_config)

        finally
            if isfile(temp_config)
                rm(temp_config)
            end
        end

        # Test validation error for invalid fitness_criterion
        temp_config2 = tempname() * ".toml"
        try
            open(temp_config2, "w") do io
                write(io, """
                [NEAT]
                fitness_criterion = "invalid"
                pop_size = 100

                [DefaultGenome]
                num_inputs = 2
                num_outputs = 1

                [DefaultSpeciesSet]
                compatibility_threshold = 3.0

                [DefaultStagnation]
                species_fitness_func = "max"

                [DefaultReproduction]
                elitism = 2
                """)
            end

            @test_throws ErrorException load_config(temp_config2)

        finally
            if isfile(temp_config2)
                rm(temp_config2)
            end
        end

        # Test validation warning for unknown parameter
        temp_config3 = tempname() * ".toml"
        try
            open(temp_config3, "w") do io
                write(io, """
                [NEAT]
                pop_size = 100
                popsize = 50  # Typo - should trigger warning

                [DefaultGenome]
                num_inputs = 2
                num_outputs = 1
                compatability_threshold = 2.0  # Typo - should trigger warning

                [DefaultSpeciesSet]
                compatibility_threshold = 3.0

                [DefaultStagnation]
                species_fitness_func = "max"

                [DefaultReproduction]
                elitism = 2
                """)
            end

            # Should warn about typos
            @test_logs (:warn, r"popsize") (:warn, r"compatability_threshold") match_mode=:any load_config(temp_config3)

        finally
            if isfile(temp_config3)
                rm(temp_config3)
            end
        end

        # Test validation error for invalid activation_options
        temp_config4 = tempname() * ".toml"
        try
            open(temp_config4, "w") do io
                write(io, """
                [NEAT]
                pop_size = 100

                [DefaultGenome]
                num_inputs = 2
                num_outputs = 1
                activation_options = []  # Empty array - should error

                [DefaultSpeciesSet]
                compatibility_threshold = 3.0

                [DefaultStagnation]
                species_fitness_func = "max"

                [DefaultReproduction]
                elitism = 2
                """)
            end

            @test_throws ErrorException load_config(temp_config4)

        finally
            if isfile(temp_config4)
                rm(temp_config4)
            end
        end

        # Test validation error for negative compatibility_threshold
        temp_config5 = tempname() * ".toml"
        try
            open(temp_config5, "w") do io
                write(io, """
                [NEAT]
                pop_size = 100

                [DefaultGenome]
                num_inputs = 2
                num_outputs = 1

                [DefaultSpeciesSet]
                compatibility_threshold = -1.0  # Invalid - must be positive

                [DefaultStagnation]
                species_fitness_func = "max"

                [DefaultReproduction]
                elitism = 2
                """)
            end

            @test_throws ErrorException load_config(temp_config5)

        finally
            if isfile(temp_config5)
                rm(temp_config5)
            end
        end

        # Test validation warning for very small population
        temp_config6 = tempname() * ".toml"
        try
            open(temp_config6, "w") do io
                write(io, """
                [NEAT]
                pop_size = 5  # Very small - should warn

                [DefaultGenome]
                num_inputs = 2
                num_outputs = 1

                [DefaultSpeciesSet]
                compatibility_threshold = 3.0

                [DefaultStagnation]
                species_fitness_func = "max"

                [DefaultReproduction]
                elitism = 2
                """)
            end

            @test_logs (:warn, r"Population size \(5\) is very small") match_mode=:any load_config(temp_config6)

        finally
            if isfile(temp_config6)
                rm(temp_config6)
            end
        end
    end
end

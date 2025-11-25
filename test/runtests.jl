using Test
using NEAT
using Random

# Include test files
include("test_genes.jl")
include("test_population_seeding.jl")

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

        @testset "Nodes with no inputs (orphaned nodes)" begin
            # Test that orphaned nodes (no incoming connections) are handled correctly
            # These can occur after deletion mutations

            # Test 1: Single orphaned hidden node feeding output
            inputs = [-1]
            outputs = [0]
            connections = [(1, 0)]  # Hidden node 1 -> output, but 1 has no inputs

            required = required_for_output(inputs, outputs, connections)
            @test 0 in required
            @test 1 in required  # Node 1 is required even though it has no inputs

            layers = feed_forward_layers(inputs, outputs, connections)
            @test length(layers) == 2
            @test 1 in layers[1]  # Orphaned node in first layer (as bias neuron)
            @test 0 in layers[2]  # Output in second layer

            # Test 2: Multiple orphaned nodes
            inputs = [-1, -2]
            outputs = [0]
            connections = [(1, 0), (2, 0)]  # Two orphans feeding output

            required = required_for_output(inputs, outputs, connections)
            @test 1 in required
            @test 2 in required

            layers = feed_forward_layers(inputs, outputs, connections)
            @test 1 in layers[1]
            @test 2 in layers[1]
            @test 0 in layers[2]

            # Test 3: Mixed - some nodes with inputs, some without
            inputs = [-1]
            outputs = [0]
            connections = [(-1, 2), (1, 0), (2, 0)]  # Node 1 orphaned, node 2 has input

            required = required_for_output(inputs, outputs, connections)
            @test 1 in required  # Orphan
            @test 2 in required  # Normal

            layers = feed_forward_layers(inputs, outputs, connections)
            @test length(layers) == 3
            @test 1 in layers[1]  # Orphan in first layer
            @test 2 in layers[2]  # Normal node in second layer (after inputs)
            @test 0 in layers[3]  # Output in third layer

            # Test 4: Orphaned output node
            inputs = [-1]
            outputs = [0]
            connections = Tuple{Int, Int}[]  # No connections at all!

            required = required_for_output(inputs, outputs, connections)
            @test 0 in required

            layers = feed_forward_layers(inputs, outputs, connections)
            @test length(layers) == 1
            @test 0 in layers[1]  # Output node with no inputs in first layer
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

        @testset "Orphaned node integration test" begin
            # Test network with orphaned (no-input) node created from genome
            config_path = joinpath(dirname(@__DIR__), "examples", "xor", "config.toml")
            config = load_config(config_path)

            genome = Genome(1)

            # Add output node
            output_node = NodeGene(0)
            output_node.bias = 0.0
            output_node.response = 1.0
            output_node.activation = :sigmoid
            output_node.aggregation = :sum
            genome.nodes[0] = output_node

            # Add orphaned hidden node (no incoming connections)
            hidden_node = NodeGene(1)
            hidden_node.bias = 3.0  # Will output sigmoid(5*3.0) due to 5x scaling
            hidden_node.response = 1.0
            hidden_node.activation = :sigmoid
            hidden_node.aggregation = :sum
            genome.nodes[1] = hidden_node

            # Hidden -> output (weight=2.0)
            conn_key = (1, 0)
            conn = ConnectionGene(conn_key, 1)
            conn.weight = 2.0
            conn.enabled = true
            genome.connections[conn_key] = conn

            # Input -1 -> output (weight=1.0)
            conn_key2 = (-1, 0)
            conn2 = ConnectionGene(conn_key2, 2)
            conn2.weight = 1.0
            conn2.enabled = true
            genome.connections[conn_key2] = conn2

            # Create network
            net = FeedForwardNetwork(genome, config.genome_config)

            # Network should have 2 node evals: hidden (orphan) and output
            @test length(net.node_evals) == 2

            # Verify orphan node has no inputs
            orphan_eval = net.node_evals[1]
            @test orphan_eval[1] == 1  # Node ID
            @test orphan_eval[4] == 3.0  # Bias
            @test isempty(orphan_eval[6])  # No input links

            # Activate network
            output = activate!(net, [1.0, 0.5])

            # Expected calculation:
            # Hidden (node 1): sigmoid_activation(3.0) = sigmoid(5*3.0) = sigmoid(15.0)
            hidden_val = 1.0 / (1.0 + exp(-15.0))
            # Output: sigmoid_activation(0.5*1.0 + hidden_val*2.0)
            sum_val = 0.5 * 1.0 + hidden_val * 2.0
            expected = 1.0 / (1.0 + exp(-5.0 * sum_val))

            @test output[1] ≈ expected atol=1e-6
            @test net.values[1] ≈ hidden_val atol=1e-6
        end
    end

    @testset "Feed-Forward Network Input Validation" begin
        config_path = joinpath(dirname(@__DIR__), "examples", "xor", "config.toml")
        config = load_config(config_path)

        # Create a simple genome with 2 inputs, 1 output
        genome = Genome(1)
        configure_new!(genome, config.genome_config)

        # Build network
        net = FeedForwardNetwork(genome, config.genome_config)

        # Test with correct number of inputs (should work)
        inputs_correct = [1.0, 0.0]
        output = activate!(net, inputs_correct)
        @test length(output) == 1

        # Test with too few inputs (should error)
        inputs_too_few = [1.0]
        @test_throws ErrorException activate!(net, inputs_too_few)

        # Verify error message is informative
        try
            activate!(net, inputs_too_few)
            @test false  # Should not reach here
        catch e
            @test occursin(r"Expected \d+ inputs?, got \d+", e.msg)
        end

        # Test with too many inputs (should error)
        inputs_too_many = [1.0, 0.0, 0.5]
        @test_throws ErrorException activate!(net, inputs_too_many)

        # Test with empty inputs (should error)
        inputs_empty = Float64[]
        @test_throws ErrorException activate!(net, inputs_empty)
    end

    @testset "Feed-Forward Network Pruning" begin
        config_path = joinpath(dirname(@__DIR__), "examples", "xor", "config.toml")
        config = load_config(config_path)
        rng = MersenneTwister(42)

        # Manually construct a genome with an unreachable node
        genome = Genome(1)

        # Add output node (key 0)
        output_node = NodeGene(0)
        NEAT.init_attributes!(output_node, config.genome_config, rng)
        genome.nodes[0] = output_node

        # Add reachable hidden node (key 1)
        hidden_reachable = NodeGene(1)
        NEAT.init_attributes!(hidden_reachable, config.genome_config, rng)
        genome.nodes[1] = hidden_reachable

        # Add unreachable hidden node (key 2) - no path to output
        hidden_unreachable = NodeGene(2)
        NEAT.init_attributes!(hidden_unreachable, config.genome_config, rng)
        genome.nodes[2] = hidden_unreachable

        # Connect input -1 to reachable hidden node 1
        conn1 = ConnectionGene((-1, 1), 1)
        NEAT.init_attributes!(conn1, config.genome_config, rng)
        genome.connections[(-1, 1)] = conn1

        # Connect reachable hidden node 1 to output 0
        conn2 = ConnectionGene((1, 0), 2)
        NEAT.init_attributes!(conn2, config.genome_config, rng)
        genome.connections[(1, 0)] = conn2

        # Connect input -2 to unreachable hidden node 2
        # (but node 2 doesn't connect to output, so it should be pruned)
        conn3 = ConnectionGene((-2, 2), 3)
        NEAT.init_attributes!(conn3, config.genome_config, rng)
        genome.connections[(-2, 2)] = conn3

        # Build network - should prune unreachable node 2
        net = FeedForwardNetwork(genome, config.genome_config)

        # Verify node 2 is not in the network's evaluation list
        node_keys_in_network = [eval[1] for eval in net.node_evals]  # First element is node_id
        @test 0 in node_keys_in_network  # Output should be present
        @test 1 in node_keys_in_network  # Reachable hidden should be present
        @test !(2 in node_keys_in_network)  # Unreachable hidden should be pruned

        # Verify network still functions correctly
        inputs = [1.0, 0.0]
        output = activate!(net, inputs)
        @test length(output) == 1
        @test !isnan(output[1]) && !isinf(output[1])
    end

    @testset "Feed-Forward Network with Disabled Connections" begin
        config_path = joinpath(dirname(@__DIR__), "examples", "xor", "config.toml")
        config = load_config(config_path)
        rng = MersenneTwister(123)

        # Create a genome with some disabled connections
        genome = Genome(1)

        # Add output node
        output_node = NodeGene(0)
        NEAT.init_attributes!(output_node, config.genome_config, rng)
        genome.nodes[0] = output_node

        # Add hidden node
        hidden_node = NodeGene(1)
        NEAT.init_attributes!(hidden_node, config.genome_config, rng)
        genome.nodes[1] = hidden_node

        # Add enabled connection: input -1 -> hidden 1
        conn1 = ConnectionGene((-1, 1), 1)
        NEAT.init_attributes!(conn1, config.genome_config, rng)
        conn1.weight = 1.0
        conn1.enabled = true
        genome.connections[(-1, 1)] = conn1

        # Add enabled connection: hidden 1 -> output 0
        conn2 = ConnectionGene((1, 0), 2)
        NEAT.init_attributes!(conn2, config.genome_config, rng)
        conn2.weight = 1.0
        conn2.enabled = true
        genome.connections[(1, 0)] = conn2

        # Add disabled connection: input -2 -> output 0
        # This should be ignored when building the network
        conn3 = ConnectionGene((-2, 0), 3)
        NEAT.init_attributes!(conn3, config.genome_config, rng)
        conn3.weight = 10.0  # Large weight that would affect output if not disabled
        conn3.enabled = false
        genome.connections[(-2, 0)] = conn3

        # Build network
        net = FeedForwardNetwork(genome, config.genome_config)

        # Test that input -2 doesn't affect output (connection is disabled)
        output1 = activate!(net, [1.0, 0.0])
        output2 = activate!(net, [1.0, 1.0])

        # If the disabled connection were active, output2 would be significantly different
        # But since it's disabled, only the path through node 1 matters
        @test !isnan(output1[1]) && !isinf(output1[1])
        @test !isnan(output2[1]) && !isinf(output2[1])

        # Verify that the disabled connection is not in the network structure
        # Check that node 0's inputs don't include a direct connection from input -2
        output_eval = net.node_evals[end]  # Output should be last
        @test output_eval[1] == 0  # First element is node_id

        # The output should only receive input from hidden node 1, not directly from input -2
        # Sixth element (index 6) contains the input links as Vector{Tuple{Int, Float64}}
        input_sources = [link[1] for link in output_eval[6]]  # First element of each link is input_id
        @test 1 in input_sources  # Should have input from hidden node 1
        @test !(-2 in input_sources)  # Should NOT have direct input from -2
    end

    @testset "Network with Self-Connections" begin
        config_path = joinpath(dirname(@__DIR__), "examples", "xor", "config.toml")
        config = load_config(config_path)

        # Check if the config allows recurrent networks
        # If feed_forward is true, skip this test or modify config temporarily
        if config.genome_config.feed_forward
            @info "Skipping self-connection test (feed_forward=true in config)"
        else
            rng = MersenneTwister(456)

            # Create genome with self-connection
            genome = Genome(1)

            output_node = NodeGene(0)
            NEAT.init_attributes!(output_node, config.genome_config, rng)
            genome.nodes[0] = output_node

            hidden_node = NodeGene(1)
            NEAT.init_attributes!(hidden_node, config.genome_config, rng)
            genome.nodes[1] = hidden_node

            # Input -> hidden
            conn1 = ConnectionGene((-1, 1), 1)
            NEAT.init_attributes!(conn1, config.genome_config, rng)
            genome.connections[(-1, 1)] = conn1

            # Hidden -> output
            conn2 = ConnectionGene((1, 0), 2)
            NEAT.init_attributes!(conn2, config.genome_config, rng)
            genome.connections[(1, 0)] = conn2

            # Self-connection on hidden node
            conn3 = ConnectionGene((1, 1), 3)
            NEAT.init_attributes!(conn3, config.genome_config, rng)
            genome.connections[(1, 1)] = conn3

            # For feed-forward network, this should detect the cycle
            @test_throws ErrorException FeedForwardNetwork(genome, config.genome_config)
        end
    end

    @testset "Complex Multi-Layer Network" begin
        config_path = joinpath(dirname(@__DIR__), "examples", "xor", "config.toml")
        config = load_config(config_path)
        rng = MersenneTwister(789)

        # Build a 2-3-2-1 network (2 inputs, 3 hidden layer 1, 2 hidden layer 2, 1 output)
        genome = Genome(1)

        # Output layer (node 0)
        output = NodeGene(0)
        NEAT.init_attributes!(output, config.genome_config, rng)
        genome.nodes[0] = output

        # Hidden layer 2 (nodes 1, 2)
        for i in 1:2
            hidden2 = NodeGene(i)
            NEAT.init_attributes!(hidden2, config.genome_config, rng)
            genome.nodes[i] = hidden2
        end

        # Hidden layer 1 (nodes 3, 4, 5)
        for i in 3:5
            hidden1 = NodeGene(i)
            NEAT.init_attributes!(hidden1, config.genome_config, rng)
            genome.nodes[i] = hidden1
        end

        innovation = 1

        # Connect inputs to hidden layer 1
        for input_id in [-1, -2]
            for hidden_id in 3:5
                conn = ConnectionGene((input_id, hidden_id), innovation)
                NEAT.init_attributes!(conn, config.genome_config, rng)
                genome.connections[(input_id, hidden_id)] = conn
                innovation += 1
            end
        end

        # Connect hidden layer 1 to hidden layer 2
        for h1_id in 3:5
            for h2_id in 1:2
                conn = ConnectionGene((h1_id, h2_id), innovation)
                NEAT.init_attributes!(conn, config.genome_config, rng)
                genome.connections[(h1_id, h2_id)] = conn
                innovation += 1
            end
        end

        # Connect hidden layer 2 to output
        for h2_id in 1:2
            conn = ConnectionGene((h2_id, 0), innovation)
            NEAT.init_attributes!(conn, config.genome_config, rng)
            genome.connections[(h2_id, 0)] = conn
            innovation += 1
        end

        # Build and test network
        net = FeedForwardNetwork(genome, config.genome_config)

        # Verify all nodes are present (should have all layers)
        node_keys = [eval[1] for eval in net.node_evals]  # First element is node_id
        @test 0 in node_keys  # Output
        @test all(i in node_keys for i in 1:2)  # Hidden layer 2
        # Note: Some nodes in layer 1 may be pruned if they don't contribute to output
        # This is normal NEAT behavior - unused nodes are removed
        @test length(node_keys) >= 3  # At least output + some hidden nodes

        # Test network evaluation
        inputs = [0.5, -0.3]
        output = activate!(net, inputs)
        @test length(output) == 1
        @test !isnan(output[1]) && !isinf(output[1])

        # Test determinism: same inputs should give same output
        output2 = activate!(net, inputs)
        @test output[1] == output2[1]

        # Test different inputs give different outputs (probabilistically)
        output3 = activate!(net, [1.0, 1.0])
        @test output[1] != output3[1]  # Should be different with different inputs
    end

    @testset "Network with Various Activation Functions" begin
        config_path = joinpath(dirname(@__DIR__), "examples", "xor", "config.toml")
        config = load_config(config_path)
        rng = MersenneTwister(321)

        # Test each available activation function
        activations = [:sigmoid, :tanh, :relu, :sin, :gauss, :softplus,
                       :identity, :clamped, :abs, :square, :cube]

        for activation in activations
            genome = Genome(1)

            # Output node with specific activation
            output = NodeGene(0)
            NEAT.init_attributes!(output, config.genome_config, rng)
            output.activation = activation
            genome.nodes[0] = output

            # Direct connection from input to output
            conn = ConnectionGene((-1, 0), 1)
            NEAT.init_attributes!(conn, config.genome_config, rng)
            conn.weight = 1.0
            genome.connections[(-1, 0)] = conn

            # Build network
            net = FeedForwardNetwork(genome, config.genome_config)

            # Test with various inputs
            # Config expects 2 inputs, so provide 2 (only first one is used due to our connection)
            for test_input in [-2.0, -1.0, 0.0, 1.0, 2.0]
                inputs = [test_input, 0.0]  # Second input is unused but required by config
                output_val = activate!(net, inputs)

                # Verify output is valid (not NaN or Inf)
                @test !isnan(output_val[1])
                @test !isinf(output_val[1])
            end
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

        @testset "Add node mutation - bias neutrality" begin
            # Regression test for node bias initialization
            # Ensures new nodes added via mutation have bias=0.0 per NEAT paper
            Random.seed!(100)
            temp_config = GenomeConfig(Dict(
                :num_inputs => 2,
                :num_outputs => 1,
                :num_hidden => 0,
                :initial_connection => :full,
                :feed_forward => true,
                :conn_add_prob => 0.0,
                :node_add_prob => 1.0,  # Force node addition
                :activation_default => "sigmoid",
                :aggregation_default => "sum",
                :activation_options => ["sigmoid"],
                :aggregation_options => ["sum"],
                :bias_init_mean => 5.0,  # Non-zero mean
                :bias_init_stdev => 2.0,  # Non-zero stdev
                :response_init_mean => 1.0,
                :response_init_stdev => 0.0,
                :weight_init_mean => 0.0,
                :weight_init_stdev => 1.0,
                :enabled_default => true
            ))

            g = Genome(1)
            configure_new!(g, temp_config)

            # Record initial nodes (should have output node 0)
            initial_node_ids = Set(keys(g.nodes))

            # Add multiple nodes via mutation
            rng = Random.MersenneTwister(100)
            for _ in 1:5
                success = NEAT.mutate_add_node!(g, temp_config, rng)
                if success
                    # Find the newly added node(s)
                    current_node_ids = Set(keys(g.nodes))
                    new_node_ids = setdiff(current_node_ids, initial_node_ids)

                    # Check that all newly added nodes have bias=0.0
                    for node_id in new_node_ids
                        node = g.nodes[node_id]
                        @test node.bias == 0.0
                    end

                    # Update initial_node_ids for next iteration
                    initial_node_ids = current_node_ids
                end
            end

            # Verify we actually added some nodes
            @test length(g.nodes) > 1
        end
    end

    @testset "75% Disable Rule" begin
        using NEAT: crossover

        @testset "One Parent Disabled" begin
            # Create two genes with same key, one disabled and one enabled
            gene1 = ConnectionGene((0, 1), 1)
            gene1.weight = 1.0
            gene1.enabled = false  # Disabled

            gene2 = ConnectionGene((0, 1), 1)
            gene2.weight = 1.0
            gene2.enabled = true   # Enabled

            # Run many trials to get statistical validation
            trials = 2000
            rng = MersenneTwister(42)
            disabled_count = 0
            for _ in 1:trials
                if !crossover(gene1, gene2, rng).enabled
                    disabled_count += 1
                end
            end

            disable_rate = disabled_count / trials

            # When one parent disabled: 75% chance offspring disabled
            # (rand() > 0.75 gives 25% chance enabled, 75% chance disabled)
            @test 0.72 < disable_rate < 0.78
            println("  One parent disabled: $(round(disable_rate * 100, digits=2))% disabled offspring (expected ~75%)")
        end

        @testset "Both Parents Disabled" begin
            gene1 = ConnectionGene((0, 1), 1)
            gene1.weight = 1.0
            gene1.enabled = false

            gene2 = ConnectionGene((0, 1), 1)
            gene2.weight = 2.0
            gene2.enabled = false

            trials = 1000
            rng = MersenneTwister(43)
            disabled_count = 0
            for _ in 1:trials
                if !crossover(gene1, gene2, rng).enabled
                    disabled_count += 1
                end
            end

            disable_rate = disabled_count / trials

            # Should be ~75% disabled (both parents disabled, then 75% rule applied)
            @test disable_rate > 0.72 && disable_rate < 0.78
            println("  Both parents disabled: $(round(disable_rate * 100, digits=2))% disabled offspring (expected ~75%)")
        end

        @testset "Both Parents Enabled" begin
            gene1 = ConnectionGene((0, 1), 1)
            gene1.weight = 1.0
            gene1.enabled = true

            gene2 = ConnectionGene((0, 1), 1)
            gene2.weight = 2.0
            gene2.enabled = true

            trials = 1000
            rng = MersenneTwister(44)
            disabled_count = 0
            for _ in 1:trials
                if !crossover(gene1, gene2, rng).enabled
                    disabled_count += 1
                end
            end

            disable_rate = disabled_count / trials

            # Should be 0% disabled (rule doesn't apply when both enabled)
            @test disable_rate == 0.0
            println("  Both parents enabled: $(round(disable_rate * 100, digits=2))% disabled offspring (expected 0%)")
        end

        @testset "Weight Inheritance Independence" begin
            # Verify that the disable rule doesn't affect weight inheritance
            gene1 = ConnectionGene((0, 1), 1)
            gene1.weight = 1.0
            gene1.enabled = false

            gene2 = ConnectionGene((0, 1), 1)
            gene2.weight = 2.0
            gene2.enabled = true

            trials = 1000
            rng = MersenneTwister(45)
            weight_1_count = 0
            for _ in 1:trials
                offspring = crossover(gene1, gene2, rng)
                if offspring.weight == 1.0
                    weight_1_count += 1
                end
            end

            weight_ratio = weight_1_count / trials

            # Weight should be inherited 50/50 from either parent
            @test 0.45 < weight_ratio < 0.55
            println("  Weight inheritance: $(round(weight_ratio * 100, digits=2))% from gene1 (expected ~50%)")
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
                NEAT.mutate_add_node!(genome, config.genome_config, Random.default_rng())
                NEAT.mutate_add_connection!(genome, config.genome_config, Random.default_rng())

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
                    NEAT.mutate_add_node!(genome2, config.genome_config, Random.default_rng())
                    genome2.fitness = 2.0

                    genome3 = Genome(3)
                    configure_new!(genome3, config.genome_config)
                    NEAT.mutate_add_node!(genome3, config.genome_config, Random.default_rng())
                    NEAT.mutate_add_node!(genome3, config.genome_config, Random.default_rng())
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

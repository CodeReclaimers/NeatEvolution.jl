using Test
using NeatEvolution
using Random

@testset "Gene Tests" begin

    @testset "NodeGene Construction and Copy" begin
        # Test default construction
        node = NodeGene(5)
        @test node.key == 5
        @test node.bias == 0.0
        @test node.response == 1.0
        @test node.activation == :sigmoid
        @test node.aggregation == :sum

        # Test copy
        node.bias = 1.5
        node.response = 2.0
        node.activation = :tanh
        node.aggregation = :max

        node_copy = copy(node)
        @test node_copy.key == node.key
        @test node_copy.bias == node.bias
        @test node_copy.response == node.response
        @test node_copy.activation == node.activation
        @test node_copy.aggregation == node.aggregation

        # Verify it's a deep copy
        node_copy.bias = 3.0
        @test node.bias == 1.5
        @test node_copy.bias == 3.0
    end

    @testset "NodeGene Initialization" begin
        # Load a config for testing
        config_path = joinpath(dirname(@__DIR__), "examples", "xor", "config.toml")
        config = load_config(config_path)

        # Test with deterministic RNG
        rng = MersenneTwister(42)
        node = NodeGene(10)
        NeatEvolution.init_attributes!(node, config.genome_config, rng)

        # Record initial values
        initial_bias = node.bias
        initial_response = node.response
        initial_activation = node.activation
        initial_aggregation = node.aggregation

        # Test determinism: same seed should give same result
        rng2 = MersenneTwister(42)
        node2 = NodeGene(10)
        NeatEvolution.init_attributes!(node2, config.genome_config, rng2)

        @test node2.bias == initial_bias
        @test node2.response == initial_response
        @test node2.activation == initial_activation
        @test node2.aggregation == initial_aggregation

        # Test that different seed gives different result (probabilistically)
        rng3 = MersenneTwister(999)
        node3 = NodeGene(10)
        NeatEvolution.init_attributes!(node3, config.genome_config, rng3)

        # At least one attribute should differ (very high probability)
        different = (node3.bias != initial_bias ||
                    node3.response != initial_response ||
                    node3.activation != initial_activation ||
                    node3.aggregation != initial_aggregation)
        @test different
    end

    @testset "NodeGene Mutation" begin
        config_path = joinpath(dirname(@__DIR__), "examples", "xor", "config.toml")
        config = load_config(config_path)

        # Test mutation changes values
        rng = MersenneTwister(123)
        node = NodeGene(15)
        NeatEvolution.init_attributes!(node, config.genome_config, rng)

        original_bias = node.bias
        original_response = node.response
        original_activation = node.activation
        original_aggregation = node.aggregation

        # Mutate multiple times to increase chance of change
        changed = false
        for _ in 1:20
            mutate!(node, config.genome_config, rng)
            if (node.bias != original_bias ||
                node.response != original_response ||
                node.activation != original_activation ||
                node.aggregation != original_aggregation)
                changed = true
                break
            end
        end
        @test changed  # At least one mutation should change something
    end

    @testset "NodeGene Distance" begin
        config_path = joinpath(dirname(@__DIR__), "examples", "xor", "config.toml")
        config = load_config(config_path)

        # Test identical nodes
        node1 = NodeGene(20)
        node1.bias = 1.0
        node1.response = 1.5
        node1.activation = :sigmoid
        node1.aggregation = :sum

        node2 = NodeGene(20)
        node2.bias = 1.0
        node2.response = 1.5
        node2.activation = :sigmoid
        node2.aggregation = :sum

        d = NeatEvolution.distance(node1, node2, config.genome_config)
        @test d == 0.0

        # Test with different bias (distance should be |1.0 - 2.0| * weight_coeff)
        node2.bias = 2.0
        d = NeatEvolution.distance(node1, node2, config.genome_config)
        expected = abs(1.0 - 2.0) * config.genome_config.compatibility_weight_coefficient
        @test d == expected

        # Reset and test with different response
        node2.bias = 1.0
        node2.response = 2.5
        d = NeatEvolution.distance(node1, node2, config.genome_config)
        expected = abs(1.5 - 2.5) * config.genome_config.compatibility_weight_coefficient
        @test d == expected

        # Test with different activation
        node2.response = 1.5
        node2.activation = :tanh
        d = NeatEvolution.distance(node1, node2, config.genome_config)
        expected = 1.0 * config.genome_config.compatibility_weight_coefficient
        @test d == expected

        # Test with different aggregation
        node2.activation = :sigmoid
        node2.aggregation = :max
        d = NeatEvolution.distance(node1, node2, config.genome_config)
        expected = 1.0 * config.genome_config.compatibility_weight_coefficient
        @test d == expected

        # Test with all differences
        node2.bias = 2.5
        node2.response = 3.0
        node2.activation = :relu
        node2.aggregation = :product
        d = NeatEvolution.distance(node1, node2, config.genome_config)
        expected = (abs(1.0 - 2.5) + abs(1.5 - 3.0) + 1.0 + 1.0) *
                   config.genome_config.compatibility_weight_coefficient
        @test d ≈ expected atol=1e-10
    end

    @testset "NodeGene Crossover" begin
        rng = MersenneTwister(456)

        node1 = NodeGene(25)
        node1.bias = 1.0
        node1.response = 1.5
        node1.activation = :sigmoid
        node1.aggregation = :sum

        node2 = NodeGene(25)
        node2.bias = 2.0
        node2.response = 2.5
        node2.activation = :tanh
        node2.aggregation = :max

        # Test that crossover preserves key
        offspring = NeatEvolution.crossover(node1, node2, rng)
        @test offspring.key == 25

        # Test that attributes come from one parent or the other
        trials = 1000
        bias_from_1 = 0
        response_from_1 = 0
        activation_from_1 = 0
        aggregation_from_1 = 0

        for _ in 1:trials
            child = NeatEvolution.crossover(node1, node2, rng)
            @test child.bias == 1.0 || child.bias == 2.0
            @test child.response == 1.5 || child.response == 2.5
            @test child.activation == :sigmoid || child.activation == :tanh
            @test child.aggregation == :sum || child.aggregation == :max

            bias_from_1 += (child.bias == 1.0)
            response_from_1 += (child.response == 1.5)
            activation_from_1 += (child.activation == :sigmoid)
            aggregation_from_1 += (child.aggregation == :sum)
        end

        # Each attribute should be inherited ~50% from each parent
        @test 0.4 < bias_from_1 / trials < 0.6
        @test 0.4 < response_from_1 / trials < 0.6
        @test 0.4 < activation_from_1 / trials < 0.6
        @test 0.4 < aggregation_from_1 / trials < 0.6
    end

    @testset "ConnectionGene Construction and Copy" begin
        # Test default construction
        conn = ConnectionGene((1, 2), 42)
        @test conn.key == (1, 2)
        @test conn.weight == 0.0
        @test conn.enabled == true
        @test conn.innovation == 42

        # Test copy
        conn.weight = 1.5
        conn.enabled = false

        conn_copy = copy(conn)
        @test conn_copy.key == conn.key
        @test conn_copy.weight == conn.weight
        @test conn_copy.enabled == conn.enabled
        @test conn_copy.innovation == conn.innovation

        # Verify it's a deep copy
        conn_copy.weight = 3.0
        @test conn.weight == 1.5
        @test conn_copy.weight == 3.0
    end

    @testset "ConnectionGene Initialization" begin
        config_path = joinpath(dirname(@__DIR__), "examples", "xor", "config.toml")
        config = load_config(config_path)

        # Test with deterministic RNG
        rng = MersenneTwister(789)
        conn = ConnectionGene((3, 4), 100)
        NeatEvolution.init_attributes!(conn, config.genome_config, rng)

        initial_weight = conn.weight
        initial_enabled = conn.enabled

        # Test determinism
        rng2 = MersenneTwister(789)
        conn2 = ConnectionGene((3, 4), 100)
        NeatEvolution.init_attributes!(conn2, config.genome_config, rng2)

        @test conn2.weight == initial_weight
        @test conn2.enabled == initial_enabled
    end

    @testset "ConnectionGene Mutation" begin
        config_path = joinpath(dirname(@__DIR__), "examples", "xor", "config.toml")
        config = load_config(config_path)

        rng = MersenneTwister(321)
        conn = ConnectionGene((5, 6), 200)
        NeatEvolution.init_attributes!(conn, config.genome_config, rng)

        original_weight = conn.weight
        original_enabled = conn.enabled

        # Mutate multiple times
        changed = false
        for _ in 1:20
            mutate!(conn, config.genome_config, rng)
            if conn.weight != original_weight || conn.enabled != original_enabled
                changed = true
                break
            end
        end
        @test changed
    end

    @testset "ConnectionGene Distance" begin
        config_path = joinpath(dirname(@__DIR__), "examples", "xor", "config.toml")
        config = load_config(config_path)

        # Test identical connections
        conn1 = ConnectionGene((7, 8), 300)
        conn1.weight = 1.0
        conn1.enabled = true

        conn2 = ConnectionGene((7, 8), 300)
        conn2.weight = 1.0
        conn2.enabled = true

        d = NeatEvolution.distance(conn1, conn2, config.genome_config)
        @test d == 0.0

        # Test with different weight
        conn2.weight = 2.0
        d = NeatEvolution.distance(conn1, conn2, config.genome_config)
        expected = abs(1.0 - 2.0) * config.genome_config.compatibility_weight_coefficient
        @test d == expected

        # Test with different enabled status
        conn2.weight = 1.0
        conn2.enabled = false
        d = NeatEvolution.distance(conn1, conn2, config.genome_config)
        expected = 1.0 * config.genome_config.compatibility_weight_coefficient
        @test d == expected

        # Test with both different
        conn2.weight = 3.5
        d = NeatEvolution.distance(conn1, conn2, config.genome_config)
        expected = (abs(1.0 - 3.5) + 1.0) * config.genome_config.compatibility_weight_coefficient
        @test d ≈ expected atol=1e-10
    end

    @testset "ConnectionGene Crossover - Basic Inheritance" begin
        rng = MersenneTwister(654)

        # Test crossover with both parents enabled
        conn1 = ConnectionGene((9, 10), 400)
        conn1.weight = 1.0
        conn1.enabled = true

        conn2 = ConnectionGene((9, 10), 400)
        conn2.weight = 2.0
        conn2.enabled = true

        offspring = NeatEvolution.crossover(conn1, conn2, rng)
        @test offspring.key == (9, 10)
        @test offspring.innovation == 400
        @test offspring.enabled == true  # Both enabled -> offspring enabled
        @test offspring.weight == 1.0 || offspring.weight == 2.0

        # Test weight inheritance ratio
        trials = 1000
        weight_from_1 = sum(1 for _ in 1:trials if NeatEvolution.crossover(conn1, conn2, rng).weight == 1.0)
        @test 0.4 < weight_from_1 / trials < 0.6
    end

    @testset "ConnectionGene Crossover - Innovation Preservation" begin
        rng = MersenneTwister(987)

        conn1 = ConnectionGene((11, 12), 500)
        conn1.weight = 1.5

        conn2 = ConnectionGene((11, 12), 500)
        conn2.weight = 2.5

        # Innovation number should be preserved from first parent
        offspring = NeatEvolution.crossover(conn1, conn2, rng)
        @test offspring.innovation == 500
    end
end

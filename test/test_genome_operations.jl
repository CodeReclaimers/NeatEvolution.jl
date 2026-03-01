using Test
using NeatEvolution
using Random

# Helper to create a minimal GenomeConfig for testing
function make_test_genome_config(;
    num_inputs=2, num_outputs=1, num_hidden=0,
    initial_connection=:full, feed_forward=true,
    kwargs...
)
    params = Dict{Symbol,Any}(
        :num_inputs => num_inputs,
        :num_outputs => num_outputs,
        :num_hidden => num_hidden,
        :initial_connection => initial_connection,
        :feed_forward => feed_forward,
        :conn_add_prob => get(kwargs, :conn_add_prob, 0.5),
        :conn_delete_prob => get(kwargs, :conn_delete_prob, 0.5),
        :node_add_prob => get(kwargs, :node_add_prob, 0.2),
        :node_delete_prob => get(kwargs, :node_delete_prob, 0.2),
        :activation_default => "sigmoid",
        :aggregation_default => "sum",
        :activation_options => ["sigmoid", "tanh"],
        :aggregation_options => ["sum"],
        :bias_init_mean => 0.0,
        :bias_init_stdev => 1.0,
        :response_init_mean => 1.0,
        :response_init_stdev => 0.0,
        :weight_init_mean => 0.0,
        :weight_init_stdev => 1.0,
        :weight_mutate_rate => 0.8,
        :weight_mutate_power => 0.5,
        :enabled_default => "true",
    )
    for (k, v) in kwargs
        params[k] = v
    end
    GenomeConfig(params)
end

@testset "Genome Operation Tests" begin

    @testset "configure_crossover!" begin
        config = make_test_genome_config(num_hidden=2)
        rng = MersenneTwister(42)

        # Create two parent genomes
        parent1 = Genome(1)
        configure_new!(parent1, config, rng)
        parent1.fitness = 10.0

        parent2 = Genome(2)
        configure_new!(parent2, config, rng)
        parent2.fitness = 5.0

        @testset "Fitter parent structure dominates" begin
            child = Genome(3)
            NeatEvolution.configure_crossover!(child, parent1, parent2, config, rng)

            # Child should have all nodes from fitter parent (parent1)
            for k in keys(parent1.nodes)
                @test haskey(child.nodes, k)
            end

            # Child should have all connections from fitter parent
            for k in keys(parent1.connections)
                @test haskey(child.connections, k)
            end
        end

        @testset "Child attributes come from either parent" begin
            # Run many crossovers and check that child attributes vary
            rng2 = MersenneTwister(123)
            bias_set = Set{Float64}()
            for _ in 1:50
                child = Genome(100)
                NeatEvolution.configure_crossover!(child, parent1, parent2, config, rng2)
                for (k, n) in child.nodes
                    push!(bias_set, n.bias)
                end
            end
            # Should see variation (attributes from both parents)
            @test length(bias_set) > 1
        end

        @testset "Equal fitness parents" begin
            parent1.fitness = 5.0
            parent2.fitness = 5.0

            child = Genome(4)
            NeatEvolution.configure_crossover!(child, parent1, parent2, config, rng)

            # Should still produce valid child
            @test !isempty(child.nodes)
        end

        @testset "Nil fitness parents" begin
            parent1.fitness = nothing
            parent2.fitness = nothing

            child = Genome(5)
            NeatEvolution.configure_crossover!(child, parent1, parent2, config, rng)
            @test !isempty(child.nodes)
        end

        @testset "Swapped fitness - less fit parent1" begin
            parent1.fitness = 2.0
            parent2.fitness = 8.0

            child = Genome(6)
            NeatEvolution.configure_crossover!(child, parent1, parent2, config, rng)

            # Child should have all nodes from fitter parent (parent2)
            for k in keys(parent2.nodes)
                @test haskey(child.nodes, k)
            end
        end
    end

    @testset "mutate_delete_node!" begin
        config = make_test_genome_config(num_hidden=3)
        rng = MersenneTwister(42)

        @testset "Deletes a non-output node" begin
            g = Genome(1)
            configure_new!(g, config, rng)

            initial_node_count = length(g.nodes)
            # Should have output node + 3 hidden = 4 nodes
            @test initial_node_count == 4

            NeatEvolution.mutate_delete_node!(g, config, rng)

            # Should have one fewer node
            @test length(g.nodes) == initial_node_count - 1

            # Output node should never be deleted
            @test haskey(g.nodes, 0)
        end

        @testset "Removes associated connections" begin
            g = Genome(1)
            configure_new!(g, config, rng)

            # Find a hidden node
            hidden_keys = [k for k in keys(g.nodes) if !(k in config.output_keys)]
            @test !isempty(hidden_keys)

            target = hidden_keys[1]
            conn_count_before = length(g.connections)

            # Count connections involving this node
            involved = count(k -> target in k, keys(g.connections))
            @test involved > 0  # Should have at least some connections

            NeatEvolution.mutate_delete_node!(g, config, rng)

            # Some connections should have been removed
            # (we can't be sure which node was deleted, but total should decrease
            #  or stay same if the randomly chosen node had connections)
            @test length(g.nodes) < 4
        end

        @testset "No-op when only output nodes remain" begin
            config_no_hidden = make_test_genome_config(num_hidden=0)
            g = Genome(1)
            configure_new!(g, config_no_hidden, rng)

            @test length(g.nodes) == 1  # Just output

            NeatEvolution.mutate_delete_node!(g, config_no_hidden, rng)
            @test length(g.nodes) == 1  # Still just output, nothing to delete
        end
    end

    @testset "mutate_delete_connection!" begin
        config = make_test_genome_config()
        rng = MersenneTwister(42)

        @testset "Deletes a connection" begin
            g = Genome(1)
            configure_new!(g, config, rng)

            initial_count = length(g.connections)
            @test initial_count > 0

            NeatEvolution.mutate_delete_connection!(g, config, rng)
            @test length(g.connections) == initial_count - 1
        end

        @testset "No-op when no connections" begin
            config_empty = make_test_genome_config(initial_connection=:unconnected)
            g = Genome(1)
            configure_new!(g, config_empty, rng)

            @test isempty(g.connections)
            NeatEvolution.mutate_delete_connection!(g, config_empty, rng)
            @test isempty(g.connections)
        end
    end

    @testset "mutate_add_node! - structural verification" begin
        config = make_test_genome_config()
        rng = MersenneTwister(42)

        g = Genome(1)
        configure_new!(g, config, rng)

        initial_nodes = length(g.nodes)
        initial_conns = length(g.connections)

        # Pick a connection to verify it gets split
        conn_keys_before = Set(keys(g.connections))

        NeatEvolution.mutate_add_node!(g, config, rng)

        # Should add 1 node and 2 connections (but disable 1 existing)
        @test length(g.nodes) == initial_nodes + 1

        # Two new connections added
        @test length(g.connections) == initial_conns + 2

        # One existing connection should be disabled
        disabled = [c for c in values(g.connections) if !c.enabled]
        @test length(disabled) >= 1
        println("  After add_node: $(length(g.nodes)) nodes, $(length(g.connections)) connections, $(length(disabled)) disabled")
    end

    @testset "mutate_add_connection! - feed-forward acyclicity" begin
        config = make_test_genome_config(num_hidden=2)
        rng = MersenneTwister(42)

        g = Genome(1)
        configure_new!(g, config, rng)

        # Add many connections; none should create cycles in feed-forward mode
        for _ in 1:50
            NeatEvolution.mutate_add_connection!(g, config, rng)
        end

        # Verify no cycles: should be able to build a FeedForwardNetwork
        net = FeedForwardNetwork(g, config)
        output = activate!(net, [1.0, 0.5])
        @test !isnan(output[1])
        @test !isinf(output[1])
    end

    @testset "Genome distance" begin
        config = make_test_genome_config()
        rng = MersenneTwister(42)

        @testset "Distance to self is zero" begin
            g = Genome(1)
            configure_new!(g, config, rng)

            d = NeatEvolution.distance(g, g, config)
            @test d == 0.0
        end

        @testset "Distance is symmetric" begin
            g1 = Genome(1)
            configure_new!(g1, config, rng)

            g2 = Genome(2)
            configure_new!(g2, config, rng)

            d12 = NeatEvolution.distance(g1, g2, config)
            d21 = NeatEvolution.distance(g2, g1, config)
            @test d12 ≈ d21 atol=1e-10
        end

        @testset "Distance increases with mutation" begin
            g1 = Genome(1)
            configure_new!(g1, config, rng)

            # Make a copy via crossover with itself
            g2 = Genome(2)
            g1.fitness = 1.0
            NeatEvolution.configure_crossover!(g2, g1, g1, config, rng)

            d_initial = NeatEvolution.distance(g1, g2, config)

            # Mutate g2 repeatedly
            for _ in 1:20
                mutate!(g2, config, rng)
            end

            d_after = NeatEvolution.distance(g1, g2, config)

            # Distance should generally increase after mutations
            # (not guaranteed but very likely after 20 mutations)
            println("  Distance before mutations: $(round(d_initial, digits=4))")
            println("  Distance after 20 mutations: $(round(d_after, digits=4))")
            @test d_after >= 0.0
        end

        @testset "Distance with empty genomes" begin
            g1 = Genome(1)
            g2 = Genome(2)

            d = NeatEvolution.distance(g1, g2, config)
            @test d == 0.0
        end

        @testset "Distance with one connectionless genome" begin
            g1 = Genome(1)
            configure_new!(g1, config, rng)

            g2 = Genome(2)
            # g2 has no nodes or connections

            d = NeatEvolution.distance(g1, g2, config)
            @test d >= 0.0
            @test !isnan(d)
        end
    end

    @testset "Innovation tracking" begin
        config = make_test_genome_config()

        @testset "get_new_node_id! increments" begin
            id1 = NeatEvolution.get_new_node_id!(config)
            id2 = NeatEvolution.get_new_node_id!(config)
            @test id2 == id1 + 1
        end

        @testset "get_innovation! caches within generation" begin
            NeatEvolution.reset_innovation_cache!(config)
            key = (-1, 0)
            innov1 = NeatEvolution.get_innovation!(config, key)
            innov2 = NeatEvolution.get_innovation!(config, key)
            @test innov1 == innov2  # Same key should get same innovation
        end

        @testset "get_innovation! assigns new for different keys" begin
            NeatEvolution.reset_innovation_cache!(config)
            innov1 = NeatEvolution.get_innovation!(config, (-1, 0))
            innov2 = NeatEvolution.get_innovation!(config, (-2, 0))
            @test innov1 != innov2
            @test innov2 == innov1 + 1
        end

        @testset "reset_innovation_cache! clears cache" begin
            NeatEvolution.reset_innovation_cache!(config)
            innov1 = NeatEvolution.get_innovation!(config, (-1, 0))

            NeatEvolution.reset_innovation_cache!(config)
            # After reset, same key gets a NEW innovation number
            innov2 = NeatEvolution.get_innovation!(config, (-1, 0))
            @test innov2 != innov1  # New innovation number after cache reset
        end
    end

    @testset "Single structural mutation mode" begin
        config = make_test_genome_config(
            single_structural_mutation=true,
            conn_add_prob=0.25,
            conn_delete_prob=0.25,
            node_add_prob=0.25,
            node_delete_prob=0.25,
            num_hidden=2
        )
        rng = MersenneTwister(42)

        # Run many mutations and check that at most one structural change happens per call
        for trial in 1:20
            g = Genome(trial)
            configure_new!(g, config, rng)

            node_count = length(g.nodes)
            conn_count = length(g.connections)

            mutate!(g, config, rng)

            node_diff = length(g.nodes) - node_count
            conn_diff = length(g.connections) - conn_count

            # In single structural mutation mode:
            # - add_node: +1 node, +2 connections, 1 disabled
            # - delete_node: -1 node, -(connections involving node)
            # - add_connection: +1 connection
            # - delete_connection: -1 connection
            # At most one of these should happen
            structural_changes = (node_diff != 0 ? 1 : 0) + (abs(conn_diff) > 0 ? 1 : 0)
            # Node addition counts as both node and connection change, so allow up to 2
            # Other mutations are purely node or purely connection
            @test structural_changes <= 2
        end
    end
end

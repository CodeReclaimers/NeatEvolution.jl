"""
Tests for CTRNN (Continuous-Time Recurrent Neural Network) evaluator.

Verifies construction, forward Euler integration, per-node time constants,
reset, set_node_value!, variable dt handling, and determinism.
"""

using Test
using NEAT
using NEAT: NodeGene, ConnectionGene, Genome, GenomeConfig,
            CTRNNNetwork, CTRNNNodeEval, advance!, reset!, set_node_value!,
            get_activation_function, get_aggregation_function

"""
Create a GenomeConfig suitable for CTRNN testing.
"""
function make_ctrnn_test_config(; num_inputs=1, num_outputs=1)
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
        :time_constant_init_mean => 0.01,
        :time_constant_init_stdev => 0.0,
        :time_constant_min_value => 0.001,
        :time_constant_max_value => 10.0,
        :time_constant_mutate_rate => 0.0,
        :time_constant_mutate_power => 0.0,
        :time_constant_replace_rate => 0.0,
    )
    return GenomeConfig(params)
end

"""
Helper: create a NodeGene with CTRNN fields set.
"""
function make_ctrnn_node(key::Int; bias=0.0, response=1.0, tau=0.01,
                         activation=:identity, aggregation=:sum)
    node = NodeGene(key)
    node.bias = bias
    node.response = response
    node.activation = activation
    node.aggregation = aggregation
    node.time_constant = tau
    return node
end

"""
Helper: create a ConnectionGene.
"""
function make_ctrnn_conn(from::Int, to::Int; weight=1.0, enabled=true, innovation=0)
    conn = ConnectionGene((from, to))
    conn.weight = weight
    conn.enabled = enabled
    conn.innovation = innovation
    return conn
end

@testset "CTRNNNetwork" begin

    @testset "Construction from genome" begin
        config = make_ctrnn_test_config()
        genome = Genome(1)
        genome.nodes[0] = make_ctrnn_node(0, bias=0.5, tau=0.02)
        genome.connections[(-1, 0)] = make_ctrnn_conn(-1, 0, weight=1.0)

        net = CTRNNNetwork(genome, config)

        @test length(net.input_nodes) == 1
        @test length(net.output_nodes) == 1
        @test haskey(net.node_evals, 0)
        @test net.node_evals[0].time_constant == 0.02
        @test net.node_evals[0].bias == 0.5
        @test net.node_evals[0].response == 1.0
        @test length(net.node_evals[0].links) == 1
        @test net.time_seconds == 0.0
        println("  Construction: node 0 has tau=$(net.node_evals[0].time_constant), 1 link")
    end

    @testset "Error on NaN time_constant" begin
        config = make_ctrnn_test_config()
        genome = Genome(1)
        # Use a plain NodeGene (time_constant is NaN)
        node = NodeGene(0)
        node.bias = 0.0; node.response = 1.0
        node.activation = :identity; node.aggregation = :sum
        genome.nodes[0] = node
        genome.connections[(-1, 0)] = make_ctrnn_conn(-1, 0)

        @test_throws ErrorException CTRNNNetwork(genome, config)
        println("  NaN time_constant correctly raises error")
    end

    @testset "Single Euler step: hand-computed" begin
        # Setup: input=1.0, weight=1.0, bias=0.0, response=1.0, tau=0.1
        # identity activation, sum aggregation
        # Initial y=0
        #
        # z = identity(0 + 1 * sum([1.0 * 1.0])) = 1.0
        # y_new = 0 + (0.01/0.1) * (-0 + 1.0) = 0.1
        config = make_ctrnn_test_config()
        genome = Genome(1)
        genome.nodes[0] = make_ctrnn_node(0, bias=0.0, response=1.0, tau=0.1)
        genome.connections[(-1, 0)] = make_ctrnn_conn(-1, 0, weight=1.0)

        net = CTRNNNetwork(genome, config)
        out = advance!(net, [1.0], 0.01, 0.01)  # one step of dt=0.01

        println("  Single step: output=$(out[1]), expected=0.1")
        @test out[1] ≈ 0.1 atol=1e-10
        @test net.time_seconds ≈ 0.01 atol=1e-15
    end

    @testset "Steady-state convergence" begin
        # With identity activation, bias=0, response=1, weight=1, input=x:
        # z = identity(0 + 1 * x) = x
        # Steady state: dy/dt = 0 → -y + x = 0 → y = x
        # Use small tau for fast convergence
        config = make_ctrnn_test_config()
        genome = Genome(1)
        genome.nodes[0] = make_ctrnn_node(0, bias=0.0, response=1.0, tau=0.01)
        genome.connections[(-1, 0)] = make_ctrnn_conn(-1, 0, weight=1.0)

        net = CTRNNNetwork(genome, config)

        target = 3.14
        # Run for 0.5 seconds (50x the time constant) — should converge
        out = advance!(net, [target], 0.5, 0.001)

        println("  Steady-state: target=$target, output=$(out[1])")
        @test out[1] ≈ target atol=1e-6
    end

    @testset "Per-node time constant effect" begin
        # Two output nodes: one with small tau (fast), one with large tau (slow)
        # Both receive the same input. After a fixed time, the fast node
        # should be much closer to steady state.
        config = make_ctrnn_test_config(num_inputs=1, num_outputs=2)
        genome = Genome(1)

        # Fast node (tau=0.01)
        genome.nodes[0] = make_ctrnn_node(0, bias=0.0, response=1.0, tau=0.01)
        genome.connections[(-1, 0)] = make_ctrnn_conn(-1, 0, weight=1.0, innovation=0)

        # Slow node (tau=1.0)
        genome.nodes[1] = make_ctrnn_node(1, bias=0.0, response=1.0, tau=1.0)
        genome.connections[(-1, 1)] = make_ctrnn_conn(-1, 1, weight=1.0, innovation=1)

        net = CTRNNNetwork(genome, config)

        # Advance for 0.1 seconds (10x fast tau, 0.1x slow tau)
        out = advance!(net, [1.0], 0.1, 0.001)

        fast_error = abs(out[1] - 1.0)
        slow_error = abs(out[2] - 1.0)

        println("  Per-node tau: fast(tau=0.01) error=$(fast_error), slow(tau=1.0) error=$(slow_error)")
        @test fast_error < 0.01   # Fast node nearly converged (Euler error ~dt/tau)
        @test slow_error > 0.05   # Slow node still far from steady state
        @test fast_error < slow_error
    end

    @testset "Reset restores initial state" begin
        config = make_ctrnn_test_config()
        genome = Genome(1)
        genome.nodes[0] = make_ctrnn_node(0, bias=0.0, response=1.0, tau=0.01)
        genome.connections[(-1, 0)] = make_ctrnn_conn(-1, 0, weight=1.0)

        net = CTRNNNetwork(genome, config)

        # Run to build state
        advance!(net, [1.0], 0.1, 0.001)
        @test net.time_seconds > 0.0

        # Reset
        reset!(net)
        @test net.time_seconds == 0.0
        @test net.active == 1

        # Fresh network for comparison
        net2 = CTRNNNetwork(genome, config)
        out1 = advance!(net, [1.0], 0.01, 0.01)
        out2 = advance!(net2, [1.0], 0.01, 0.01)

        println("  Reset: after reset output=$(out1[1]), fresh output=$(out2[1])")
        @test out1[1] ≈ out2[1] atol=1e-10
    end

    @testset "set_node_value! propagation" begin
        config = make_ctrnn_test_config()
        genome = Genome(1)
        genome.nodes[0] = make_ctrnn_node(0, bias=0.0, response=1.0, tau=0.1)
        genome.connections[(-1, 0)] = make_ctrnn_conn(-1, 0, weight=1.0)

        net = CTRNNNetwork(genome, config)

        # Set the output node to 5.0
        set_node_value!(net, 0, 5.0)
        @test net.values[1][0] == 5.0
        @test net.values[2][0] == 5.0

        # With input=0, z=0, so: y_new = 5.0 + (0.01/0.1)*(-5.0 + 0) = 5.0 - 0.5 = 4.5
        out = advance!(net, [0.0], 0.01, 0.01)
        println("  set_node_value!: initial=5.0, after decay with input=0: $(out[1])")
        @test out[1] ≈ 4.5 atol=1e-10
    end

    @testset "Variable dt: advance_time not evenly divisible by time_step" begin
        config = make_ctrnn_test_config()
        genome = Genome(1)
        genome.nodes[0] = make_ctrnn_node(0, bias=0.0, response=1.0, tau=0.05)
        genome.connections[(-1, 0)] = make_ctrnn_conn(-1, 0, weight=1.0)

        net = CTRNNNetwork(genome, config)

        # advance_time=0.025, time_step=0.01 → steps of 0.01, 0.01, 0.005
        advance!(net, [1.0], 0.025, 0.01)

        println("  Variable dt: time_seconds=$(net.time_seconds)")
        @test net.time_seconds ≈ 0.025 atol=1e-15
    end

    @testset "Determinism: two networks from same genome" begin
        config = make_ctrnn_test_config()
        genome = Genome(1)
        genome.nodes[0] = make_ctrnn_node(0, bias=0.1, response=1.0, tau=0.05)
        genome.connections[(-1, 0)] = make_ctrnn_conn(-1, 0, weight=0.7)

        net1 = CTRNNNetwork(genome, config)
        net2 = CTRNNNetwork(genome, config)

        for _ in 1:5
            out1 = advance!(net1, [1.0], 0.01, 0.005)
            out2 = advance!(net2, [1.0], 0.01, 0.005)
            @test out1[1] ≈ out2[1] atol=1e-15
        end
        println("  Determinism: two networks produced identical outputs for 5 advances")
    end

    @testset "Multi-node recurrent CTRNN" begin
        # Hidden node with self-connection feeding into output
        config = make_ctrnn_test_config(num_inputs=1, num_outputs=1)
        genome = Genome(1)
        genome.nodes[0] = make_ctrnn_node(0, bias=0.0, response=1.0, tau=0.01)
        genome.nodes[1] = make_ctrnn_node(1, bias=0.0, response=1.0, tau=0.02)
        genome.connections[(-1, 1)] = make_ctrnn_conn(-1, 1, weight=1.0, innovation=0)
        genome.connections[(1, 0)] = make_ctrnn_conn(1, 0, weight=1.0, innovation=1)
        genome.connections[(1, 1)] = make_ctrnn_conn(1, 1, weight=0.5, innovation=2)

        net = CTRNNNetwork(genome, config)

        # Run several steps, verify output changes over time
        out_prev = copy(advance!(net, [1.0], 0.01, 0.001))
        for _ in 1:5
            out_curr = advance!(net, [1.0], 0.01, 0.001)
            @test out_curr[1] != out_prev[1]  # State should evolve
            out_prev = copy(out_curr)
        end
        println("  Multi-node recurrent: output evolves over time, final=$(out_prev[1])")
    end

    @testset "Sigmoid activation CTRNN" begin
        # Verify non-identity activation works
        config = make_ctrnn_test_config()
        genome = Genome(1)
        node = make_ctrnn_node(0, bias=0.0, response=1.0, tau=0.01)
        node.activation = :sigmoid
        genome.nodes[0] = node
        genome.connections[(-1, 0)] = make_ctrnn_conn(-1, 0, weight=1.0)

        net = CTRNNNetwork(genome, config)

        # z = sigmoid(0 + 1 * 0) = sigmoid(0) = 0.5
        # y_new = 0 + (0.01/0.01)*(-0 + 0.5) = 0.5
        out = advance!(net, [0.0], 0.01, 0.01)
        println("  Sigmoid CTRNN: input=0, output=$(out[1]) (expected ~0.5)")
        @test out[1] ≈ 0.5 atol=1e-10
    end
end

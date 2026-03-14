"""
Tests for CTRNN (Continuous-Time Recurrent Neural Network) evaluator.

Verifies construction, forward Euler integration, per-node time constants,
reset, set_node_value!, variable dt handling, and determinism.
"""

using Test
using NeatEvolution
using NeatEvolution: NodeGene, ConnectionGene, Genome, GenomeConfig,
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

    @testset "Input count mismatch error" begin
        config = make_ctrnn_test_config(num_inputs=2, num_outputs=1)
        genome = Genome(1)
        genome.nodes[0] = make_ctrnn_node(0)
        genome.connections[(-2, 0)] = make_ctrnn_conn(-2, 0, weight=1.0, innovation=0)
        genome.connections[(-1, 0)] = make_ctrnn_conn(-1, 0, weight=1.0, innovation=1)

        net = CTRNNNetwork(genome, config)

        # Pass 1 input when 2 are expected
        @test_throws ErrorException advance!(net, [1.0], 0.01, 0.01)
        # Pass 3 inputs when 2 are expected
        @test_throws ErrorException advance!(net, [1.0, 2.0, 3.0], 0.01, 0.01)
        println("  Input mismatch: correctly throws for wrong input count")
    end

    @testset "Disabled connections excluded" begin
        # Genome A: two connections, one disabled
        config = make_ctrnn_test_config()
        genome_a = Genome(1)
        genome_a.nodes[0] = make_ctrnn_node(0, bias=0.0, response=1.0, tau=0.1)
        genome_a.connections[(-1, 0)] = make_ctrnn_conn(-1, 0, weight=1.0, innovation=0)

        # Add a second connection (self-loop) that is disabled
        disabled = make_ctrnn_conn(0, 0, weight=100.0, enabled=false, innovation=1)
        genome_a.connections[(0, 0)] = disabled

        # Genome B: only the enabled connection
        genome_b = Genome(2)
        genome_b.nodes[0] = make_ctrnn_node(0, bias=0.0, response=1.0, tau=0.1)
        genome_b.connections[(-1, 0)] = make_ctrnn_conn(-1, 0, weight=1.0, innovation=0)

        net_a = CTRNNNetwork(genome_a, config)
        net_b = CTRNNNetwork(genome_b, config)

        out_a = advance!(net_a, [1.0], 0.05, 0.001)
        out_b = advance!(net_b, [1.0], 0.05, 0.001)

        println("  Disabled conn: with_disabled=$(out_a[1]), without=$(out_b[1])")
        @test out_a[1] ≈ out_b[1] atol=1e-10
    end

    @testset "Multiple inputs: hand-computed" begin
        # num_inputs=2, num_outputs=1
        # input_keys = [-2, -1], output_keys = [0]
        # w1=0.5 (from -2), w2=2.0 (from -1), bias=0, response=1.0
        # x1=1.0, x2=0.5, tau=0.1, dt=0.01
        #
        # s = sum([0.5*1.0, 2.0*0.5]) = sum([0.5, 1.0]) = 1.5
        # z = identity(0 + 1.0 * 1.5) = 1.5
        # y_new = 0 + (0.01/0.1)*(-0 + 1.5) = 0.15
        config = make_ctrnn_test_config(num_inputs=2, num_outputs=1)
        genome = Genome(1)
        genome.nodes[0] = make_ctrnn_node(0, bias=0.0, response=1.0, tau=0.1)
        genome.connections[(-2, 0)] = make_ctrnn_conn(-2, 0, weight=0.5, innovation=0)
        genome.connections[(-1, 0)] = make_ctrnn_conn(-1, 0, weight=2.0, innovation=1)

        net = CTRNNNetwork(genome, config)
        out = advance!(net, [1.0, 0.5], 0.01, 0.01)

        println("  Multiple inputs: output=$(out[1]), expected=0.15")
        @test out[1] ≈ 0.15 atol=1e-10
    end

    @testset "Bias effect: hand-computed" begin
        # bias=2.0, input=0.0, weight=1.0, identity activation, tau=0.1, dt=0.01
        # s = sum([1.0 * 0.0]) = 0.0
        # z = identity(2.0 + 1.0 * 0.0) = 2.0
        # y_new = 0 + (0.01/0.1)*(-0 + 2.0) = 0.2
        config = make_ctrnn_test_config()
        genome = Genome(1)
        genome.nodes[0] = make_ctrnn_node(0, bias=2.0, response=1.0, tau=0.1)
        genome.connections[(-1, 0)] = make_ctrnn_conn(-1, 0, weight=1.0)

        net = CTRNNNetwork(genome, config)
        out = advance!(net, [0.0], 0.01, 0.01)

        println("  Bias effect: output=$(out[1]), expected=0.2")
        @test out[1] ≈ 0.2 atol=1e-10
    end

    @testset "Non-unity response: hand-computed" begin
        # response=0.5, bias=0, input=1.0, weight=1.0, tau=0.1, dt=0.01
        # s = sum([1.0 * 1.0]) = 1.0
        # z = identity(0 + 0.5 * 1.0) = 0.5
        # y_new = 0 + (0.01/0.1)*(-0 + 0.5) = 0.05
        config = make_ctrnn_test_config()
        genome = Genome(1)
        genome.nodes[0] = make_ctrnn_node(0, bias=0.0, response=0.5, tau=0.1)
        genome.connections[(-1, 0)] = make_ctrnn_conn(-1, 0, weight=1.0)

        net = CTRNNNetwork(genome, config)
        out = advance!(net, [1.0], 0.01, 0.01)

        println("  Non-unity response: output=$(out[1]), expected=0.05")
        @test out[1] ≈ 0.05 atol=1e-10
    end

    @testset "Negative (inhibitory) weight" begin
        # Two output nodes: one with w=+1.0, one with w=-1.0
        # Same input=1.0, bias=0, response=1.0, identity activation
        # After one step, outputs should be equal magnitude, opposite sign
        config = make_ctrnn_test_config(num_inputs=1, num_outputs=2)
        genome = Genome(1)
        genome.nodes[0] = make_ctrnn_node(0, bias=0.0, response=1.0, tau=0.1)
        genome.nodes[1] = make_ctrnn_node(1, bias=0.0, response=1.0, tau=0.1)
        genome.connections[(-1, 0)] = make_ctrnn_conn(-1, 0, weight=1.0, innovation=0)
        genome.connections[(-1, 1)] = make_ctrnn_conn(-1, 1, weight=-1.0, innovation=1)

        net = CTRNNNetwork(genome, config)
        out = advance!(net, [1.0], 0.01, 0.01)

        println("  Inhibitory weight: excitatory=$(out[1]), inhibitory=$(out[2])")
        @test out[1] > 0.0
        @test out[2] < 0.0
        @test out[1] ≈ -out[2] atol=1e-10
    end

    @testset "Disconnected node excluded from node_evals" begin
        # Output node 0 connected to input -1
        # Hidden node 1 with no path to output 0
        config = make_ctrnn_test_config(num_inputs=1, num_outputs=1)
        genome = Genome(1)
        genome.nodes[0] = make_ctrnn_node(0, bias=0.0, response=1.0, tau=0.1)
        genome.nodes[1] = make_ctrnn_node(1, bias=0.0, response=1.0, tau=0.1)
        genome.connections[(-1, 0)] = make_ctrnn_conn(-1, 0, weight=1.0, innovation=0)
        # Hidden node 1 gets input but doesn't feed into output
        genome.connections[(-1, 1)] = make_ctrnn_conn(-1, 1, weight=1.0, innovation=1)

        net = CTRNNNetwork(genome, config)

        @test haskey(net.node_evals, 0)
        @test !haskey(net.node_evals, 1)
        println("  Disconnected node: node 0 in evals=$(haskey(net.node_evals, 0)), node 1 in evals=$(haskey(net.node_evals, 1))")
    end

    @testset "Non-sum aggregation (product)" begin
        # Two inputs, aggregation=:product
        # w1=1.0, w2=1.0, x1=2.0, x2=3.0, bias=0, response=1.0, tau=0.1, dt=0.01
        # s = product([1.0*2.0, 1.0*3.0]) = product([2.0, 3.0]) = 6.0
        # z = identity(0 + 1.0 * 6.0) = 6.0
        # y_new = 0 + (0.01/0.1)*(-0 + 6.0) = 0.6
        config = make_ctrnn_test_config(num_inputs=2, num_outputs=1)
        genome = Genome(1)
        node = make_ctrnn_node(0, bias=0.0, response=1.0, tau=0.1)
        node.aggregation = :product
        genome.nodes[0] = node
        genome.connections[(-2, 0)] = make_ctrnn_conn(-2, 0, weight=1.0, innovation=0)
        genome.connections[(-1, 0)] = make_ctrnn_conn(-1, 0, weight=1.0, innovation=1)

        net = CTRNNNetwork(genome, config)
        out = advance!(net, [2.0, 3.0], 0.01, 0.01)

        println("  Product aggregation: output=$(out[1]), expected=0.6")
        @test out[1] ≈ 0.6 atol=1e-10
    end

    @testset "Convenience constructor (Config wrapper)" begin
        # Build a Config manually and verify the convenience constructor
        # produces the same network as using GenomeConfig directly
        gc = make_ctrnn_test_config(num_inputs=1, num_outputs=1)
        sc = SpeciesConfig(Dict{Symbol,Any}())
        stc = StagnationConfig(Dict{Symbol,Any}())
        rc = ReproductionConfig(Dict{Symbol,Any}())
        cfg = Config(Dict{Symbol,Any}(), gc, sc, stc, rc)

        genome = Genome(1)
        genome.nodes[0] = make_ctrnn_node(0, bias=0.5, response=1.0, tau=0.05)
        genome.connections[(-1, 0)] = make_ctrnn_conn(-1, 0, weight=0.7)

        net_gc = CTRNNNetwork(genome, gc)
        net_cfg = CTRNNNetwork(genome, cfg)

        # Same structure
        @test net_gc.input_nodes == net_cfg.input_nodes
        @test net_gc.output_nodes == net_cfg.output_nodes
        @test length(net_gc.node_evals) == length(net_cfg.node_evals)

        # Same outputs after advancing
        out_gc = advance!(net_gc, [1.0], 0.05, 0.001)
        out_cfg = advance!(net_cfg, [1.0], 0.05, 0.001)

        println("  Config wrapper: GenomeConfig output=$(out_gc[1]), Config output=$(out_cfg[1])")
        @test out_gc[1] ≈ out_cfg[1] atol=1e-15
    end

    @testset "AbstractNetwork interface" begin
        config = make_ctrnn_test_config(num_inputs=2, num_outputs=3)
        genome = Genome(1)
        # Output nodes 0, 1, 2 for num_outputs=3
        for i in 0:2
            genome.nodes[i] = make_ctrnn_node(i, tau=0.01)
            genome.connections[(-1, i)] = make_ctrnn_conn(-1, i, weight=1.0, innovation=i)
        end

        net = CTRNNNetwork(genome, config)

        @test net isa AbstractNetwork
        @test input_nodes(net) == [-2, -1]
        @test output_nodes(net) == [0, 1, 2]
        @test num_inputs(net) == 2
        @test num_outputs(net) == 3
        println("  AbstractNetwork: inputs=$(input_nodes(net)), outputs=$(output_nodes(net)), num_in=$(num_inputs(net)), num_out=$(num_outputs(net))")
    end

    @testset "Analytical solution comparison" begin
        # For identity activation, bias=0, response=1, weight=1, constant input x:
        # The continuous ODE is: tau * dy/dt = -y + x
        # Exact continuous solution: y(t) = x * (1 - exp(-t/tau))
        #
        # With small dt, the forward Euler discretization error relative to
        # this formula is O(dt), validating integration accuracy.
        tau = 0.1
        x_input = 2.5
        dt = 0.0001  # small step for accuracy

        config = make_ctrnn_test_config()
        genome = Genome(1)
        genome.nodes[0] = make_ctrnn_node(0, bias=0.0, response=1.0, tau=tau)
        genome.connections[(-1, 0)] = make_ctrnn_conn(-1, 0, weight=1.0)

        net = CTRNNNetwork(genome, config)

        # Check at several time points
        check_times = [0.01, 0.05, 0.1, 0.2, 0.5]
        prev_time = 0.0

        for t in check_times
            advance_by = t - prev_time
            advance!(net, [x_input], advance_by, dt)
            prev_time = t

            y_analytical = x_input * (1.0 - exp(-t / tau))
            y_euler = net.values[net.active][0]

            println("  Analytical: t=$t, expected=$y_analytical, euler=$y_euler, err=$(abs(y_euler - y_analytical))")
            @test y_euler ≈ y_analytical atol=1e-3
        end
    end

    @testset "Time accumulation across multiple advance! calls" begin
        config = make_ctrnn_test_config()
        genome = Genome(1)
        genome.nodes[0] = make_ctrnn_node(0, tau=0.01)
        genome.connections[(-1, 0)] = make_ctrnn_conn(-1, 0, weight=1.0)

        net = CTRNNNetwork(genome, config)

        t1, t2, t3 = 0.013, 0.027, 0.041
        advance!(net, [1.0], t1, 0.001)
        advance!(net, [1.0], t2, 0.001)
        advance!(net, [1.0], t3, 0.001)

        expected = t1 + t2 + t3
        println("  Time accumulation: net.time=$(net.time_seconds), expected=$expected")
        @test net.time_seconds ≈ expected atol=1e-12
    end

    @testset "Zero-link output node" begin
        # Output node with no incoming connections
        # s = sum([]) = 0.0
        # z = identity(bias + response * 0.0) = bias
        # Steady state: y = bias
        bias = 1.7
        tau = 0.01

        config = make_ctrnn_test_config()
        genome = Genome(1)
        genome.nodes[0] = make_ctrnn_node(0, bias=bias, response=1.0, tau=tau)
        # No connections — output node is always required

        net = CTRNNNetwork(genome, config)

        @test haskey(net.node_evals, 0)
        @test length(net.node_evals[0].links) == 0

        # Run long enough to converge (50x time constant)
        out = advance!(net, [0.0], 0.5, 0.001)

        println("  Zero-link node: bias=$bias, output=$(out[1]) (should converge to bias)")
        @test out[1] ≈ bias atol=1e-6
    end
end

"""
Tests for RecurrentNetwork evaluator.

These tests construct genomes by hand with known topology and weights,
then verify that RecurrentNetwork produces the expected outputs across
multiple timesteps. This exercises:
  - Self-connections (simplest recurrence)
  - Cycles (A→B→A)
  - State persistence across activate! calls
  - reset! restoring initial behavior
  - Equivalence with FeedForwardNetwork for acyclic genomes
  - Reproducibility across independent network instances
"""

using NeatEvolution: NodeGene, ConnectionGene, Genome, GenomeConfig, NodeEval,
            FeedForwardNetwork, RecurrentNetwork, activate!, reset!,
            get_activation_function, get_aggregation_function,
            sigmoid_activation, identity_activation, tanh_activation,
            sum_aggregation

"""
Create a minimal GenomeConfig for testing with given num_inputs and num_outputs.
"""
function make_test_config(; num_inputs=1, num_outputs=1)
    params = Dict{Symbol, Any}(
        :num_inputs => num_inputs,
        :num_outputs => num_outputs,
        :num_hidden => 0,
        :feed_forward => false,  # recurrent
        :initial_connection => :unconnected,
        :activation_default => "identity",
        :activation_options => ["identity"],
        :activation_mutate_rate => 0.0,
        :aggregation_default => "sum",
        :aggregation_options => ["sum"],
        :aggregation_mutate_rate => 0.0,
    )
    return GenomeConfig(params)
end

"""
Helper to create a NodeGene with specified activation and aggregation.
"""
function make_node(key::Int; bias=0.0, response=1.0,
                   activation=:identity, aggregation=:sum)
    node = NodeGene(key)
    node.bias = bias
    node.response = response
    node.activation = activation
    node.aggregation = aggregation
    return node
end

"""
Helper to create a ConnectionGene.
"""
function make_conn(from::Int, to::Int; weight=1.0, enabled=true, innovation=0)
    conn = ConnectionGene((from, to))
    conn.weight = weight
    conn.enabled = enabled
    conn.innovation = innovation
    return conn
end

@testset "RecurrentNetwork" begin
    @testset "Construction from genome with self-connection" begin
        # Genome: input(-1) → output(0) with self-connection on output
        config = make_test_config(num_inputs=1, num_outputs=1)
        genome = Genome(1)
        genome.nodes[0] = make_node(0, bias=0.0, response=1.0)
        genome.connections[(-1, 0)] = make_conn(-1, 0, weight=1.0, innovation=0)
        genome.connections[(0, 0)] = make_conn(0, 0, weight=0.5, innovation=1)

        net = RecurrentNetwork(genome, config)

        @test length(net.input_nodes) == 1
        @test length(net.output_nodes) == 1
        @test length(net.node_evals) == 1  # just the output node
        # Node eval should have 2 inputs: from input and from self
        _, _, _, _, _, links = net.node_evals[1]
        @test length(links) == 2
        println("  Construction: output node has $(length(links)) inputs (input + self-connection)")
    end

    @testset "State persistence: same inputs produce different outputs" begin
        # With identity activation, sum aggregation, bias=0, response=1:
        # output = sum(weighted_inputs_from_prev_values)
        # Self-connection weight = 0.5, input weight = 1.0
        #
        # Timestep 1: prev_values all 0, input=1.0
        #   output = identity(0 + 1*(1.0*1.0 + 0.5*0.0)) = 1.0
        # Timestep 2: prev_values[output]=1.0, input=1.0
        #   output = identity(0 + 1*(1.0*1.0 + 0.5*1.0)) = 1.5
        # Timestep 3: prev_values[output]=1.5, input=1.0
        #   output = identity(0 + 1*(1.0*1.0 + 0.5*1.5)) = 1.75
        config = make_test_config(num_inputs=1, num_outputs=1)
        genome = Genome(1)
        genome.nodes[0] = make_node(0, bias=0.0, response=1.0)
        genome.connections[(-1, 0)] = make_conn(-1, 0, weight=1.0, innovation=0)
        genome.connections[(0, 0)] = make_conn(0, 0, weight=0.5, innovation=1)

        net = RecurrentNetwork(genome, config)

        out1 = copy(activate!(net, [1.0]))
        out2 = copy(activate!(net, [1.0]))
        out3 = copy(activate!(net, [1.0]))

        println("  State persistence: t1=$(out1[1]), t2=$(out2[1]), t3=$(out3[1])")
        @test out1[1] ≈ 1.0 atol=1e-10
        @test out2[1] ≈ 1.5 atol=1e-10
        @test out3[1] ≈ 1.75 atol=1e-10

        # Outputs must differ (proves recurrence)
        @test out1[1] != out2[1]
        @test out2[1] != out3[1]
    end

    @testset "reset! restores initial behavior" begin
        config = make_test_config(num_inputs=1, num_outputs=1)
        genome = Genome(1)
        genome.nodes[0] = make_node(0, bias=0.0, response=1.0)
        genome.connections[(-1, 0)] = make_conn(-1, 0, weight=1.0, innovation=0)
        genome.connections[(0, 0)] = make_conn(0, 0, weight=0.5, innovation=1)

        net = RecurrentNetwork(genome, config)

        # Run a few timesteps to build state
        out_before_1 = copy(activate!(net, [1.0]))
        activate!(net, [1.0])
        activate!(net, [1.0])

        # Reset and verify first output matches
        reset!(net)
        out_after_1 = activate!(net, [1.0])

        println("  Reset: before=$(out_before_1[1]), after=$(out_after_1[1])")
        @test out_before_1[1] ≈ out_after_1[1] atol=1e-10
    end

    @testset "Self-connection: hand-computed 3 timesteps with bias" begin
        # Node with bias=0.5, response=1.0, identity activation, sum aggregation
        # Self-connection weight=0.3, input weight=2.0
        #
        # output(t) = identity(0.5 + 1.0 * (2.0*input + 0.3*output(t-1)))
        # output(0) = 0 (initial)
        #
        # t=1: output = 0.5 + 1*(2.0*1.0 + 0.3*0.0) = 2.5
        # t=2: output = 0.5 + 1*(2.0*1.0 + 0.3*2.5) = 3.25
        # t=3: output = 0.5 + 1*(2.0*1.0 + 0.3*3.25) = 3.475
        config = make_test_config(num_inputs=1, num_outputs=1)
        genome = Genome(1)
        genome.nodes[0] = make_node(0, bias=0.5, response=1.0)
        genome.connections[(-1, 0)] = make_conn(-1, 0, weight=2.0, innovation=0)
        genome.connections[(0, 0)] = make_conn(0, 0, weight=0.3, innovation=1)

        net = RecurrentNetwork(genome, config)

        out1 = copy(activate!(net, [1.0]))
        out2 = copy(activate!(net, [1.0]))
        out3 = copy(activate!(net, [1.0]))

        println("  Self-connection with bias: t1=$(out1[1]), t2=$(out2[1]), t3=$(out3[1])")
        @test out1[1] ≈ 2.5 atol=1e-10
        @test out2[1] ≈ 3.25 atol=1e-10
        @test out3[1] ≈ 3.475 atol=1e-10
    end

    @testset "Cycle A→B→A: hand-computed expected values" begin
        # Topology: input(-1) → hidden(1) → output(0) → hidden(1) (cycle)
        # All identity activation, sum aggregation, bias=0, response=1
        #
        # Connections:
        #   (-1, 1) weight=1.0  (input to hidden)
        #   (1, 0)  weight=1.0  (hidden to output)
        #   (0, 1)  weight=0.5  (output to hidden -- creates cycle)
        #
        # Node eval order (sorted by ID): node 0, then node 1
        # But node 0 reads from prev_values, and node 1 reads from prev_values
        # Since nodes are sorted: [0, 1]
        # Node 0 inputs: from node 1 (weight 1.0)
        # Node 1 inputs: from node -1 (weight 1.0), from node 0 (weight 0.5)
        #
        # t=1: all prev_values=0, input=1.0
        #   node 0: identity(0 + 1*(1.0*prev[1])) = identity(0 + 1*0) = 0.0
        #   node 1: identity(0 + 1*(1.0*prev[-1] + 0.5*prev[0])) = identity(0 + 1*(1.0 + 0)) = 1.0
        #   values = {-1:1.0, 0:0.0, 1:1.0}
        #   output = values[0] = 0.0
        #
        # t=2: prev = {-1:1.0, 0:0.0, 1:1.0}, input=1.0
        #   node 0: identity(0 + 1*(1.0*prev[1])) = identity(1.0) = 1.0
        #   node 1: identity(0 + 1*(1.0*prev[-1] + 0.5*prev[0])) = identity(1.0 + 0) = 1.0
        #   values = {-1:1.0, 0:1.0, 1:1.0}
        #   output = 1.0
        #
        # t=3: prev = {-1:1.0, 0:1.0, 1:1.0}, input=1.0
        #   node 0: identity(0 + 1*(1.0*prev[1])) = 1.0
        #   node 1: identity(0 + 1*(1.0*prev[-1] + 0.5*prev[0])) = identity(1.0 + 0.5) = 1.5
        #   values = {-1:1.0, 0:1.0, 1:1.5}
        #   output = 1.0
        #
        # t=4: prev = {-1:1.0, 0:1.0, 1:1.5}, input=1.0
        #   node 0: identity(1.0*1.5) = 1.5
        #   node 1: identity(1.0*1.0 + 0.5*1.0) = 1.5
        #   output = 1.5
        config = make_test_config(num_inputs=1, num_outputs=1)
        genome = Genome(1)
        genome.nodes[0] = make_node(0, bias=0.0, response=1.0)
        genome.nodes[1] = make_node(1, bias=0.0, response=1.0)
        genome.connections[(-1, 1)] = make_conn(-1, 1, weight=1.0, innovation=0)
        genome.connections[(1, 0)] = make_conn(1, 0, weight=1.0, innovation=1)
        genome.connections[(0, 1)] = make_conn(0, 1, weight=0.5, innovation=2)

        net = RecurrentNetwork(genome, config)

        out1 = copy(activate!(net, [1.0]))
        out2 = copy(activate!(net, [1.0]))
        out3 = copy(activate!(net, [1.0]))
        out4 = copy(activate!(net, [1.0]))

        println("  Cycle A→B→A: t1=$(out1[1]), t2=$(out2[1]), t3=$(out3[1]), t4=$(out4[1])")
        @test out1[1] ≈ 0.0 atol=1e-10
        @test out2[1] ≈ 1.0 atol=1e-10
        @test out3[1] ≈ 1.0 atol=1e-10
        @test out4[1] ≈ 1.5 atol=1e-10
    end

    @testset "Equivalence with FeedForwardNetwork for acyclic genome" begin
        # Simple acyclic: input(-1) → output(0), no cycles
        # Both networks should produce same result on first activation
        # (when RecurrentNetwork prev_values are all zero)
        config = make_test_config(num_inputs=1, num_outputs=1)
        genome = Genome(1)
        genome.nodes[0] = make_node(0, bias=0.5, response=1.0)
        genome.connections[(-1, 0)] = make_conn(-1, 0, weight=2.0, innovation=0)

        ff_net = FeedForwardNetwork(genome, config)
        rnn_net = RecurrentNetwork(genome, config)

        test_inputs = [0.0, 0.5, 1.0, -1.0, 3.14]
        for x in test_inputs
            # Reset RNN before each test to ensure prev_values are zero
            reset!(rnn_net)
            ff_out = activate!(ff_net, [x])
            rnn_out = activate!(rnn_net, [x])
            println("  Equivalence: input=$x, FF=$(ff_out[1]), RNN=$(rnn_out[1])")
            @test ff_out[1] ≈ rnn_out[1] atol=1e-10
        end
    end

    @testset "Multi-layer convergence to feed-forward on second step" begin
        # For a multi-layer acyclic graph, RecurrentNetwork reads from prev_values
        # (all zeros on first call), so hidden layer output doesn't propagate to
        # the output layer within the same timestep. But on the second call with
        # the same inputs, the hidden layer values from step 1 are available.
        #
        # input(-1,-2) → hidden(1) → output(0)
        # After two calls with same inputs, RNN output should match FF output.
        config = make_test_config(num_inputs=2, num_outputs=1)
        genome = Genome(1)
        genome.nodes[0] = make_node(0, bias=0.1, response=1.0)
        genome.nodes[1] = make_node(1, bias=-0.2, response=1.0)
        genome.connections[(-1, 1)] = make_conn(-1, 1, weight=0.5, innovation=0)
        genome.connections[(-2, 1)] = make_conn(-2, 1, weight=-0.3, innovation=1)
        genome.connections[(1, 0)] = make_conn(1, 0, weight=1.5, innovation=2)

        ff_net = FeedForwardNetwork(genome, config)
        rnn_net = RecurrentNetwork(genome, config)

        ff_out = copy(activate!(ff_net, [1.0, 2.0]))
        rnn_out1 = copy(activate!(rnn_net, [1.0, 2.0]))  # hidden=computed, output reads prev hidden=0
        rnn_out2 = activate!(rnn_net, [1.0, 2.0])  # output reads prev hidden (now correct)

        println("  Multi-layer: FF=$(ff_out[1]), RNN step1=$(rnn_out1[1]), RNN step2=$(rnn_out2[1])")
        # On step 1, output differs (reads zero prev_values for hidden)
        @test rnn_out1[1] != ff_out[1]
        # On step 2, output matches FF (hidden values propagated)
        @test rnn_out2[1] ≈ ff_out[1] atol=1e-10
    end

    @testset "Reproducibility: two networks from same genome" begin
        config = make_test_config(num_inputs=1, num_outputs=1)
        genome = Genome(1)
        genome.nodes[0] = make_node(0, bias=0.0, response=1.0)
        genome.connections[(-1, 0)] = make_conn(-1, 0, weight=1.0, innovation=0)
        genome.connections[(0, 0)] = make_conn(0, 0, weight=0.5, innovation=1)

        net1 = RecurrentNetwork(genome, config)
        net2 = RecurrentNetwork(genome, config)

        for t in 1:5
            out1 = activate!(net1, [1.0])
            out2 = activate!(net2, [1.0])
            @test out1[1] ≈ out2[1] atol=1e-10
        end
        println("  Reproducibility: two independent networks produced identical outputs for 5 timesteps")
    end

    @testset "Multiple inputs and outputs" begin
        # 2 inputs, 2 outputs, with self-connections
        # input_keys = [-2, -1], output_keys = [0, 1]
        # inputs[1]=1.0 → node -2, inputs[2]=2.0 → node -1
        config = make_test_config(num_inputs=2, num_outputs=2)
        genome = Genome(1)
        genome.nodes[0] = make_node(0, bias=0.0, response=1.0)
        genome.nodes[1] = make_node(1, bias=0.0, response=1.0)
        # Connection: node -1 (value=2.0) → output 0
        genome.connections[(-1, 0)] = make_conn(-1, 0, weight=1.0, innovation=0)
        # Connection: node -2 (value=1.0) → output 1
        genome.connections[(-2, 1)] = make_conn(-2, 1, weight=1.0, innovation=1)
        # Self-connections
        genome.connections[(0, 0)] = make_conn(0, 0, weight=0.1, innovation=2)
        genome.connections[(1, 1)] = make_conn(1, 1, weight=0.1, innovation=3)

        net = RecurrentNetwork(genome, config)
        out = activate!(net, [1.0, 2.0])
        @test length(out) == 2
        println("  Multi-IO: inputs=[1.0, 2.0] (node -2=1.0, node -1=2.0), outputs=$out")

        # t=1: node -2=1.0, node -1=2.0
        # output 0 = 1.0*prev[-1] + 0.1*prev[0] = 1.0*2.0 + 0.1*0 = 2.0
        # output 1 = 1.0*prev[-2] + 0.1*prev[1] = 1.0*1.0 + 0.1*0 = 1.0
        @test out[1] ≈ 2.0 atol=1e-10
        @test out[2] ≈ 1.0 atol=1e-10
    end
end

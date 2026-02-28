"""
Tests for Izhikevich spiking neural network evaluator.

Verifies initial state, spiking behavior, overflow handling, network
construction from genome, recurrent spike propagation, reset, and
spike timing.
"""

using Test
using NeatEvolution
using NeatEvolution: NodeGene, ConnectionGene, Genome, GenomeConfig,
            IZNNNetwork, IZNeuron, advance!, reset!, set_inputs!,
            IZ_REGULAR_SPIKING, IZ_FAST_SPIKING

"""
Create a GenomeConfig suitable for IZNN testing.
"""
function make_iznn_test_config(; num_inputs=1, num_outputs=1)
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
        # Izhikevich parameters (regular spiking)
        :iz_a_init_mean => 0.02, :iz_a_init_stdev => 0.0,
        :iz_a_min_value => 0.001, :iz_a_max_value => 0.2,
        :iz_a_mutate_rate => 0.0, :iz_a_mutate_power => 0.0,
        :iz_a_replace_rate => 0.0,
        :iz_b_init_mean => 0.20, :iz_b_init_stdev => 0.0,
        :iz_b_min_value => 0.01, :iz_b_max_value => 0.3,
        :iz_b_mutate_rate => 0.0, :iz_b_mutate_power => 0.0,
        :iz_b_replace_rate => 0.0,
        :iz_c_init_mean => -65.0, :iz_c_init_stdev => 0.0,
        :iz_c_min_value => -80.0, :iz_c_max_value => -40.0,
        :iz_c_mutate_rate => 0.0, :iz_c_mutate_power => 0.0,
        :iz_c_replace_rate => 0.0,
        :iz_d_init_mean => 8.0, :iz_d_init_stdev => 0.0,
        :iz_d_min_value => 0.05, :iz_d_max_value => 10.0,
        :iz_d_mutate_rate => 0.0, :iz_d_mutate_power => 0.0,
        :iz_d_replace_rate => 0.0,
    )
    return GenomeConfig(params)
end

"""
Helper: create a NodeGene with Izhikevich fields set.
"""
function make_iz_node(key::Int; bias=0.0, a=0.02, b=0.20, c=-65.0, d=8.0,
                      activation=:identity, aggregation=:sum)
    node = NodeGene(key)
    node.bias = bias
    node.response = 1.0
    node.activation = activation
    node.aggregation = aggregation
    node.iz_a = a
    node.iz_b = b
    node.iz_c = c
    node.iz_d = d
    return node
end

"""
Helper: create a ConnectionGene.
"""
function make_iz_conn(from::Int, to::Int; weight=1.0, enabled=true, innovation=0)
    conn = ConnectionGene((from, to))
    conn.weight = weight
    conn.enabled = enabled
    conn.innovation = innovation
    return conn
end

@testset "IZNNNetwork" begin

    @testset "IZNeuron initial state" begin
        # Regular spiking: a=0.02, b=0.20, c=-65, d=8
        neuron = IZNeuron(0.0, 0.02, 0.20, -65.0, 8.0, Tuple{Int,Float64}[])
        @test neuron.v == -65.0  # v = c
        @test neuron.u == 0.20 * -65.0  # u = b*c = -13.0
        @test neuron.fired == 0.0
        @test neuron.current == 0.0  # bias=0
        println("  Initial state: v=$(neuron.v), u=$(neuron.u), fired=$(neuron.fired)")
    end

    @testset "Single neuron spike with sufficient current" begin
        # Regular spiking neuron with strong constant current
        neuron = IZNeuron(0.0, 0.02, 0.20, -65.0, 8.0, Tuple{Int,Float64}[])
        neuron.current = 20.0  # Strong input current

        # Step until spike occurs
        spike_step = nothing
        for step in 1:100
            advance!(neuron, 1.0)  # 1ms steps
            if neuron.fired == 1.0
                spike_step = step
                break
            end
        end

        @test spike_step !== nothing
        println("  Spike at step $spike_step")

        # After spike: v should be reset to c, u should have increased by d
        @test neuron.v == -65.0  # reset to c
        # u was incremented by d at spike time
    end

    @testset "No spike with insufficient current" begin
        # Low current that doesn't trigger a spike
        neuron = IZNeuron(0.0, 0.02, 0.20, -65.0, 8.0, Tuple{Int,Float64}[])
        neuron.current = 0.0  # No input

        for _ in 1:100
            advance!(neuron, 1.0)
        end

        @test neuron.fired == 0.0
        println("  No spike with zero current: v=$(neuron.v)")
    end

    @testset "Preset behavior: regular spiking spike train" begin
        # Regular spiking params with fixed current should produce spike train
        p = IZ_REGULAR_SPIKING
        neuron = IZNeuron(0.0, p.iz_a, p.iz_b, p.iz_c, p.iz_d, Tuple{Int,Float64}[])
        neuron.current = 15.0

        spike_count = 0
        for _ in 1:100  # 100ms
            advance!(neuron, 1.0)
            if neuron.fired == 1.0
                spike_count += 1
            end
        end

        @test spike_count > 0
        @test spike_count < 50  # Regular spiking, not insanely fast
        println("  Regular spiking: $spike_count spikes in 100ms")
    end

    @testset "Fast spiking vs regular spiking rate" begin
        # Fast spiking should produce more spikes than regular spiking
        p_fast = IZ_FAST_SPIKING
        p_reg = IZ_REGULAR_SPIKING
        current = 15.0

        n_fast = IZNeuron(0.0, p_fast.iz_a, p_fast.iz_b, p_fast.iz_c, p_fast.iz_d, Tuple{Int,Float64}[])
        n_reg = IZNeuron(0.0, p_reg.iz_a, p_reg.iz_b, p_reg.iz_c, p_reg.iz_d, Tuple{Int,Float64}[])
        n_fast.current = current
        n_reg.current = current

        fast_spikes = 0
        reg_spikes = 0
        for _ in 1:200
            advance!(n_fast, 1.0)
            advance!(n_reg, 1.0)
            fast_spikes += Int(n_fast.fired == 1.0)
            reg_spikes += Int(n_reg.fired == 1.0)
        end

        println("  Fast spiking: $fast_spikes spikes, Regular: $reg_spikes spikes in 200ms")
        @test fast_spikes > reg_spikes
    end

    @testset "Overflow handling" begin
        # Test with truly extreme current that produces NaN/Inf in half-step
        neuron = IZNeuron(0.0, 0.02, 0.20, -65.0, 8.0, Tuple{Int,Float64}[])
        neuron.current = 1e200  # Extreme enough to produce Inf

        # Should not throw — guard catches NaN/Inf and resets to (c, b*c)
        advance!(neuron, 1.0)

        @test isfinite(neuron.v)
        @test isfinite(neuron.u)
        @test neuron.v == -65.0  # Reset to c
        @test neuron.u == 0.20 * -65.0  # Reset to b*c
        @test neuron.fired == 0.0  # No spike emitted on overflow
        println("  Overflow handling: v=$(neuron.v), u=$(neuron.u), fired=$(neuron.fired)")
    end

    @testset "Network from genome" begin
        config = make_iznn_test_config()
        genome = Genome(1)
        genome.nodes[0] = make_iz_node(0, bias=0.0)
        genome.connections[(-1, 0)] = make_iz_conn(-1, 0, weight=20.0)

        net = IZNNNetwork(genome, config)

        @test haskey(net.neurons, 0)
        @test net.neurons[0].a == 0.02
        @test net.neurons[0].b == 0.20
        @test net.neurons[0].c == -65.0
        @test net.neurons[0].d == 8.0
        @test length(net.neurons[0].inputs) == 1
        println("  Network from genome: neuron 0 params match NodeGene iz_ fields")
    end

    @testset "Error on NaN iz parameters" begin
        config = make_iznn_test_config()
        genome = Genome(1)
        node = NodeGene(0)
        node.bias = 0.0; node.response = 1.0
        node.activation = :identity; node.aggregation = :sum
        # iz_a etc. are NaN by default
        genome.nodes[0] = node
        genome.connections[(-1, 0)] = make_iz_conn(-1, 0)

        @test_throws ErrorException IZNNNetwork(genome, config)
        println("  NaN iz parameters correctly raises error")
    end

    @testset "External input via set_inputs!" begin
        config = make_iznn_test_config()
        genome = Genome(1)
        genome.nodes[0] = make_iz_node(0, bias=0.0)
        genome.connections[(-1, 0)] = make_iz_conn(-1, 0, weight=20.0)

        net = IZNNNetwork(genome, config)
        set_inputs!(net, [1.0])  # weight=20 → current = 0 + 20*1 = 20

        # Step until spike
        spike_found = false
        for _ in 1:100
            out = advance!(net, 1.0)
            if out[1] == 1.0
                spike_found = true
                break
            end
        end
        @test spike_found
        println("  set_inputs! drives neuron to spike")
    end

    @testset "Recurrent spike propagation" begin
        # Two neurons: A receives input, B receives from A
        # When A spikes, B should receive current on next step
        config = make_iznn_test_config(num_inputs=1, num_outputs=2)
        genome = Genome(1)
        genome.nodes[0] = make_iz_node(0, bias=0.0)  # neuron A
        genome.nodes[1] = make_iz_node(1, bias=0.0)  # neuron B
        genome.connections[(-1, 0)] = make_iz_conn(-1, 0, weight=20.0, innovation=0)
        genome.connections[(0, 1)] = make_iz_conn(0, 1, weight=50.0, innovation=1)

        net = IZNNNetwork(genome, config)
        set_inputs!(net, [1.0])

        # Run until A spikes
        a_spiked = false
        b_spiked_after_a = false
        for step in 1:200
            out = advance!(net, 1.0)
            if out[1] == 1.0 && !a_spiked
                a_spiked = true
                println("  Recurrent: A spiked at step $step")
            end
            if a_spiked && out[2] == 1.0
                b_spiked_after_a = true
                println("  Recurrent: B spiked at step $step (after A)")
                break
            end
        end
        @test a_spiked
        @test b_spiked_after_a
    end

    @testset "Reset restores initial state" begin
        config = make_iznn_test_config()
        genome = Genome(1)
        genome.nodes[0] = make_iz_node(0, bias=0.0)
        genome.connections[(-1, 0)] = make_iz_conn(-1, 0, weight=20.0)

        net = IZNNNetwork(genome, config)
        set_inputs!(net, [1.0])

        # Run several steps
        for _ in 1:50
            advance!(net, 1.0)
        end

        # Reset
        reset!(net)

        # Verify state matches construction
        @test net.neurons[0].v == -65.0
        @test net.neurons[0].u == 0.20 * -65.0
        @test net.neurons[0].fired == 0.0
        @test net.input_values[-1] == 0.0
        println("  Reset: v=$(net.neurons[0].v), u=$(net.neurons[0].u)")
    end

    @testset "Spike timing: hand-computed first spike step" begin
        # With known parameters and current, we can hand-compute the exact
        # number of steps to the first spike.
        # Regular spiking: a=0.02, b=0.2, c=-65, d=8, bias=0, current=10
        # v₀=-65, u₀=-13
        #
        # We'll simulate by hand and verify the code matches.
        neuron = IZNeuron(0.0, 0.02, 0.20, -65.0, 8.0, Tuple{Int,Float64}[])
        neuron.current = 10.0

        # Hand-simulate to find first spike step
        v = -65.0
        u = -13.0
        I = 10.0
        hand_spike_step = nothing
        for step in 1:200
            # Half-step 1
            v += 0.5 * (0.04 * v * v + 5.0 * v + 140.0 - u + I)
            # Half-step 2
            v += 0.5 * (0.04 * v * v + 5.0 * v + 140.0 - u + I)
            # Recovery
            u += 1.0 * 0.02 * (0.2 * v - u)
            if v > 30.0
                hand_spike_step = step
                v = -65.0
                u = u + 8.0
                break
            end
        end

        # Now verify the code matches
        code_spike_step = nothing
        for step in 1:200
            advance!(neuron, 1.0)
            if neuron.fired == 1.0
                code_spike_step = step
                break
            end
        end

        println("  Spike timing: hand=$hand_spike_step, code=$code_spike_step")
        @test hand_spike_step !== nothing
        @test code_spike_step == hand_spike_step
    end

    @testset "Preset constants are accessible" begin
        @test IZ_REGULAR_SPIKING.iz_a == 0.02
        @test IZ_REGULAR_SPIKING.iz_c == -65.0
        @test IZ_FAST_SPIKING.iz_a == 0.10
        println("  Preset constants: accessible and correct")
    end
end

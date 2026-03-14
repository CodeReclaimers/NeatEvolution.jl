"""
Izhikevich spiking neural network evaluation for NEAT genomes.

Izhikevich neurons model biologically realistic spiking dynamics with
4 parameters (a, b, c, d):

    dv/dt = 0.04v² + 5v + 140 - u + I
    du/dt = a(bv - u)
    if v ≥ 30: v ← c, u ← u + d, fired = 1.0

Different parameter sets produce distinct spiking behaviors:
regular spiking, bursting, fast spiking, chattering, etc.

Communication is spike-based (binary 0/1), fundamentally different
from continuous-output networks. This enables experiments with
temporal coding, spike timing, and biologically inspired learning rules.

Matches neat-python's neat/iznn implementation.
"""

# Named parameter presets for common Izhikevich neuron types.
# Values from Izhikevich (2003) "Simple Model of Spiking Neurons".
const IZ_REGULAR_SPIKING  = (iz_a=0.02, iz_b=0.20, iz_c=-65.0, iz_d=8.00)
const IZ_INTRINSIC_BURST  = (iz_a=0.02, iz_b=0.20, iz_c=-55.0, iz_d=4.00)
const IZ_CHATTERING       = (iz_a=0.02, iz_b=0.20, iz_c=-50.0, iz_d=2.00)
const IZ_FAST_SPIKING     = (iz_a=0.10, iz_b=0.20, iz_c=-65.0, iz_d=2.00)
const IZ_THALAMO_CORTICAL = (iz_a=0.02, iz_b=0.25, iz_c=-65.0, iz_d=0.05)
const IZ_RESONATOR        = (iz_a=0.10, iz_b=0.25, iz_c=-65.0, iz_d=2.00)
const IZ_LOW_THRESHOLD    = (iz_a=0.02, iz_b=0.25, iz_c=-65.0, iz_d=2.00)

"""
Izhikevich neuron with model parameters and state variables.

Uses bare field names (a, b, c, d) internally since they are local to the
evaluator with no collision risk. The iz_ prefix is used in the genome/config
layer where disambiguation matters.
"""
mutable struct IZNeuron
    # Model parameters (from genome node gene)
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    bias::Float64
    inputs::Vector{Tuple{Int, Float64}}  # (source_node_id, weight)
    # State variables
    v::Float64       # membrane potential (mV)
    u::Float64       # recovery variable
    fired::Float64   # 0.0 or 1.0
    current::Float64 # total input current
end

"""
    IZNeuron(bias, a, b, c, d, inputs)

Create an Izhikevich neuron with initial state v=c, u=b*c.
"""
function IZNeuron(bias::Float64, a::Float64, b::Float64, c::Float64, d::Float64,
                  inputs::Vector{Tuple{Int, Float64}})
    IZNeuron(a, b, c, d, bias, inputs, c, b * c, 0.0, bias)
end

"""
    advance!(neuron::IZNeuron, dt_msec::Float64)

Advance the neuron state by dt_msec milliseconds.

Uses two half-step voltage updates for accuracy (matching neat-python),
one full recovery step, and a spike check at v > 30.
"""
function advance!(neuron::IZNeuron, dt_msec::Float64)
    # Two half-steps for voltage (improves numerical stability)
    half_dt = 0.5 * dt_msec
    v = neuron.v
    u = neuron.u
    I = neuron.current

    # Half-step 1
    v += half_dt * (0.04 * v * v + 5.0 * v + 140.0 - u + I)

    # Guard against overflow (Julia produces Inf/NaN instead of raising)
    if !isfinite(v)
        neuron.v = neuron.c
        neuron.u = neuron.b * neuron.c
        neuron.fired = 0.0
        return
    end

    # Half-step 2
    v += half_dt * (0.04 * v * v + 5.0 * v + 140.0 - u + I)

    if !isfinite(v)
        neuron.v = neuron.c
        neuron.u = neuron.b * neuron.c
        neuron.fired = 0.0
        return
    end

    # Full recovery step
    u += dt_msec * neuron.a * (neuron.b * v - u)

    if !isfinite(u)
        neuron.v = neuron.c
        neuron.u = neuron.b * neuron.c
        neuron.fired = 0.0
        return
    end

    # Spike check
    if v >= 30.0
        neuron.fired = 1.0
        neuron.v = neuron.c
        neuron.u = u + neuron.d
    else
        neuron.fired = 0.0
        neuron.v = v
        neuron.u = u
    end
end

"""
Izhikevich spiking neural network.
"""
mutable struct IZNNNetwork <: AbstractNetwork
    neurons::Dict{Int, IZNeuron}
    input_nodes::Vector{Int}
    output_nodes::Vector{Int}
    input_values::Dict{Int, Float64}
    _output::Vector{Float64}   # pre-allocated output buffer (returned by advance!)
end

"""
    IZNNNetwork(genome::Genome, config::GenomeConfig)

Create an IZNN network from a genome. Requires that NodeGene iz_a/b/c/d
fields are configured (not NaN) for all relevant nodes.
"""
function IZNNNetwork(genome::Genome, config::GenomeConfig)
    # Gather enabled connections
    connections = [cg.key for cg in Base.values(genome.connections) if cg.enabled]

    # Find all nodes required to compute output
    required = required_for_output(config.input_keys, config.output_keys, connections)

    # Build neurons for all required non-input nodes
    neurons = Dict{Int, IZNeuron}()
    eval_nodes = sort(collect(setdiff(required, Set(config.input_keys))))

    for node_id in eval_nodes
        ng = genome.nodes[node_id]

        # Validate Izhikevich parameters are configured
        if isnan(ng.iz_a) || isnan(ng.iz_b) || isnan(ng.iz_c) || isnan(ng.iz_d)
            error("Node $node_id has NaN Izhikevich parameters. " *
                  "Add iz_a_init_mean (and related params) to your " *
                  "[DefaultGenome] config section to use IZNNNetwork.")
        end

        # Find inputs for this node
        inputs = Tuple{Int, Float64}[]
        for conn_key in connections
            inode, onode = conn_key
            if onode == node_id
                cg = genome.connections[conn_key]
                push!(inputs, (inode, cg.weight))
            end
        end

        neurons[node_id] = IZNeuron(ng.bias, ng.iz_a, ng.iz_b, ng.iz_c, ng.iz_d, inputs)
    end

    # Initialize input values
    input_values = Dict{Int, Float64}()
    for key in config.input_keys
        input_values[key] = 0.0
    end

    _output = Vector{Float64}(undef, length(config.output_keys))

    IZNNNetwork(neurons, config.input_keys, config.output_keys, input_values, _output)
end

"""
    set_inputs!(net::IZNNNetwork, inputs::Vector{Float64})

Set external input values for the network.
"""
function set_inputs!(net::IZNNNetwork, inputs::Vector{Float64})
    if length(net.input_nodes) != length(inputs)
        error("Expected $(length(net.input_nodes)) inputs, got $(length(inputs))")
    end
    for (k, v) in zip(net.input_nodes, inputs)
        net.input_values[k] = v
    end
end

"""
    advance!(net::IZNNNetwork, dt_msec::Float64)

Advance the network by dt_msec milliseconds.

1. Accumulate currents for each neuron from connected sources
2. Advance all neurons
3. Return output spike values

Source values are: `fired` for recurrent neuron connections,
`input_values[id]` for external inputs.
"""
function advance!(net::IZNNNetwork, dt_msec::Float64)
    # Accumulate currents
    for (node_id, neuron) in net.neurons
        neuron.current = neuron.bias
        for (source_id, weight) in neuron.inputs
            if haskey(net.neurons, source_id)
                # Recurrent connection: use source neuron's fired state
                neuron.current += weight * net.neurons[source_id].fired
            elseif haskey(net.input_values, source_id)
                # External input
                neuron.current += weight * net.input_values[source_id]
            end
        end
    end

    # Advance all neurons
    for (_, neuron) in net.neurons
        advance!(neuron, dt_msec)
    end

    # Fill pre-allocated output buffer
    for (j, i) in enumerate(net.output_nodes)
        net._output[j] = net.neurons[i].fired
    end
    return net._output
end

"""
    reset!(net::IZNNNetwork)

Reset all neurons to initial state (v=c, u=b*c, fired=0, current=bias).
"""
function reset!(net::IZNNNetwork)
    for (_, neuron) in net.neurons
        neuron.v = neuron.c
        neuron.u = neuron.b * neuron.c
        neuron.fired = 0.0
        neuron.current = neuron.bias
    end
    for key in keys(net.input_values)
        net.input_values[key] = 0.0
    end
end

"""Convenience constructor accepting `Config` instead of `GenomeConfig`."""
IZNNNetwork(genome::Genome, config::Config) = IZNNNetwork(genome, config.genome_config)

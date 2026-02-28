"""
Recurrent neural network evaluation for NEAT genomes.

Unlike FeedForwardNetwork which requires an acyclic graph and computes in
layer order, RecurrentNetwork handles cycles (including self-connections) by
maintaining internal state across calls to activate!. Each call reads from
the previous timestep's node values and writes to current values, then swaps.

This matches neat-python's neat/nn/recurrent.py implementation.
"""

"""
RecurrentNetwork evaluates a genome as a recurrent neural network.

Maintains internal state between activations via `values` (current timestep)
and `prev_values` (previous timestep). Cycles are handled by reading inputs
from prev_values while writing outputs to values.
"""
mutable struct RecurrentNetwork <: AbstractNetwork
    input_nodes::Vector{Int}
    output_nodes::Vector{Int}
    node_evals::Vector{NodeEval}
    values::Dict{Int, Float64}
    prev_values::Dict{Int, Float64}
    _buffer::Vector{Float64}   # pre-allocated workspace for aggregation
    _output::Vector{Float64}   # pre-allocated output buffer (returned by activate!)
end

"""
Create a recurrent network from a genome.

Uses required_for_output() to find relevant nodes, but does NOT call
feed_forward_layers() (which requires an acyclic graph). Non-input nodes
are evaluated in sorted order by node ID for determinism.
"""
function RecurrentNetwork(genome::Genome, config::GenomeConfig)
    # Gather enabled connections
    connections = [cg.key for cg in Base.values(genome.connections) if cg.enabled]

    # Find all nodes required to compute output
    required = required_for_output(config.input_keys, config.output_keys, connections)

    # Build node evaluation list for all required non-input nodes, sorted by ID
    node_evals = NodeEval[]
    eval_nodes = sort(collect(setdiff(required, Set(config.input_keys))))

    for node in eval_nodes
        # Find inputs for this node from enabled connections
        inputs = Tuple{Int, Float64}[]
        for conn_key in connections
            inode, onode = conn_key
            if onode == node
                cg = genome.connections[conn_key]
                push!(inputs, (inode, cg.weight))
            end
        end

        # Get node gene
        ng = genome.nodes[node]

        # Wrap activation and aggregation functions for type stability
        act_func = ActivationFn(get_activation_function(ng.activation))
        agg_func = AggregationFn(get_aggregation_function(ng.aggregation))

        push!(node_evals, (node, act_func, agg_func, ng.bias, ng.response, inputs))
    end

    # Initialize value dictionaries with zeros
    values = Dict{Int, Float64}()
    prev_values = Dict{Int, Float64}()

    for key in config.input_keys
        values[key] = 0.0
        prev_values[key] = 0.0
    end
    for key in config.output_keys
        values[key] = 0.0
        prev_values[key] = 0.0
    end
    for (node, _, _, _, _, _) in node_evals
        if !haskey(values, node)
            values[node] = 0.0
            prev_values[node] = 0.0
        end
    end

    # Pre-allocate buffer for aggregation inputs
    max_links = isempty(node_evals) ? 0 : maximum(length(last(ne)) for ne in node_evals)
    _buffer = Vector{Float64}(undef, max_links)

    _output = Vector{Float64}(undef, length(config.output_keys))

    RecurrentNetwork(config.input_keys, config.output_keys, node_evals, values, prev_values, _buffer, _output)
end

"""
Activate the recurrent network with given inputs and return outputs.

Reads input values from prev_values (previous timestep) and writes computed
values to values (current timestep). After all nodes are computed, copies
values to prev_values for the next call.

Returns a reference to an internal buffer. Callers who need to store results
across multiple `activate!` calls should copy the returned vector.
"""
function activate!(network::RecurrentNetwork, inputs::Vector{Float64})
    if length(network.input_nodes) != length(inputs)
        error("Expected $(length(network.input_nodes)) inputs, got $(length(inputs))")
    end

    # Set input values in both current and previous
    for (k, v) in zip(network.input_nodes, inputs)
        network.values[k] = v
        network.prev_values[k] = v
    end

    # Evaluate each node, reading from prev_values
    for (node, act_func, agg_func, bias, response, links) in network.node_evals
        n = length(links)
        resize!(network._buffer, n)
        for (j, (i, w)) in enumerate(links)
            network._buffer[j] = network.prev_values[i] * w
        end

        s = agg_func(network._buffer)
        network.values[node] = act_func(bias + response * s)
    end

    # Copy current values to prev_values for next timestep
    for (k, v) in network.values
        network.prev_values[k] = v
    end

    # Fill pre-allocated output buffer
    for (j, i) in enumerate(network.output_nodes)
        network._output[j] = network.values[i]
    end
    return network._output
end

"""
Reset the recurrent network state.

Zeros out both value dictionaries, restoring the network to its initial state.
"""
function reset!(network::RecurrentNetwork)
    for k in keys(network.values)
        network.values[k] = 0.0
    end
    for k in keys(network.prev_values)
        network.prev_values[k] = 0.0
    end
end

"""Convenience constructor accepting `Config` instead of `GenomeConfig`."""
RecurrentNetwork(genome::Genome, config::Config) = RecurrentNetwork(genome, config.genome_config)

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
mutable struct RecurrentNetwork
    input_nodes::Vector{Int}
    output_nodes::Vector{Int}
    node_evals::Vector{NodeEval}
    values::Dict{Int, Float64}
    prev_values::Dict{Int, Float64}
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

        # Get activation and aggregation functions
        activation_func = get_activation_function(ng.activation)
        aggregation_func = get_aggregation_function(ng.aggregation)

        push!(node_evals, (node, activation_func, aggregation_func, ng.bias, ng.response, inputs))
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

    RecurrentNetwork(config.input_keys, config.output_keys, node_evals, values, prev_values)
end

"""
Activate the recurrent network with given inputs and return outputs.

Reads input values from prev_values (previous timestep) and writes computed
values to values (current timestep). After all nodes are computed, copies
values to prev_values for the next call.
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
        node_inputs = Float64[]
        for (i, w) in links
            push!(node_inputs, network.prev_values[i] * w)
        end

        s = agg_func(node_inputs)
        network.values[node] = act_func(bias + response * s)
    end

    # Copy current values to prev_values for next timestep
    for (k, v) in network.values
        network.prev_values[k] = v
    end

    # Return output values
    return [network.values[i] for i in network.output_nodes]
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

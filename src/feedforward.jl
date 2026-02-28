"""
Feed-forward neural network evaluation for NEAT genomes.
"""

"""
Type-stable function wrappers for activation and aggregation functions.
"""
const ActivationFn = FunctionWrapper{Float64, Tuple{Float64}}
const AggregationFn = FunctionWrapper{Float64, Tuple{Vector{Float64}}}

"""
Node evaluation tuple: (node_id, activation_func, aggregation_func, bias, response, inputs)
where inputs is a vector of (input_node_id, weight) tuples.
"""
const NodeEval = Tuple{Int, ActivationFn, AggregationFn, Float64, Float64, Vector{Tuple{Int, Float64}}}

"""
FeedForwardNetwork evaluates a genome as a neural network.
"""
struct FeedForwardNetwork <: AbstractNetwork
    input_nodes::Vector{Int}
    output_nodes::Vector{Int}
    node_evals::Vector{NodeEval}
    values::Dict{Int, Float64}
    _buffer::Vector{Float64}   # pre-allocated workspace for aggregation
    _output::Vector{Float64}   # pre-allocated output buffer (returned by activate!)
end

"""
Create a feed-forward network from a genome.
"""
function FeedForwardNetwork(genome::Genome, config::GenomeConfig)
    # Gather enabled connections
    connections = [cg.key for cg in Base.values(genome.connections) if cg.enabled]

    # Compute feed-forward layers
    layers = feed_forward_layers(config.input_keys, config.output_keys, connections)

    # Build node evaluation list
    node_evals = NodeEval[]
    for layer in layers
        for node in layer
            # Find inputs for this node
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
    end

    # Initialize values dict
    values = Dict{Int, Float64}()
    # Include inputs, outputs, and all nodes in the evaluation order
    for key in config.input_keys
        values[key] = 0.0
    end
    for key in config.output_keys
        values[key] = 0.0
    end
    # Also initialize values for nodes in the evaluation list
    for (node, _, _, _, _, _) in node_evals
        if !haskey(values, node)
            values[node] = 0.0
        end
    end

    # Pre-allocate buffer for aggregation inputs
    max_links = isempty(node_evals) ? 0 : maximum(length(last(ne)) for ne in node_evals)
    _buffer = Vector{Float64}(undef, max_links)

    _output = Vector{Float64}(undef, length(config.output_keys))

    FeedForwardNetwork(config.input_keys, config.output_keys, node_evals, values, _buffer, _output)
end

"""
Activate the network with given inputs and return outputs.

Returns a reference to an internal buffer. Callers who need to store results
across multiple `activate!` calls should copy the returned vector.
"""
function activate!(network::FeedForwardNetwork, inputs::Vector{Float64})
    if length(network.input_nodes) != length(inputs)
        error("Expected $(length(network.input_nodes)) inputs, got $(length(inputs))")
    end

    # Set input values
    for (k, v) in zip(network.input_nodes, inputs)
        network.values[k] = v
    end

    # Evaluate each node in order
    for (node, act_func, agg_func, bias, response, links) in network.node_evals
        n = length(links)
        resize!(network._buffer, n)
        for (j, (i, w)) in enumerate(links)
            network._buffer[j] = network.values[i] * w
        end

        s = agg_func(network._buffer)
        network.values[node] = act_func(bias + response * s)
    end

    # Fill pre-allocated output buffer
    for (j, i) in enumerate(network.output_nodes)
        network._output[j] = network.values[i]
    end
    return network._output
end

"""Convenience constructor accepting `Config` instead of `GenomeConfig`."""
FeedForwardNetwork(genome::Genome, config::Config) = FeedForwardNetwork(genome, config.genome_config)

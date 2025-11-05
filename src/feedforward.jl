"""
Feed-forward neural network evaluation for NEAT genomes.
"""

"""
Node evaluation tuple: (node_id, activation_func, aggregation_func, bias, response, inputs)
where inputs is a vector of (input_node_id, weight) tuples.
"""
const NodeEval = Tuple{Int, Function, Function, Float64, Float64, Vector{Tuple{Int, Float64}}}

"""
FeedForwardNetwork evaluates a genome as a neural network.
"""
struct FeedForwardNetwork
    input_nodes::Vector{Int}
    output_nodes::Vector{Int}
    node_evals::Vector{NodeEval}
    values::Dict{Int, Float64}
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

            # Get activation and aggregation functions
            activation_func = get_activation_function(ng.activation)
            aggregation_func = get_aggregation_function(ng.aggregation)

            push!(node_evals, (node, activation_func, aggregation_func, ng.bias, ng.response, inputs))
        end
    end

    # Initialize values dict
    values = Dict{Int, Float64}()
    for key in vcat(config.input_keys, config.output_keys)
        values[key] = 0.0
    end

    FeedForwardNetwork(config.input_keys, config.output_keys, node_evals, values)
end

"""
Activate the network with given inputs and return outputs.
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
        node_inputs = Float64[]
        for (i, w) in links
            push!(node_inputs, network.values[i] * w)
        end

        s = agg_func(node_inputs)
        network.values[node] = act_func(bias + response * s)
    end

    # Return output values
    return [network.values[i] for i in network.output_nodes]
end

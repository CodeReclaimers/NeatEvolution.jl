"""
Continuous-Time Recurrent Neural Network (CTRNN) evaluation for NEAT genomes.

CTRNNs model continuous temporal dynamics using the equation:

    τᵢ · dyᵢ/dt = -yᵢ + fᵢ(βᵢ + ρᵢ · Σⱼ wᵢⱼ · yⱼ)

Integrated via forward Euler:

    yᵢ(t+Δt) = yᵢ(t) + (Δt/τᵢ) · (-yᵢ(t) + z)

where z = activation(bias + response * aggregation(weighted_inputs)).

Per-node time constants allow evolving heterogeneous temporal responses:
nodes with small tau react quickly, nodes with large tau integrate over time.

Matches neat-python's neat/ctrnn implementation with the extension of
per-node evolvable time constants (neat-python uses a single global tau).
"""

"""
Evaluation data for a single CTRNN node.
"""
struct CTRNNNodeEval
    node_id::Int
    time_constant::Float64
    activation::ActivationFn
    aggregation::AggregationFn
    bias::Float64
    response::Float64
    links::Vector{Tuple{Int, Float64}}
end

"""
CTRNN network with double-buffered state for integration.

Uses two value buffers and flips between them during integration steps,
matching neat-python's implementation.
"""
mutable struct CTRNNNetwork <: AbstractNetwork
    input_nodes::Vector{Int}
    output_nodes::Vector{Int}
    node_evals::Dict{Int, CTRNNNodeEval}
    values::Vector{Dict{Int, Float64}}   # [buffer1, buffer2]
    active::Int                          # 1 or 2 (current read buffer)
    time_seconds::Float64
    _buffer::Vector{Float64}   # pre-allocated workspace for aggregation
    _output::Vector{Float64}   # pre-allocated output buffer (returned by advance!)
end

"""
    CTRNNNetwork(genome::Genome, config::GenomeConfig)

Create a CTRNN network from a genome. Requires that NodeGene.time_constant
is configured (not NaN) for all relevant nodes.
"""
function CTRNNNetwork(genome::Genome, config::GenomeConfig)
    # Gather enabled connections
    connections = [cg.key for cg in Base.values(genome.connections) if cg.enabled]

    # Find all nodes required to compute output
    required = required_for_output(config.input_keys, config.output_keys, connections)

    # Build node evaluations for all required non-input nodes
    node_evals = Dict{Int, CTRNNNodeEval}()
    eval_nodes = sort(collect(setdiff(required, Set(config.input_keys))))

    max_links = 0
    for node_id in eval_nodes
        ng = genome.nodes[node_id]

        # Validate time_constant is configured
        if isnan(ng.time_constant)
            error("Node $node_id has NaN time_constant. " *
                  "Add time_constant_init_mean (and related params) to your " *
                  "[DefaultGenome] config section to use CTRNNNetwork.")
        end

        # Find inputs for this node
        links = Tuple{Int, Float64}[]
        for conn_key in connections
            inode, onode = conn_key
            if onode == node_id
                cg = genome.connections[conn_key]
                push!(links, (inode, cg.weight))
            end
        end

        # Wrap activation and aggregation functions for type stability
        act_func = ActivationFn(get_activation_function(ng.activation))
        agg_func = AggregationFn(get_aggregation_function(ng.aggregation))

        node_evals[node_id] = CTRNNNodeEval(
            node_id, ng.time_constant,
            act_func, agg_func,
            ng.bias, ng.response, links
        )

        max_links = max(max_links, length(links))
    end

    # Initialize double buffers with zeros for all relevant nodes
    buf1 = Dict{Int, Float64}()
    buf2 = Dict{Int, Float64}()
    for key in config.input_keys
        buf1[key] = 0.0
        buf2[key] = 0.0
    end
    for node_id in keys(node_evals)
        buf1[node_id] = 0.0
        buf2[node_id] = 0.0
    end

    # Pre-allocate buffer for aggregation inputs
    _buffer = Vector{Float64}(undef, max_links)

    _output = Vector{Float64}(undef, length(config.output_keys))

    CTRNNNetwork(config.input_keys, config.output_keys, node_evals,
                 [buf1, buf2], 1, 0.0, _buffer, _output)
end

"""
    advance!(net::CTRNNNetwork, inputs::Vector{Float64}, advance_time::Float64, time_step::Float64)

Advance the CTRNN simulation by `advance_time` seconds using forward Euler
integration with step size `time_step`.

Returns output node values after integration.
"""
function advance!(net::CTRNNNetwork, inputs::Vector{Float64},
                  advance_time::Float64, time_step::Float64)
    if length(net.input_nodes) != length(inputs)
        error("Expected $(length(net.input_nodes)) inputs, got $(length(inputs))")
    end

    final_time = net.time_seconds + advance_time

    while net.time_seconds < final_time
        dt = min(time_step, final_time - net.time_seconds)

        ivalues = net.values[net.active]
        net.active = 3 - net.active  # flip: 1→2, 2→1
        ovalues = net.values[net.active]

        # Set inputs in both buffers
        for (k, v) in zip(net.input_nodes, inputs)
            ivalues[k] = v
            ovalues[k] = v
        end

        # Update each node using forward Euler
        for (node_id, ne) in net.node_evals
            # Aggregate weighted inputs from read buffer
            n = length(ne.links)
            resize!(net._buffer, n)
            for (j, (i, w)) in enumerate(ne.links)
                net._buffer[j] = ivalues[i] * w
            end

            s = ne.aggregation(net._buffer)
            z = ne.activation(ne.bias + ne.response * s)

            # Forward Euler: y(t+dt) = y(t) + (dt/tau) * (-y(t) + z)
            ovalues[node_id] = ivalues[node_id] + (dt / ne.time_constant) * (-ivalues[node_id] + z)
        end

        net.time_seconds += dt
    end

    # Fill pre-allocated output buffer
    active_buf = net.values[net.active]
    for (j, i) in enumerate(net.output_nodes)
        net._output[j] = active_buf[i]
    end
    return net._output
end

"""
    reset!(net::CTRNNNetwork)

Reset the CTRNN network state. Zeros both buffers and resets time.
"""
function reset!(net::CTRNNNetwork)
    for buf in net.values
        for k in keys(buf)
            buf[k] = 0.0
        end
    end
    net.active = 1
    net.time_seconds = 0.0
end

"""
    set_node_value!(net::CTRNNNetwork, node_key::Int, value::Float64)

Set a node's value in both buffers. Useful for injecting state.
"""
function set_node_value!(net::CTRNNNetwork, node_key::Int, value::Float64)
    net.values[1][node_key] = value
    net.values[2][node_key] = value
end

"""Convenience constructor accepting `Config` instead of `GenomeConfig`."""
CTRNNNetwork(genome::Genome, config::Config) = CTRNNNetwork(genome, config.genome_config)

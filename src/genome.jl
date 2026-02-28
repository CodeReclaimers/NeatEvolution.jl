"""
Genome representation for NEAT.

A genome contains nodes and connections that define a neural network.
"""

using Random

"""
Genome represents an individual in the population.
"""
mutable struct Genome
    key::Int
    nodes::Dict{Int, NodeGene}
    connections::Dict{Tuple{Int, Int}, ConnectionGene}
    fitness::Union{Float64, Nothing}
end

function Genome(key::Int)
    Genome(key, Dict{Int, NodeGene}(), Dict{Tuple{Int, Int}, ConnectionGene}(), nothing)
end

"""
Initialize a new genome with random structure based on configuration.
"""
function configure_new!(genome::Genome, config::GenomeConfig, rng::AbstractRNG=Random.GLOBAL_RNG)
    # Create output nodes
    for node_key in config.output_keys
        node = NodeGene(node_key)
        init_attributes!(node, config, rng)
        genome.nodes[node_key] = node
    end

    # Add hidden nodes if requested
    for _ in 1:config.num_hidden
        node_key = get_new_node_id!(config)
        node = NodeGene(node_key)
        init_attributes!(node, config, rng)
        genome.nodes[node_key] = node
    end

    # Add connections based on initial_connection strategy
    if config.initial_connection == :unconnected
        # No connections
    elseif config.initial_connection == :full || config.initial_connection == :full_nodirect
        connect_full!(genome, config, false, rng)
    elseif config.initial_connection == :full_direct
        connect_full!(genome, config, true, rng)
    elseif config.initial_connection == :fs_neat || config.initial_connection == :fs_neat_nohidden
        connect_fs_neat!(genome, config, false, rng)
    elseif config.initial_connection == :fs_neat_hidden
        connect_fs_neat!(genome, config, true, rng)
    elseif config.initial_connection == :partial || config.initial_connection == :partial_nodirect
        connect_partial!(genome, config, false, rng)
    elseif config.initial_connection == :partial_direct
        connect_partial!(genome, config, true, rng)
    end

    return genome
end

"""
Create connections for a fully-connected network.
"""
function connect_full!(genome::Genome, config::GenomeConfig, direct::Bool, rng::AbstractRNG)
    # Separate hidden and output nodes
    hidden = [k for k in keys(genome.nodes) if !(k in config.output_keys)]
    output = [k for k in keys(genome.nodes) if k in config.output_keys]

    # Connect inputs to hidden nodes (or outputs if no hidden)
    if !isempty(hidden)
        for input_key in config.input_keys
            for hidden_key in hidden
                add_connection!(genome, config, input_key, hidden_key, rng)
            end
        end
        # Connect hidden to outputs
        for hidden_key in hidden
            for output_key in output
                add_connection!(genome, config, hidden_key, output_key, rng)
            end
        end
    end

    # Direct input-output connections
    if direct || isempty(hidden)
        for input_key in config.input_keys
            for output_key in output
                add_connection!(genome, config, input_key, output_key, rng)
            end
        end
    end

    # For recurrent networks, add self-connections
    if !config.feed_forward
        for node_key in keys(genome.nodes)
            add_connection!(genome, config, node_key, node_key, rng)
        end
    end
end

"""
Create connections for FS-NEAT initialization (inputs to outputs only).
"""
function connect_fs_neat!(genome::Genome, config::GenomeConfig, connect_hidden::Bool, rng::AbstractRNG)
    hidden = [k for k in keys(genome.nodes) if !(k in config.output_keys)]
    output = [k for k in keys(genome.nodes) if k in config.output_keys]

    # If connect_hidden is true and there are hidden nodes, connect through hidden
    if connect_hidden && !isempty(hidden)
        for input_key in config.input_keys
            for hidden_key in hidden
                add_connection!(genome, config, input_key, hidden_key, rng)
            end
        end
        for hidden_key in hidden
            for output_key in output
                add_connection!(genome, config, hidden_key, output_key, rng)
            end
        end
    else
        # Otherwise, connect inputs directly to outputs
        for input_key in config.input_keys
            for output_key in output
                add_connection!(genome, config, input_key, output_key, rng)
            end
        end
    end
end

"""
Create partial connections based on connection_fraction.
"""
function connect_partial!(genome::Genome, config::GenomeConfig, direct::Bool, rng::AbstractRNG)
    hidden = [k for k in keys(genome.nodes) if !(k in config.output_keys)]
    output = [k for k in keys(genome.nodes) if k in config.output_keys]

    # Build list of all possible connections
    possible_connections = Tuple{Int, Int}[]

    if !isempty(hidden)
        # Input to hidden connections
        for input_key in config.input_keys
            for hidden_key in hidden
                push!(possible_connections, (input_key, hidden_key))
            end
        end
        # Hidden to output connections
        for hidden_key in hidden
            for output_key in output
                push!(possible_connections, (hidden_key, output_key))
            end
        end
    end

    # Direct input-output connections
    if direct || isempty(hidden)
        for input_key in config.input_keys
            for output_key in output
                push!(possible_connections, (input_key, output_key))
            end
        end
    end

    # Randomly select a fraction of connections
    num_connections = round(Int, length(possible_connections) * config.connection_fraction)
    shuffle!(rng, possible_connections)
    for i in 1:num_connections
        input_key, output_key = possible_connections[i]
        add_connection!(genome, config, input_key, output_key, rng)
    end
end

"""
Add a connection between two nodes.

Assigns an innovation number using the config's innovation tracking system.
Per NEAT paper: if the same connection structure appears multiple times
in a generation, it receives the same innovation number.
"""
function add_connection!(genome::Genome, config::GenomeConfig, input_key::Int, output_key::Int,
                        rng::AbstractRNG=Random.GLOBAL_RNG)
    key = (input_key, output_key)
    innovation = get_innovation!(config, key)
    conn = ConnectionGene(key, innovation)
    init_attributes!(conn, config, rng)
    genome.connections[key] = conn
end

"""
Configure a genome through crossover of two parents.
"""
function configure_crossover!(genome::Genome, parent1::Genome, parent2::Genome,
                             config::GenomeConfig, rng::AbstractRNG=Random.GLOBAL_RNG)
    # Determine which parent is fitter
    if parent1.fitness === nothing || parent2.fitness === nothing ||
       parent1.fitness == parent2.fitness
        fitter, other = parent1, parent2
    elseif parent1.fitness > parent2.fitness
        fitter, other = parent1, parent2
    else
        fitter, other = parent2, parent1
    end

    # Inherit connection genes
    for (key, cg1) in fitter.connections
        cg2 = get(other.connections, key, nothing)
        if cg2 === nothing
            # Disjoint/excess gene: copy from fitter parent
            genome.connections[key] = copy(cg1)
        else
            # Homologous gene: crossover
            genome.connections[key] = crossover(cg1, cg2, rng)
        end
    end

    # Inherit node genes
    for (key, ng1) in fitter.nodes
        ng2 = get(other.nodes, key, nothing)
        if ng2 === nothing
            # Excess gene: copy from fitter parent
            genome.nodes[key] = copy(ng1)
        else
            # Homologous gene: crossover
            genome.nodes[key] = crossover(ng1, ng2, rng)
        end
    end

    return genome
end

"""
Mutate this genome.
"""
function mutate!(genome::Genome, config::GenomeConfig, rng::AbstractRNG=Random.GLOBAL_RNG)
    if config.single_structural_mutation
        # Only one structural mutation per call
        div = config.node_add_prob + config.node_delete_prob +
              config.conn_add_prob + config.conn_delete_prob

        # Only perform structural mutation if at least one probability is non-zero
        if div > 0.0
            div = max(1.0, div)

            r = rand(rng)
            if r < config.node_add_prob / div
                mutate_add_node!(genome, config, rng)
            elseif r < (config.node_add_prob + config.node_delete_prob) / div
                mutate_delete_node!(genome, config, rng)
            elseif r < (config.node_add_prob + config.node_delete_prob +
                        config.conn_add_prob) / div
                mutate_add_connection!(genome, config, rng)
            elseif r < 1.0
                mutate_delete_connection!(genome, config, rng)
            end
        end
    else
        # Multiple structural mutations possible
        if rand(rng) < config.node_add_prob
            mutate_add_node!(genome, config, rng)
        end
        if rand(rng) < config.node_delete_prob
            mutate_delete_node!(genome, config, rng)
        end
        if rand(rng) < config.conn_add_prob
            mutate_add_connection!(genome, config, rng)
        end
        if rand(rng) < config.conn_delete_prob
            mutate_delete_connection!(genome, config, rng)
        end
    end

    # Mutate connection genes
    for cg in values(genome.connections)
        mutate!(cg, config, rng)
    end

    # Mutate node genes
    for ng in values(genome.nodes)
        mutate!(ng, config, rng)
    end

    return genome
end

"""
Add a new node by splitting an existing connection.
"""
function mutate_add_node!(genome::Genome, config::GenomeConfig, rng::AbstractRNG)
    if isempty(genome.connections)
        return
    end

    # Choose random connection to split
    conn_to_split = rand(rng, collect(values(genome.connections)))

    # Create new node
    new_node_id = get_new_node_id!(config)
    ng = NodeGene(new_node_id)
    init_attributes!(ng, config, rng)
    ng.bias = 0.0  # Set to zero per NEAT paper to keep mutation neutral
    genome.nodes[new_node_id] = ng

    # Disable old connection
    conn_to_split.enabled = false

    # Add two new connections
    i, o = conn_to_split.key
    add_connection!(genome, config, i, new_node_id, rng)
    genome.connections[(i, new_node_id)].weight = 1.0
    genome.connections[(i, new_node_id)].enabled = true

    add_connection!(genome, config, new_node_id, o, rng)
    genome.connections[(new_node_id, o)].weight = conn_to_split.weight
    genome.connections[(new_node_id, o)].enabled = true
end

"""
Delete a random node (excluding output nodes).
"""
function mutate_delete_node!(genome::Genome, config::GenomeConfig, rng::AbstractRNG)
    # Find nodes that can be deleted (not outputs)
    available = [k for k in keys(genome.nodes) if !(k in config.output_keys)]

    if isempty(available)
        return
    end

    del_key = rand(rng, available)

    # Delete connections involving this node
    to_delete = [k for (k, v) in genome.connections if del_key in k]
    for k in to_delete
        delete!(genome.connections, k)
    end

    # Delete the node
    delete!(genome.nodes, del_key)
end

"""
Add a new connection between random nodes.
"""
function mutate_add_connection!(genome::Genome, config::GenomeConfig, rng::AbstractRNG)
    possible_outputs = collect(keys(genome.nodes))
    possible_inputs = vcat(possible_outputs, config.input_keys)

    if isempty(possible_outputs)
        return
    end

    out_node = rand(rng, possible_outputs)
    in_node = rand(rng, possible_inputs)

    key = (in_node, out_node)

    # Don't duplicate connections
    if haskey(genome.connections, key)
        return
    end

    # Don't connect two output nodes
    if in_node in config.output_keys && out_node in config.output_keys
        return
    end

    # For feed-forward networks, avoid cycles
    if config.feed_forward
        existing_conns = collect(keys(genome.connections))
        if creates_cycle(existing_conns, key)
            return
        end
    end

    add_connection!(genome, config, in_node, out_node, rng)
end

"""
Delete a random connection.
"""
function mutate_delete_connection!(genome::Genome, config::GenomeConfig, rng::AbstractRNG)
    if !isempty(genome.connections)
        key = rand(rng, collect(keys(genome.connections)))
        delete!(genome.connections, key)
    end
end

"""
Compute genetic distance between two genomes for speciation.

Per NEAT paper (Equation 1, Section 3.3):
δ = (c₁·E)/N + (c₂·D)/N + c₃·W̄

Where:
- E = number of excess genes (beyond the other genome's innovation range)
- D = number of disjoint genes (within range but not matching)
- W̄ = average weight difference of matching genes
- N = number of genes in larger genome
- c₁, c₂, c₃ = compatibility coefficients

The paper only specifies distance for connection genes. For node genes,
we use a simplified approach: count non-matching nodes as disjoint.
"""
function distance(genome1::Genome, genome2::Genome, config::GenomeConfig)
    # Connection gene distance per NEAT paper
    if isempty(genome1.connections) && isempty(genome2.connections)
        # Both genomes have no connections - use simple node distance
        return _node_distance_simple(genome1, genome2, config)
    end

    # Get all connection genes sorted by innovation number
    conn1 = sort(collect(values(genome1.connections)), by=c->c.innovation)
    conn2 = sort(collect(values(genome2.connections)), by=c->c.innovation)

    # Find innovation number ranges
    if !isempty(conn1) && !isempty(conn2)
        max_innov1 = maximum(c.innovation for c in conn1)
        max_innov2 = maximum(c.innovation for c in conn2)
        min_innov1 = minimum(c.innovation for c in conn1)
        min_innov2 = minimum(c.innovation for c in conn2)

        # Build innovation sets for quick lookup
        innov1 = Set(c.innovation for c in conn1)
        innov2 = Set(c.innovation for c in conn2)

        # Count excess, disjoint, and matching genes
        excess = 0
        disjoint = 0
        weight_diff = 0.0
        matching = 0

        # Check all innovations in genome1
        for c1 in conn1
            if c1.innovation in innov2
                # Matching gene - compute weight difference
                c2 = conn2[findfirst(c->c.innovation == c1.innovation, conn2)]
                weight_diff += abs(c1.weight - c2.weight)
                matching += 1
            elseif c1.innovation > max_innov2
                # Excess gene (beyond genome2's range)
                excess += 1
            else
                # Disjoint gene (within range but not matching)
                disjoint += 1
            end
        end

        # Check genes only in genome2
        for c2 in conn2
            if !(c2.innovation in innov1)
                if c2.innovation > max_innov1
                    # Excess gene (beyond genome1's range)
                    excess += 1
                else
                    # Disjoint gene (within range but not matching)
                    disjoint += 1
                end
            end
        end

        # Calculate average weight difference
        avg_weight_diff = matching > 0 ? weight_diff / matching : 0.0

        # N = number of genes in larger genome
        N = max(length(conn1), length(conn2))
        N = max(N, 1)  # Avoid division by zero

        # Apply NEAT paper's formula: δ = (c₁·E)/N + (c₂·D)/N + c₃·W̄
        connection_distance = (config.compatibility_excess_coefficient * excess) / N +
                             (config.compatibility_disjoint_coefficient * disjoint) / N +
                             config.compatibility_weight_coefficient * avg_weight_diff

        # Add simple node distance (paper doesn't specify, but needed for completeness)
        node_distance = _node_distance_simple(genome1, genome2, config)

        return connection_distance + node_distance
    else
        # One genome has connections, the other doesn't - all are excess
        N = max(length(conn1), length(conn2))
        connection_distance = (config.compatibility_excess_coefficient * N) / N
        node_distance = _node_distance_simple(genome1, genome2, config)
        return connection_distance + node_distance
    end
end

"""
Helper function for simple node distance calculation.
The NEAT paper doesn't specify how to handle node gene distance,
so we use a simplified approach.
"""
function _node_distance_simple(genome1::Genome, genome2::Genome, config::GenomeConfig)
    if isempty(genome1.nodes) && isempty(genome2.nodes)
        return 0.0
    end

    disjoint_nodes = 0
    weight_diff = 0.0
    matching = 0

    # Count disjoint nodes in genome2
    for k2 in keys(genome2.nodes)
        if !haskey(genome1.nodes, k2)
            disjoint_nodes += 1
        end
    end

    # Count disjoint nodes in genome1 and compute attribute differences
    for (k1, n1) in genome1.nodes
        n2 = get(genome2.nodes, k1, nothing)
        if n2 === nothing
            disjoint_nodes += 1
        else
            # For matching nodes, compute attribute distance
            weight_diff += abs(n1.bias - n2.bias) + abs(n1.response - n2.response)
            if n1.activation != n2.activation
                weight_diff += 1.0
            end
            if n1.aggregation != n2.aggregation
                weight_diff += 1.0
            end
            if !isnan(n1.time_constant) && !isnan(n2.time_constant)
                weight_diff += abs(n1.time_constant - n2.time_constant)
            end
            if !isnan(n1.iz_a) && !isnan(n2.iz_a)
                weight_diff += abs(n1.iz_a - n2.iz_a) + abs(n1.iz_b - n2.iz_b) +
                               abs(n1.iz_c - n2.iz_c) + abs(n1.iz_d - n2.iz_d)
            end
            matching += 1
        end
    end

    avg_weight_diff = matching > 0 ? weight_diff / matching : 0.0
    N = max(length(genome1.nodes), length(genome2.nodes))
    N = max(N, 1)

    # Use disjoint coefficient for nodes (treating all non-matching as disjoint)
    return (config.compatibility_disjoint_coefficient * disjoint_nodes) / N +
           config.compatibility_weight_coefficient * avg_weight_diff
end

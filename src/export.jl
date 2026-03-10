"""
Model export and import functionality for NeatEvolution.jl

Supports exporting evolved networks to JSON format compatible with neat-python v1.0:
- Framework-agnostic model sharing
- Interoperability with neat-python
- Model deployment and storage
"""

using JSON
using Dates

# Format version (compatible with neat-python v1.0)
const FORMAT_VERSION = "1.0"

# Built-in activation functions (from activations.jl)
const BUILTIN_ACTIVATIONS = Set([
    :sigmoid, :tanh, :sin, :gauss, :relu, :elu, :lelu, :selu,
    :softplus, :identity, :clamped, :inv, :log, :exp, :abs,
    :hat, :square, :cube
])

# Built-in aggregation functions (from aggregations.jl)
const BUILTIN_AGGREGATIONS = Set([
    :product, :sum, :max, :min, :maxabs, :median, :mean
])

"""
    is_builtin_activation(name::Symbol) -> Bool

Check if an activation function is built-in to NeatEvolution.jl.
"""
function is_builtin_activation(name::Symbol)
    return name in BUILTIN_ACTIVATIONS
end

"""
    is_builtin_aggregation(name::Symbol) -> Bool

Check if an aggregation function is built-in to NeatEvolution.jl.
"""
function is_builtin_aggregation(name::Symbol)
    return name in BUILTIN_AGGREGATIONS
end

"""
    get_node_type(node_id::Int, config::GenomeConfig) -> String

Determine the type of a node (input, hidden, or output).
"""
function get_node_type(node_id::Int, config::GenomeConfig)
    if node_id in config.input_keys
        return "input"
    elseif node_id in config.output_keys
        return "output"
    else
        return "hidden"
    end
end

"""
    export_network_json(genome::Genome, config::GenomeConfig, filename::String;
                       generation::Union{Nothing, Int}=nothing,
                       metadata::Dict{String, Any}=Dict{String, Any}())

Export a genome's network structure to JSON format compatible with neat-python v1.0.

The JSON format follows the neat-python specification and includes:
- Format version for compatibility tracking
- Network type (feedforward/recurrent)
- All nodes with their properties, type, and custom function flags
- All connections with weights and enabled status
- Topology information (inputs/outputs)
- Rich metadata including timestamp, fitness, generation, etc.

# Arguments
- `genome::Genome`: The genome to export
- `config::GenomeConfig`: Genome configuration containing input/output specifications
- `filename::String`: Path to output JSON file
- `generation::Union{Nothing, Int}=nothing`: Optional generation number
- `metadata::Dict{String, Any}=Dict{String, Any}()`: Optional additional metadata fields

# Example
```julia
using NeatEvolution

config = load_config("config.toml")
# ... run evolution ...
winner = run!(pop, eval_genomes, 100)

# Export to JSON
export_network_json(winner, config.genome_config, "winner_network.json",
                   generation=100, metadata=Dict("problem" => "XOR"))
```

# JSON Format
The output follows the neat-python v1.0 format specification. See:
https://github.com/CodeReclaimers/neat-python/blob/master/docs/network-json-format.md
"""
function export_network_json(genome::Genome, config::GenomeConfig, filename::String;
                             generation::Union{Nothing, Int}=nothing,
                             metadata::Dict=Dict{String, Any}())
    # Build nodes array (must include input nodes, output nodes, and hidden nodes)
    nodes_array = []

    # Add input nodes
    for input_id in config.input_keys
        node_dict = Dict(
            "id" => input_id,
            "type" => "input",
            "activation" => Dict(
                "name" => "identity",
                "custom" => false
            ),
            "aggregation" => Dict(
                "name" => "none",
                "custom" => false
            ),
            "bias" => 0.0,
            "response" => 1.0
        )
        push!(nodes_array, node_dict)
    end

    # Add output and hidden nodes from genome
    for (node_id, node) in genome.nodes
        node_type = get_node_type(node_id, config)
        node_dict = Dict(
            "id" => node_id,
            "type" => node_type,
            "activation" => Dict(
                "name" => string(node.activation),
                "custom" => !is_builtin_activation(node.activation)
            ),
            "aggregation" => Dict(
                "name" => string(node.aggregation),
                "custom" => !is_builtin_aggregation(node.aggregation)
            ),
            "bias" => node.bias,
            "response" => node.response
        )
        # Conditionally add CTRNN time_constant
        if !isnan(node.time_constant)
            node_dict["time_constant"] = node.time_constant
        end
        # Conditionally add Izhikevich parameters
        if !isnan(node.iz_a)
            node_dict["izhikevich"] = Dict(
                "iz_a" => node.iz_a,
                "iz_b" => node.iz_b,
                "iz_c" => node.iz_c,
                "iz_d" => node.iz_d
            )
        end
        push!(nodes_array, node_dict)
    end

    # Build connections array
    connections_array = []
    for (conn_key, conn) in genome.connections
        conn_dict = Dict(
            "from" => conn_key[1],
            "to" => conn_key[2],
            "weight" => conn.weight,
            "enabled" => conn.enabled
        )
        push!(connections_array, conn_dict)
    end

    # Build topology
    topology = Dict(
        "num_inputs" => config.num_inputs,
        "num_outputs" => config.num_outputs,
        "input_keys" => collect(config.input_keys),
        "output_keys" => collect(config.output_keys)
    )

    # Build metadata (merge provided metadata with standard fields)
    meta = Dict{String, Any}(
        "created_timestamp" => string(Dates.now(Dates.UTC)) * "Z",
        "neat_julia_version" => string(VERSION),
        "genome_id" => genome.key
    )

    # Add optional fields
    if genome.fitness !== nothing
        meta["fitness"] = genome.fitness
    end
    if generation !== nothing
        meta["generation"] = generation
    end

    # Merge user-provided metadata
    merge!(meta, metadata)

    # Determine network type
    network_type = config.feed_forward ? "feedforward" : "recurrent"

    # Build complete network structure
    network_data = Dict(
        "format_version" => FORMAT_VERSION,
        "network_type" => network_type,
        "metadata" => meta,
        "topology" => topology,
        "nodes" => nodes_array,
        "connections" => connections_array
    )

    # Write to file with pretty formatting
    open(filename, "w") do io
        JSON.print(io, network_data, 4)
    end

    @info "Network exported" filename format_version=FORMAT_VERSION
end

"""
    import_network_json(filename::String, config::GenomeConfig) -> Genome

Import a genome from JSON format (supports both neat-python v1.0 and legacy NeatEvolution.jl formats).

Loads a network previously exported with `export_network_json` and reconstructs
the Genome object with all nodes and connections.

# Arguments
- `filename::String`: Path to JSON file containing network data
- `config::GenomeConfig`: Genome configuration (must match exported network's config)

# Returns
- `Genome`: Reconstructed genome object

# Example
```julia
using NeatEvolution

config = load_config("config.toml")

# Import previously exported network
imported_genome = import_network_json("winner_network.json", config.genome_config)

# Use the imported network
net = FeedForwardNetwork(imported_genome, config.genome_config)
output = activate!(net, [1.0, 0.0])
```
"""
function import_network_json(filename::String, config::GenomeConfig)
    # Read JSON file
    network_data = JSON.parsefile(filename)

    # Detect format version
    if haskey(network_data, "format_version")
        # New format (neat-python v1.0 compatible)
        return import_network_json_v1(network_data, config)
    else
        # Legacy NeatEvolution.jl format
        return import_network_json_legacy(network_data, config)
    end
end

"""
    import_network_json_v1(network_data::AbstractDict, config::GenomeConfig) -> Genome

Import a genome from neat-python v1.0 format.
"""
function import_network_json_v1(network_data::AbstractDict, config::GenomeConfig)
    # Extract metadata
    metadata = get(network_data, "metadata", Dict())
    genome_id = get(metadata, "genome_id", 1)
    fitness = get(metadata, "fitness", nothing)

    # Create genome
    genome = Genome(genome_id)
    genome.fitness = fitness

    # Reconstruct nodes (excluding input nodes)
    nodes_array = network_data["nodes"]
    for node_data in nodes_array
        node_id = node_data["id"]
        node_type = node_data["type"]

        # Skip input nodes (they are not stored in genome.nodes in NeatEvolution.jl)
        if node_type == "input"
            continue
        end

        node = NodeGene(node_id)
        node.bias = node_data["bias"]
        node.response = node_data["response"]

        # Extract activation function name
        if isa(node_data["activation"], AbstractDict)
            node.activation = Symbol(node_data["activation"]["name"])
        else
            node.activation = Symbol(node_data["activation"])
        end

        # Extract aggregation function name
        if isa(node_data["aggregation"], AbstractDict)
            agg_name = node_data["aggregation"]["name"]
            node.aggregation = agg_name == "none" ? :sum : Symbol(agg_name)
        else
            node.aggregation = Symbol(node_data["aggregation"])
        end

        # Import CTRNN time_constant if present
        if haskey(node_data, "time_constant")
            node.time_constant = node_data["time_constant"]
        end

        # Import Izhikevich parameters if present
        if haskey(node_data, "izhikevich")
            iz = node_data["izhikevich"]
            node.iz_a = iz["iz_a"]
            node.iz_b = iz["iz_b"]
            node.iz_c = iz["iz_c"]
            node.iz_d = iz["iz_d"]
        end

        genome.nodes[node_id] = node
    end

    # Reconstruct connections
    connections_array = network_data["connections"]
    for conn_data in connections_array
        input_id = conn_data["from"]
        output_id = conn_data["to"]
        conn_key = (input_id, output_id)

        # Innovation number is not in neat-python format, use a counter
        # The innovation system will be rebuilt if needed
        innovation = get(conn_data, "innovation", 0)
        conn = ConnectionGene(conn_key, innovation)
        conn.weight = conn_data["weight"]
        conn.enabled = conn_data["enabled"]
        genome.connections[conn_key] = conn
    end

    @info "Network imported from neat-python v1.0 format"
    return genome
end

"""
    import_network_json_legacy(network_data::AbstractDict, config::GenomeConfig) -> Genome

Import a genome from legacy NeatEvolution.jl format (pre-v1.0).
"""
function import_network_json_legacy(network_data::AbstractDict, config::GenomeConfig)
    # Extract metadata
    genome_id = get(get(network_data, "metadata", Dict()), "genome_id", 1)
    fitness = get(network_data, "fitness", nothing)

    # Create genome
    genome = Genome(genome_id)
    genome.fitness = fitness

    # Reconstruct nodes
    nodes_data = network_data["nodes"]
    for (node_id_str, node_data) in nodes_data
        node_id = parse(Int, node_id_str)
        node = NodeGene(node_id)
        node.bias = node_data["bias"]
        node.response = node_data["response"]
        node.activation = Symbol(node_data["activation"])
        node.aggregation = Symbol(node_data["aggregation"])
        genome.nodes[node_id] = node
    end

    # Reconstruct connections
    connections_data = network_data["connections"]
    for (conn_key_str, conn_data) in connections_data
        # Parse key string "(input, output)" back to tuple
        cleaned = replace(conn_key_str, "(" => "", ")" => "")
        parts = split(cleaned, ",")
        input_id = parse(Int, strip(parts[1]))
        output_id = parse(Int, strip(parts[2]))
        conn_key = (input_id, output_id)

        conn = ConnectionGene(conn_key, conn_data["innovation"])
        conn.weight = conn_data["weight"]
        conn.enabled = conn_data["enabled"]
        genome.connections[conn_key] = conn
    end

    @info "Network imported from legacy NeatEvolution.jl format"
    return genome
end

"""
    export_population_json(population::Dict{Int, Genome}, config::GenomeConfig,
                           filename::String; top_n::Union{Nothing, Int}=nothing,
                           generation::Union{Nothing, Int}=nothing)

Export multiple genomes from a population to JSON format (neat-python v1.0 compatible).

Useful for saving entire populations or the top N genomes for analysis.
Each genome is exported in the same format as `export_network_json`.

# Arguments
- `population::Dict{Int, Genome}`: Population dictionary (genome_id => genome)
- `config::GenomeConfig`: Genome configuration
- `filename::String`: Path to output JSON file
- `top_n::Union{Nothing, Int}=nothing`: If specified, only export top N genomes by fitness
- `generation::Union{Nothing, Int}=nothing`: Optional generation number to include in metadata

# Example
```julia
# Export top 10 genomes from final population
export_population_json(pop.population, config.genome_config,
                       "top10.json", top_n=10, generation=100)
```
"""
function export_population_json(population::Dict{Int, Genome},
                                config::GenomeConfig,
                                filename::String;
                                top_n::Union{Nothing, Int}=nothing,
                                generation::Union{Nothing, Int}=nothing)
    # Sort genomes by fitness
    sorted_genomes = sort(collect(values(population)),
                         by = g -> g.fitness === nothing ? -Inf : g.fitness,
                         rev = true)

    # Select top N if specified
    if top_n !== nothing
        sorted_genomes = sorted_genomes[1:min(top_n, length(sorted_genomes))]
    end

    # Export each genome using the v1.0 format
    genomes_data = []
    for genome in sorted_genomes
        # Build genome data structure similar to export_network_json
        nodes_array = []

        # Add input nodes
        for input_id in config.input_keys
            node_dict = Dict(
                "id" => input_id,
                "type" => "input",
                "activation" => Dict("name" => "identity", "custom" => false),
                "aggregation" => Dict("name" => "none", "custom" => false),
                "bias" => 0.0,
                "response" => 1.0
            )
            push!(nodes_array, node_dict)
        end

        # Add output and hidden nodes
        for (node_id, node) in genome.nodes
            node_type = get_node_type(node_id, config)
            node_dict = Dict(
                "id" => node_id,
                "type" => node_type,
                "activation" => Dict(
                    "name" => string(node.activation),
                    "custom" => !is_builtin_activation(node.activation)
                ),
                "aggregation" => Dict(
                    "name" => string(node.aggregation),
                    "custom" => !is_builtin_aggregation(node.aggregation)
                ),
                "bias" => node.bias,
                "response" => node.response
            )
            if !isnan(node.time_constant)
                node_dict["time_constant"] = node.time_constant
            end
            if !isnan(node.iz_a)
                node_dict["izhikevich"] = Dict(
                    "iz_a" => node.iz_a,
                    "iz_b" => node.iz_b,
                    "iz_c" => node.iz_c,
                    "iz_d" => node.iz_d
                )
            end
            push!(nodes_array, node_dict)
        end

        # Build connections array
        connections_array = []
        for (conn_key, conn) in genome.connections
            conn_dict = Dict(
                "from" => conn_key[1],
                "to" => conn_key[2],
                "weight" => conn.weight,
                "enabled" => conn.enabled
            )
            push!(connections_array, conn_dict)
        end

        # Build metadata
        meta = Dict{String, Any}(
            "genome_id" => genome.key,
            "created_timestamp" => string(Dates.now(Dates.UTC)) * "Z"
        )
        if genome.fitness !== nothing
            meta["fitness"] = genome.fitness
        end
        if generation !== nothing
            meta["generation"] = generation
        end

        # Build genome entry
        genome_dict = Dict(
            "format_version" => FORMAT_VERSION,
            "network_type" => config.feed_forward ? "feedforward" : "recurrent",
            "metadata" => meta,
            "topology" => Dict(
                "num_inputs" => config.num_inputs,
                "num_outputs" => config.num_outputs,
                "input_keys" => collect(config.input_keys),
                "output_keys" => collect(config.output_keys)
            ),
            "nodes" => nodes_array,
            "connections" => connections_array
        )
        push!(genomes_data, genome_dict)
    end

    # Create output structure
    output_data = Dict(
        "format_version" => FORMAT_VERSION,
        "genomes" => genomes_data,
        "count" => length(genomes_data)
    )

    # Write to file
    open(filename, "w") do io
        JSON.print(io, output_data, 4)
    end

    @info "Population exported" num_genomes=length(genomes_data) filename format_version=FORMAT_VERSION
end

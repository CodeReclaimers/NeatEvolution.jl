"""
Model export and import functionality for NEAT.jl

Supports exporting evolved networks to JSON format for:
- Framework-agnostic model sharing
- Interoperability with other ML frameworks
- Model deployment and storage
"""

using JSON

"""
    export_network_json(genome::Genome, config::GenomeConfig, filename::String)

Export a genome's network structure to JSON format.

The JSON format includes:
- All nodes with their properties (bias, response, activation, aggregation)
- All connections with weights and enabled status
- Input and output node specifications
- Network metadata (feed-forward flag)

# Arguments
- `genome::Genome`: The genome to export
- `config::GenomeConfig`: Genome configuration containing input/output specifications
- `filename::String`: Path to output JSON file

# Example
```julia
using NEAT

config = load_config("config.toml")
# ... run evolution ...
winner = run!(pop, eval_genomes, 100)

# Export to JSON
export_network_json(winner, config.genome_config, "winner_network.json")
```

# JSON Format
```json
{
  "nodes": {
    "-1": {"bias": 0.0, "response": 1.0, "activation": "sigmoid", "aggregation": "sum"},
    "0": {"bias": -0.5, "response": 1.0, "activation": "sigmoid", "aggregation": "sum"}
  },
  "connections": {
    "(-1, 0)": {"weight": 2.5, "enabled": true, "innovation": 1}
  },
  "inputs": [-1, -2],
  "outputs": [0],
  "fitness": 3.95,
  "feed_forward": true
}
```
"""
function export_network_json(genome::Genome, config::GenomeConfig, filename::String)
    # Build nodes dictionary
    nodes_dict = Dict{String, Any}()
    for (node_id, node) in genome.nodes
        nodes_dict[string(node_id)] = Dict(
            "bias" => node.bias,
            "response" => node.response,
            "activation" => string(node.activation),
            "aggregation" => string(node.aggregation)
        )
    end

    # Build connections dictionary
    connections_dict = Dict{String, Any}()
    for (conn_key, conn) in genome.connections
        key_str = string(conn_key)
        connections_dict[key_str] = Dict(
            "weight" => conn.weight,
            "enabled" => conn.enabled,
            "innovation" => conn.innovation
        )
    end

    # Build complete network structure
    network_data = Dict(
        "nodes" => nodes_dict,
        "connections" => connections_dict,
        "inputs" => collect(config.input_keys),
        "outputs" => collect(config.output_keys),
        "fitness" => genome.fitness === nothing ? 0.0 : genome.fitness,
        "feed_forward" => config.feed_forward,
        "metadata" => Dict(
            "num_inputs" => config.num_inputs,
            "num_outputs" => config.num_outputs,
            "genome_id" => genome.key
        )
    )

    # Write to file with pretty formatting
    open(filename, "w") do io
        JSON.print(io, network_data, 4)
    end

    println("Network exported to $filename")
end

"""
    import_network_json(filename::String, config::GenomeConfig) -> Genome

Import a genome from JSON format.

Loads a network previously exported with `export_network_json` and reconstructs
the Genome object with all nodes and connections.

# Arguments
- `filename::String`: Path to JSON file containing network data
- `config::GenomeConfig`: Genome configuration (must match exported network's config)

# Returns
- `Genome`: Reconstructed genome object

# Example
```julia
using NEAT

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

    println("Network imported from $filename")
    return genome
end

"""
    export_population_json(population::Dict{Int, Genome}, config::GenomeConfig,
                           filename::String; top_n::Union{Nothing, Int}=nothing)

Export multiple genomes from a population to JSON format.

Useful for saving entire populations or the top N genomes for analysis.

# Arguments
- `population::Dict{Int, Genome}`: Population dictionary (genome_id => genome)
- `config::GenomeConfig`: Genome configuration
- `filename::String`: Path to output JSON file
- `top_n::Union{Nothing, Int}=nothing`: If specified, only export top N genomes by fitness

# Example
```julia
# Export top 10 genomes from final population
export_population_json(pop.population, config.genome_config,
                       "top10.json", top_n=10)
```
"""
function export_population_json(population::Dict{Int, Genome},
                                config::GenomeConfig,
                                filename::String;
                                top_n::Union{Nothing, Int}=nothing)
    # Sort genomes by fitness
    sorted_genomes = sort(collect(values(population)),
                         by = g -> g.fitness === nothing ? -Inf : g.fitness,
                         rev = true)

    # Select top N if specified
    if top_n !== nothing
        sorted_genomes = sorted_genomes[1:min(top_n, length(sorted_genomes))]
    end

    # Export each genome to a dictionary
    genomes_data = []
    for genome in sorted_genomes
        genome_dict = Dict(
            "genome_id" => genome.key,
            "fitness" => genome.fitness,
            "num_nodes" => length(genome.nodes),
            "num_connections" => length(genome.connections),
            "nodes" => Dict(
                string(id) => Dict(
                    "bias" => n.bias,
                    "response" => n.response,
                    "activation" => string(n.activation),
                    "aggregation" => string(n.aggregation)
                ) for (id, n) in genome.nodes
            ),
            "connections" => Dict(
                string(k) => Dict(
                    "weight" => c.weight,
                    "enabled" => c.enabled,
                    "innovation" => c.innovation
                ) for (k, c) in genome.connections
            )
        )
        push!(genomes_data, genome_dict)
    end

    # Create output structure
    output_data = Dict(
        "genomes" => genomes_data,
        "count" => length(genomes_data),
        "inputs" => collect(config.input_keys),
        "outputs" => collect(config.output_keys),
        "feed_forward" => config.feed_forward
    )

    # Write to file
    open(filename, "w") do io
        JSON.print(io, output_data, 4)
    end

    println("Exported $(length(genomes_data)) genomes to $filename")
end

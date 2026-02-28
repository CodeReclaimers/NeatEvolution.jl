"""
Configuration system for NEAT.

Handles loading and parsing configuration files (TOML format).
"""

using TOML

"""
GenomeConfig holds configuration for genome structure and mutation.
"""
struct GenomeConfig
    # Network structure
    num_inputs::Int
    num_outputs::Int
    num_hidden::Int

    # Input/output keys
    input_keys::Vector{Int}
    output_keys::Vector{Int}

    # Compatibility coefficients (per NEAT paper Equation 1)
    # δ = (c₁·E)/N + (c₂·D)/N + c₃·W̄
    compatibility_excess_coefficient::Float64       # c₁ - coefficient for excess genes
    compatibility_disjoint_coefficient::Float64     # c₂ - coefficient for disjoint genes
    compatibility_weight_coefficient::Float64       # c₃ - coefficient for weight differences

    # Structural mutation probabilities
    conn_add_prob::Float64
    conn_delete_prob::Float64
    node_add_prob::Float64
    node_delete_prob::Float64

    # Mutation settings
    single_structural_mutation::Bool
    feed_forward::Bool
    initial_connection::Symbol
    connection_fraction::Float64

    # Gene attributes
    bias_attr::FloatAttribute
    response_attr::FloatAttribute
    activation_attr::StringAttribute
    aggregation_attr::StringAttribute
    weight_attr::FloatAttribute
    enabled_attr::BoolAttribute

    # Optional CTRNN/IZNN attributes (nothing when not configured)
    time_constant_attr::Union{FloatAttribute, Nothing}
    iz_a_attr::Union{FloatAttribute, Nothing}
    iz_b_attr::Union{FloatAttribute, Nothing}
    iz_c_attr::Union{FloatAttribute, Nothing}
    iz_d_attr::Union{FloatAttribute, Nothing}

    # Node indexer for generating new node IDs
    node_indexer::Ref{Int}

    # Innovation tracking for connection genes (per NEAT paper)
    # innovation_indexer: next innovation number to assign
    # innovation_cache: maps (from_node, to_node) -> innovation for current generation
    innovation_indexer::Ref{Int}
    innovation_cache::Ref{Dict{Tuple{Int,Int}, Int}}
end

function GenomeConfig(params::Dict)
    num_inputs = get(params, :num_inputs, 2)
    num_outputs = get(params, :num_outputs, 1)

    # By convention, input keys are negative, output keys are 0, 1, 2, ...
    input_keys = collect(-num_inputs:-1)
    output_keys = collect(0:num_outputs-1)

    # Create attributes
    bias_attr = FloatAttribute(:bias, params)
    response_attr = FloatAttribute(:response, params)
    activation_attr = StringAttribute(:activation, params)
    aggregation_attr = StringAttribute(:aggregation, params)
    weight_attr = FloatAttribute(:weight, params)
    enabled_attr = BoolAttribute(:enabled, params)

    # Validate attributes
    validate(bias_attr)
    validate(response_attr)
    validate(activation_attr)
    validate(aggregation_attr)
    validate(weight_attr)

    # Optional CTRNN attribute (created when time_constant_init_mean is present)
    time_constant_attr = if haskey(params, :time_constant_init_mean)
        tc = FloatAttribute(:time_constant, params)
        validate(tc)
        tc
    else
        nothing
    end

    # Optional Izhikevich attributes (created when iz_a_init_mean is present)
    iz_a_attr, iz_b_attr, iz_c_attr, iz_d_attr = if haskey(params, :iz_a_init_mean)
        a = FloatAttribute(:iz_a, params)
        b = FloatAttribute(:iz_b, params)
        c = FloatAttribute(:iz_c, params)
        d = FloatAttribute(:iz_d, params)
        validate(a); validate(b); validate(c); validate(d)
        (a, b, c, d)
    else
        (nothing, nothing, nothing, nothing)
    end

    # Determine starting node indexer value
    max_node_id = isempty(output_keys) ? 0 : maximum(output_keys)

    GenomeConfig(
        num_inputs,
        num_outputs,
        get(params, :num_hidden, 0),
        input_keys,
        output_keys,
        get(params, :compatibility_excess_coefficient, 1.0),     # c₁ per NEAT paper
        get(params, :compatibility_disjoint_coefficient, 1.0),   # c₂ per NEAT paper
        get(params, :compatibility_weight_coefficient, 0.4),     # c₃ per NEAT paper
        get(params, :conn_add_prob, 0.5),
        get(params, :conn_delete_prob, 0.5),
        get(params, :node_add_prob, 0.2),
        get(params, :node_delete_prob, 0.2),
        get(params, :single_structural_mutation, false),
        get(params, :feed_forward, true),
        begin
            ic = get(params, :initial_connection, "full")
            isa(ic, Symbol) ? ic : Symbol(lowercase(ic))
        end,
        get(params, :connection_fraction, 0.5),
        bias_attr,
        response_attr,
        activation_attr,
        aggregation_attr,
        weight_attr,
        enabled_attr,
        time_constant_attr,
        iz_a_attr,
        iz_b_attr,
        iz_c_attr,
        iz_d_attr,
        Ref(max_node_id + 1),
        Ref(0),  # innovation_indexer starts at 0
        Ref(Dict{Tuple{Int,Int}, Int}())  # innovation_cache starts empty
    )
end

function get_new_node_id!(config::GenomeConfig)
    id = config.node_indexer[]
    config.node_indexer[] += 1
    return id
end

"""
Get or create an innovation number for a connection.

Per NEAT paper: if the same structural mutation occurs multiple times
in a generation, it should receive the same innovation number.
"""
function get_innovation!(config::GenomeConfig, connection_key::Tuple{Int,Int})
    cache = config.innovation_cache[]
    if haskey(cache, connection_key)
        return cache[connection_key]
    else
        innovation = config.innovation_indexer[]
        config.innovation_indexer[] += 1
        cache[connection_key] = innovation
        return innovation
    end
end

"""
Reset the innovation cache at generation boundaries.

Per NEAT paper: innovation numbers persist across generations,
but the cache of connection_key -> innovation mappings is reset
each generation to allow the same structural mutations to be
detected within a generation.
"""
function reset_innovation_cache!(config::GenomeConfig)
    config.innovation_cache[] = Dict{Tuple{Int,Int}, Int}()
end

"""
SpeciesConfig holds configuration for speciation.
"""
struct SpeciesConfig
    compatibility_threshold::Float64
end

function SpeciesConfig(params::Dict)
    SpeciesConfig(get(params, :compatibility_threshold, 3.0))
end

"""
StagnationConfig holds configuration for stagnation detection.
"""
struct StagnationConfig
    species_fitness_func::Symbol
    max_stagnation::Int
    species_elitism::Int
end

function StagnationConfig(params::Dict)
    StagnationConfig(
        Symbol(lowercase(get(params, :species_fitness_func, "mean"))),
        get(params, :max_stagnation, 15),
        get(params, :species_elitism, 0)
    )
end

"""
ReproductionConfig holds configuration for reproduction.
"""
struct ReproductionConfig
    elitism::Int
    survival_threshold::Float64
    min_species_size::Int
end

function ReproductionConfig(params::Dict)
    ReproductionConfig(
        get(params, :elitism, 0),
        get(params, :survival_threshold, 0.2),
        get(params, :min_species_size, 1)
    )
end

"""
Config is the main configuration object for a NEAT run.
"""
struct Config
    pop_size::Int
    fitness_criterion::Symbol
    fitness_threshold::Float64
    reset_on_extinction::Bool
    no_fitness_termination::Bool

    genome_config::GenomeConfig
    species_config::SpeciesConfig
    stagnation_config::StagnationConfig
    reproduction_config::ReproductionConfig
end

"""
Construct Config from individual parameters and component configs.
"""
function Config(params::Dict, genome_config::GenomeConfig, species_config::SpeciesConfig,
                stagnation_config::StagnationConfig, reproduction_config::ReproductionConfig)
    Config(
        get(params, :pop_size, 150),
        get(params, :fitness_criterion, :max),
        get(params, :fitness_threshold, 3.9),
        get(params, :reset_on_extinction, false),
        get(params, :no_fitness_termination, false),
        genome_config,
        species_config,
        stagnation_config,
        reproduction_config
    )
end

"""
Load a NEAT configuration from a TOML file.
"""
function load_config(filename::String)
    data = TOML.parsefile(filename)

    # Convert string keys to symbols for easier access
    function symbolize_keys(d::Dict)
        Dict(Symbol(lowercase(string(k))) => (v isa Dict ? symbolize_keys(v) : v) for (k, v) in d)
    end

    config_data = symbolize_keys(data)

    # Validate configuration with enhanced error checking
    validate_config(config_data)

    # Extract main NEAT parameters
    neat_params = get(config_data, :neat, Dict())

    # Extract section-specific parameters
    genome_params = get(config_data, :defaultgenome, Dict())
    species_params = get(config_data, :defaultspeciesset, Dict())
    stagnation_params = get(config_data, :defaultstagnation, Dict())
    reproduction_params = get(config_data, :defaultreproduction, Dict())

    # Merge genome_params into a single dict for attribute creation
    # (attributes look for keys like :bias_init_mean, etc.)

    Config(
        get(neat_params, :pop_size, 150),
        Symbol(lowercase(get(neat_params, :fitness_criterion, "max"))),
        get(neat_params, :fitness_threshold, 3.9),
        get(neat_params, :reset_on_extinction, false),
        get(neat_params, :no_fitness_termination, false),
        GenomeConfig(genome_params),
        SpeciesConfig(species_params),
        StagnationConfig(stagnation_params),
        ReproductionConfig(reproduction_params)
    )
end

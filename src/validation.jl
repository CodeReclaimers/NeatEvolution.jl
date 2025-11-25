"""
Enhanced configuration validation for NEAT.jl

Provides stricter validation with helpful error messages,
typo detection, and suggestions for correct values.
"""

"""
Known parameter names for each configuration section.
Used to detect typos and unknown parameters.
"""
const KNOWN_NEAT_PARAMS = Set([
    :pop_size,
    :fitness_criterion,
    :fitness_threshold,
    :reset_on_extinction,
    :no_fitness_termination
])

const KNOWN_GENOME_PARAMS = Set([
    :num_inputs,
    :num_outputs,
    :num_hidden,
    :feed_forward,
    :initial_connection,
    :connection_fraction,
    :single_structural_mutation,
    # Compatibility coefficients
    :compatibility_excess_coefficient,
    :compatibility_disjoint_coefficient,
    :compatibility_weight_coefficient,
    # Structural mutation
    :conn_add_prob,
    :conn_delete_prob,
    :node_add_prob,
    :node_delete_prob,
    # Bias attributes
    :bias_init_mean,
    :bias_init_stdev,
    :bias_init_type,
    :bias_replace_rate,
    :bias_mutate_rate,
    :bias_mutate_power,
    :bias_min_value,
    :bias_max_value,
    # Response attributes
    :response_init_mean,
    :response_init_stdev,
    :response_init_type,
    :response_replace_rate,
    :response_mutate_rate,
    :response_mutate_power,
    :response_min_value,
    :response_max_value,
    # Weight attributes
    :weight_init_mean,
    :weight_init_stdev,
    :weight_init_type,
    :weight_replace_rate,
    :weight_mutate_rate,
    :weight_mutate_power,
    :weight_min_value,
    :weight_max_value,
    # Activation attributes
    :activation_default,
    :activation_options,
    :activation_mutate_rate,
    # Aggregation attributes
    :aggregation_default,
    :aggregation_options,
    :aggregation_mutate_rate,
    # Enabled attributes
    :enabled_default,
    :enabled_mutate_rate,
    :enabled_rate_to_true_add,
    :enabled_rate_to_false_add
])

const KNOWN_SPECIES_PARAMS = Set([
    :compatibility_threshold
])

const KNOWN_STAGNATION_PARAMS = Set([
    :species_fitness_func,
    :max_stagnation,
    :species_elitism
])

const KNOWN_REPRODUCTION_PARAMS = Set([
    :elitism,
    :survival_threshold,
    :min_species_size
])

"""
Common typos and their corrections.
"""
const TYPO_CORRECTIONS = Dict(
    # Common misspellings
    :compatability_threshold => :compatibility_threshold,
    :compatability_excess_coefficient => :compatibility_excess_coefficient,
    :compatability_disjoint_coefficient => :compatibility_disjoint_coefficient,
    :compatability_weight_coefficient => :compatibility_weight_coefficient,
    :aggregration_default => :aggregation_default,
    :aggregration_options => :aggregation_options,
    :aggreagation_default => :aggregation_default,
    :aggreagation_options => :aggregation_options,
    # Underscore variations
    :popsize => :pop_size,
    :numInputs => :num_inputs,
    :numOutputs => :num_outputs,
    :numHidden => :num_hidden,
    :feedforward => :feed_forward,
    :feedForward => :feed_forward,
    # Different naming conventions
    :population_size => :pop_size,
    :input_count => :num_inputs,
    :output_count => :num_outputs,
    :hidden_count => :num_hidden,
    :max_generations => :fitness_threshold,
    # Coefficient variations
    :c1 => :compatibility_excess_coefficient,
    :c2 => :compatibility_disjoint_coefficient,
    :c3 => :compatibility_weight_coefficient,
    :excess_coefficient => :compatibility_excess_coefficient,
    :disjoint_coefficient => :compatibility_disjoint_coefficient,
    :weight_coefficient => :compatibility_weight_coefficient
)

"""
Calculate string edit distance (Levenshtein distance) for typo detection.
"""
function edit_distance(s1::String, s2::String)
    m, n = length(s1), length(s2)
    d = zeros(Int, m + 1, n + 1)

    for i in 0:m
        d[i+1, 1] = i
    end
    for j in 0:n
        d[1, j+1] = j
    end

    for j in 1:n
        for i in 1:m
            cost = s1[i] == s2[j] ? 0 : 1
            d[i+1, j+1] = min(
                d[i, j+1] + 1,      # deletion
                d[i+1, j] + 1,      # insertion
                d[i, j] + cost      # substitution
            )
        end
    end

    return d[m+1, n+1]
end

"""
Find similar parameter names for suggestions.
"""
function find_similar_params(param::Symbol, known_params::Set{Symbol})
    param_str = string(param)
    suggestions = String[]

    for known in known_params
        known_str = string(known)
        dist = edit_distance(param_str, known_str)

        # Suggest if edit distance is small (likely typo)
        if dist <= 3
            push!(suggestions, known_str)
        end
    end

    return suggestions
end

"""
Check for unknown parameters and suggest corrections.
"""
function check_unknown_parameters(section_name::String, params::Dict, known_params::Set{Symbol})
    unknown_params = Symbol[]

    for key in keys(params)
        # Skip if it's a known parameter
        if key in known_params
            continue
        end

        # Check if it's a typo we can auto-correct
        if haskey(TYPO_CORRECTIONS, key)
            corrected = TYPO_CORRECTIONS[key]
            @warn """Unknown parameter '$key' in [$section_name].
                     Did you mean '$corrected'?
                     This parameter will be ignored."""
            continue
        end

        # Find similar parameters to suggest
        suggestions = find_similar_params(key, known_params)

        if !isempty(suggestions)
            suggestions_str = join(suggestions, ", ")
            @warn """Unknown parameter '$key' in [$section_name].
                     Did you mean one of: $suggestions_str?
                     This parameter will be ignored."""
        else
            @warn """Unknown parameter '$key' in [$section_name].
                     This parameter will be ignored.
                     See documentation for valid parameters."""
        end

        push!(unknown_params, key)
    end

    return unknown_params
end

"""
Validate NEAT section parameters.
"""
function validate_neat_params(params::Dict)
    check_unknown_parameters("NEAT", params, KNOWN_NEAT_PARAMS)

    # Validate value ranges
    if haskey(params, :pop_size)
        pop_size = params[:pop_size]
        if pop_size < 10
            @warn """Population size ($pop_size) is very small.
                     Recommended minimum: 50 for simple problems, 150+ for complex problems.
                     Small populations may lead to premature convergence or extinction."""
        elseif pop_size > 10000
            @warn """Population size ($pop_size) is very large.
                     This will significantly slow down evolution.
                     Consider using 150-1000 unless you have specific reasons for larger populations."""
        end
    end

    # Validate fitness criterion
    if haskey(params, :fitness_criterion)
        criterion = Symbol(lowercase(string(params[:fitness_criterion])))
        valid_criteria = [:max, :min, :mean]
        if !(criterion in valid_criteria)
            error("""Invalid fitness_criterion: '$(params[:fitness_criterion])'
                     Valid options: $(join(valid_criteria, ", "))
                     Use 'max' (maximize fitness) or 'min' (minimize fitness).""")
        end
    end
end

"""
Validate genome section parameters.
"""
function validate_genome_params(params::Dict)
    check_unknown_parameters("DefaultGenome", params, KNOWN_GENOME_PARAMS)

    # Check required parameters
    if !haskey(params, :num_inputs)
        @warn """Missing 'num_inputs' in [DefaultGenome] section.
                 Using default value of 2.
                 It's recommended to explicitly specify num_inputs for clarity."""
    end

    if !haskey(params, :num_outputs)
        @warn """Missing 'num_outputs' in [DefaultGenome] section.
                 Using default value of 1.
                 It's recommended to explicitly specify num_outputs for clarity."""
    end

    # Validate initial_connection
    if haskey(params, :initial_connection)
        ic = Symbol(lowercase(string(params[:initial_connection])))
        valid_options = [:full, :full_direct, :full_nodirect, :partial, :partial_direct,
                        :unconnected, :fs_neat, :fs_neat_nohidden, :fs_neat_hidden]
        if !(ic in valid_options)
            error("""Invalid initial_connection: '$(params[:initial_connection])'
                     Valid options: $(join(valid_options, ", "))

                     Recommended:
                     - 'full' or 'full_nodirect': Fully connected (no direct input→output if hidden nodes exist)
                     - 'full_direct': Fully connected including direct input→output connections
                     - 'partial': Randomly connected based on connection_fraction
                     - 'partial_direct': Partial with direct input→output allowed
                     - 'fs_neat' or 'fs_neat_nohidden': FS-NEAT style (inputs directly to outputs)
                     - 'fs_neat_hidden': FS-NEAT through hidden layer (inputs→hidden→outputs)
                     - 'unconnected': Start with no connections""")
        end
    end

    # Validate mutation probabilities
    for param in [:conn_add_prob, :conn_delete_prob, :node_add_prob, :node_delete_prob]
        if haskey(params, param)
            prob = params[param]
            if prob < 0.0 || prob > 1.0
                error("""Invalid $param: $prob
                         Mutation probabilities must be in [0.0, 1.0].""")
            end
        end
    end

    # Warn about mutation balance
    if haskey(params, :conn_add_prob) && haskey(params, :conn_delete_prob)
        add_prob = params[:conn_add_prob]
        del_prob = params[:conn_delete_prob]
        if add_prob > del_prob * 5
            @warn """Connection addition rate ($add_prob) is much higher than deletion rate ($del_prob).
                     This may lead to network complexity explosion.
                     Recommended: Keep addition/deletion rates balanced (e.g., add=0.5, delete=0.2)."""
        end
    end

    # Validate compatibility coefficients
    for coef in [:compatibility_excess_coefficient, :compatibility_disjoint_coefficient,
                 :compatibility_weight_coefficient]
        if haskey(params, coef)
            val = params[coef]
            if val < 0.0
                error("""Invalid $coef: $val
                         Compatibility coefficients must be non-negative.

                         NEAT paper default values:
                         - compatibility_excess_coefficient = 1.0
                         - compatibility_disjoint_coefficient = 1.0
                         - compatibility_weight_coefficient = 0.4""")
            end
        end
    end

    # Validate activation options
    if haskey(params, :activation_options)
        options = params[:activation_options]
        if options isa Vector && isempty(options)
            error("""activation_options cannot be empty.
                     Must specify at least one activation function.

                     Common options:
                     - sigmoid (output in [0, 1])
                     - tanh (output in [-1, 1])
                     - relu (output in [0, ∞))
                     - identity (linear)

                     Example: activation_options = ["sigmoid", "tanh", "relu"]""")
        end
    end
end

"""
Validate species section parameters.
"""
function validate_species_params(params::Dict)
    check_unknown_parameters("DefaultSpeciesSet", params, KNOWN_SPECIES_PARAMS)

    if haskey(params, :compatibility_threshold)
        threshold = params[:compatibility_threshold]
        if threshold <= 0.0
            error("""Invalid compatibility_threshold: $threshold
                     Must be positive.

                     Typical range: 2.0 to 4.0
                     - Lower values: More species (more diversity, smaller species)
                     - Higher values: Fewer species (less diversity, larger species)""")
        end

        if threshold < 1.0
            @warn """Compatibility threshold ($threshold) is very low.
                     This will create many small species, which may be unstable.
                     Typical range: 2.0 to 4.0"""
        elseif threshold > 10.0
            @warn """Compatibility threshold ($threshold) is very high.
                     This will create very few species, reducing diversity.
                     Typical range: 2.0 to 4.0"""
        end
    end
end

"""
Validate stagnation section parameters.
"""
function validate_stagnation_params(params::Dict)
    check_unknown_parameters("DefaultStagnation", params, KNOWN_STAGNATION_PARAMS)

    if haskey(params, :max_stagnation)
        max_stag = params[:max_stagnation]
        if max_stag < 5
            @warn """max_stagnation ($max_stag) is very low.
                     Species may be removed before they can improve.
                     Typical range: 15 to 20"""
        end
    end

    if haskey(params, :species_fitness_func)
        func = Symbol(lowercase(string(params[:species_fitness_func])))
        valid_funcs = [:max, :min, :mean, :median]
        if !(func in valid_funcs)
            error("""Invalid species_fitness_func: '$(params[:species_fitness_func])'
                     Valid options: $(join(valid_funcs, ", "))

                     Recommended: 'max' (use best genome in species)""")
        end
    end
end

"""
Validate reproduction section parameters.
"""
function validate_reproduction_params(params::Dict)
    check_unknown_parameters("DefaultReproduction", params, KNOWN_REPRODUCTION_PARAMS)

    if haskey(params, :survival_threshold)
        threshold = params[:survival_threshold]
        if threshold <= 0.0 || threshold > 1.0
            error("""Invalid survival_threshold: $threshold
                     Must be in (0.0, 1.0]

                     Represents fraction of each species that survives to reproduce.
                     Typical values: 0.2 (top 20% survive)""")
        end
    end

    if haskey(params, :elitism)
        elitism = params[:elitism]
        if elitism < 0
            error("""Invalid elitism: $elitism
                     Must be non-negative.

                     Elitism = number of best genomes copied unchanged to next generation.
                     Typical values: 0 to 5""")
        end
    end
end

"""
Validate entire configuration with enhanced error checking.
"""
function validate_config(config_data::Dict)
    neat_params = get(config_data, :neat, Dict())
    genome_params = get(config_data, :defaultgenome, Dict())
    species_params = get(config_data, :defaultspeciesset, Dict())
    stagnation_params = get(config_data, :defaultstagnation, Dict())
    reproduction_params = get(config_data, :defaultreproduction, Dict())

    validate_neat_params(neat_params)
    validate_genome_params(genome_params)
    validate_species_params(species_params)
    validate_stagnation_params(stagnation_params)
    validate_reproduction_params(reproduction_params)

    # Check for completely missing sections
    if !haskey(config_data, :neat)
        @warn """Missing [NEAT] section in config file.
                 Using default values for all NEAT parameters."""
    end

    if !haskey(config_data, :defaultgenome)
        @warn """Missing [DefaultGenome] section in config file.
                 Using default values for all genome parameters.
                 It's recommended to explicitly specify num_inputs and num_outputs."""
    end
end

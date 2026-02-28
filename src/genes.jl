"""
Gene representations for NEAT: NodeGene and ConnectionGene.
"""

using Random

"""
NodeGene represents a node (neuron) in the network.
"""
mutable struct NodeGene
    key::Int
    bias::Float64
    response::Float64
    activation::Symbol
    aggregation::Symbol
    # CTRNN temporal dynamics parameter
    time_constant::Float64   # tau: controls response speed (NaN = not configured)
    # Izhikevich neuron model parameters
    iz_a::Float64            # recovery time scale (NaN = not configured)
    iz_b::Float64            # recovery sensitivity
    iz_c::Float64            # after-spike reset potential (mV)
    iz_d::Float64            # after-spike recovery increment
end

function NodeGene(key::Int)
    NodeGene(key, 0.0, 1.0, :sigmoid, :sum, NaN, NaN, NaN, NaN, NaN)
end

function init_attributes!(gene::NodeGene, config, rng::AbstractRNG=Random.GLOBAL_RNG)
    gene.bias = init_value(config.bias_attr, rng)
    gene.response = init_value(config.response_attr, rng)
    gene.activation = init_value(config.activation_attr, rng)
    gene.aggregation = init_value(config.aggregation_attr, rng)
    if config.time_constant_attr !== nothing
        gene.time_constant = init_value(config.time_constant_attr, rng)
    end
    if config.iz_a_attr !== nothing
        gene.iz_a = init_value(config.iz_a_attr, rng)
        gene.iz_b = init_value(config.iz_b_attr, rng)
        gene.iz_c = init_value(config.iz_c_attr, rng)
        gene.iz_d = init_value(config.iz_d_attr, rng)
    end
    return gene
end

function mutate!(gene::NodeGene, config, rng::AbstractRNG=Random.GLOBAL_RNG)
    gene.bias = mutate_value(config.bias_attr, gene.bias, rng)
    gene.response = mutate_value(config.response_attr, gene.response, rng)
    gene.activation = mutate_value(config.activation_attr, gene.activation, rng)
    gene.aggregation = mutate_value(config.aggregation_attr, gene.aggregation, rng)
    if config.time_constant_attr !== nothing
        gene.time_constant = mutate_value(config.time_constant_attr, gene.time_constant, rng)
    end
    if config.iz_a_attr !== nothing
        gene.iz_a = mutate_value(config.iz_a_attr, gene.iz_a, rng)
        gene.iz_b = mutate_value(config.iz_b_attr, gene.iz_b, rng)
        gene.iz_c = mutate_value(config.iz_c_attr, gene.iz_c, rng)
        gene.iz_d = mutate_value(config.iz_d_attr, gene.iz_d, rng)
    end
    return gene
end

function Base.copy(gene::NodeGene)
    return NodeGene(gene.key, gene.bias, gene.response, gene.activation, gene.aggregation,
                    gene.time_constant, gene.iz_a, gene.iz_b, gene.iz_c, gene.iz_d)
end

function crossover(gene1::NodeGene, gene2::NodeGene, rng::AbstractRNG=Random.GLOBAL_RNG)
    @assert gene1.key == gene2.key "Cannot crossover genes with different keys"

    new_gene = NodeGene(gene1.key)
    new_gene.bias = rand(rng, Bool) ? gene1.bias : gene2.bias
    new_gene.response = rand(rng, Bool) ? gene1.response : gene2.response
    new_gene.activation = rand(rng, Bool) ? gene1.activation : gene2.activation
    new_gene.aggregation = rand(rng, Bool) ? gene1.aggregation : gene2.aggregation
    new_gene.time_constant = rand(rng, Bool) ? gene1.time_constant : gene2.time_constant
    new_gene.iz_a = rand(rng, Bool) ? gene1.iz_a : gene2.iz_a
    new_gene.iz_b = rand(rng, Bool) ? gene1.iz_b : gene2.iz_b
    new_gene.iz_c = rand(rng, Bool) ? gene1.iz_c : gene2.iz_c
    new_gene.iz_d = rand(rng, Bool) ? gene1.iz_d : gene2.iz_d

    return new_gene
end

function distance(gene1::NodeGene, gene2::NodeGene, config)
    d = abs(gene1.bias - gene2.bias) + abs(gene1.response - gene2.response)
    if gene1.activation != gene2.activation
        d += 1.0
    end
    if gene1.aggregation != gene2.aggregation
        d += 1.0
    end
    # Include CTRNN/IZNN parameters when both genes have them configured
    if !isnan(gene1.time_constant) && !isnan(gene2.time_constant)
        d += abs(gene1.time_constant - gene2.time_constant)
    end
    if !isnan(gene1.iz_a) && !isnan(gene2.iz_a)
        d += abs(gene1.iz_a - gene2.iz_a)
        d += abs(gene1.iz_b - gene2.iz_b)
        d += abs(gene1.iz_c - gene2.iz_c)
        d += abs(gene1.iz_d - gene2.iz_d)
    end
    return d * config.compatibility_weight_coefficient
end

"""
ConnectionGene represents a connection between nodes.

The innovation field tracks the historical origin of this connection gene,
enabling proper alignment during crossover as per the NEAT paper.
"""
mutable struct ConnectionGene
    key::Tuple{Int, Int}  # (input_id, output_id)
    weight::Float64
    enabled::Bool
    innovation::Int      # Historical marker for gene alignment
end

function ConnectionGene(key::Tuple{Int, Int}, innovation::Int=0)
    ConnectionGene(key, 0.0, true, innovation)
end

function init_attributes!(gene::ConnectionGene, config, rng::AbstractRNG=Random.GLOBAL_RNG)
    gene.weight = init_value(config.weight_attr, rng)
    gene.enabled = init_value(config.enabled_attr, rng)
    return gene
end

function mutate!(gene::ConnectionGene, config, rng::AbstractRNG=Random.GLOBAL_RNG)
    gene.weight = mutate_value(config.weight_attr, gene.weight, rng)
    gene.enabled = mutate_value(config.enabled_attr, gene.enabled, rng)
    return gene
end

function Base.copy(gene::ConnectionGene)
    return ConnectionGene(gene.key, gene.weight, gene.enabled, gene.innovation)
end

function crossover(gene1::ConnectionGene, gene2::ConnectionGene, rng::AbstractRNG=Random.GLOBAL_RNG)
    @assert gene1.key == gene2.key "Cannot crossover genes with different keys"

    # Inherit innovation number from one parent (prefer gene1 for consistency)
    # Note: In NEAT paper, matching genes should have same innovation number,
    # but we use connection keys for matching to handle edge cases
    new_gene = ConnectionGene(gene1.key, gene1.innovation)
    new_gene.weight = rand(rng, Bool) ? gene1.weight : gene2.weight

    # Per NEAT paper: if either parent is disabled, 75% chance offspring is disabled
    if !gene1.enabled || !gene2.enabled
        new_gene.enabled = rand(rng) > 0.75
    else
        new_gene.enabled = true
    end

    return new_gene
end

function distance(gene1::ConnectionGene, gene2::ConnectionGene, config)
    d = abs(gene1.weight - gene2.weight)
    if gene1.enabled != gene2.enabled
        d += 1.0
    end
    return d * config.compatibility_weight_coefficient
end

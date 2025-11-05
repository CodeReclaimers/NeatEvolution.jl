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
end

function NodeGene(key::Int)
    NodeGene(key, 0.0, 1.0, :sigmoid, :sum)
end

function init_attributes!(gene::NodeGene, config, rng::AbstractRNG=Random.GLOBAL_RNG)
    gene.bias = init_value(config.bias_attr, rng)
    gene.response = init_value(config.response_attr, rng)
    gene.activation = init_value(config.activation_attr, rng)
    gene.aggregation = init_value(config.aggregation_attr, rng)
    return gene
end

function mutate!(gene::NodeGene, config, rng::AbstractRNG=Random.GLOBAL_RNG)
    gene.bias = mutate_value(config.bias_attr, gene.bias, rng)
    gene.response = mutate_value(config.response_attr, gene.response, rng)
    gene.activation = mutate_value(config.activation_attr, gene.activation, rng)
    gene.aggregation = mutate_value(config.aggregation_attr, gene.aggregation, rng)
    return gene
end

function Base.copy(gene::NodeGene)
    return NodeGene(gene.key, gene.bias, gene.response, gene.activation, gene.aggregation)
end

function crossover(gene1::NodeGene, gene2::NodeGene, rng::AbstractRNG=Random.GLOBAL_RNG)
    @assert gene1.key == gene2.key "Cannot crossover genes with different keys"

    new_gene = NodeGene(gene1.key)
    new_gene.bias = rand(rng, Bool) ? gene1.bias : gene2.bias
    new_gene.response = rand(rng, Bool) ? gene1.response : gene2.response
    new_gene.activation = rand(rng, Bool) ? gene1.activation : gene2.activation
    new_gene.aggregation = rand(rng, Bool) ? gene1.aggregation : gene2.aggregation

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
    return d * config.compatibility_weight_coefficient
end

"""
ConnectionGene represents a connection between nodes.
"""
mutable struct ConnectionGene
    key::Tuple{Int, Int}  # (input_id, output_id)
    weight::Float64
    enabled::Bool
end

function ConnectionGene(key::Tuple{Int, Int})
    ConnectionGene(key, 0.0, true)
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
    return ConnectionGene(gene.key, gene.weight, gene.enabled)
end

function crossover(gene1::ConnectionGene, gene2::ConnectionGene, rng::AbstractRNG=Random.GLOBAL_RNG)
    @assert gene1.key == gene2.key "Cannot crossover genes with different keys"

    new_gene = ConnectionGene(gene1.key)
    new_gene.weight = rand(rng, Bool) ? gene1.weight : gene2.weight
    new_gene.enabled = rand(rng, Bool) ? gene1.enabled : gene2.enabled

    return new_gene
end

function distance(gene1::ConnectionGene, gene2::ConnectionGene, config)
    d = abs(gene1.weight - gene2.weight)
    if gene1.enabled != gene2.enabled
        d += 1.0
    end
    return d * config.compatibility_weight_coefficient
end

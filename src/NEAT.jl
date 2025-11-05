"""
NEAT (NeuroEvolution of Augmenting Topologies) implementation in Julia.

This package provides a complete implementation of the NEAT algorithm for
evolving artificial neural networks through genetic algorithms.
"""
module NEAT

using Random
using Statistics
using TOML

# Include all components
include("utils.jl")
include("attributes.jl")
include("genes.jl")
include("activations.jl")
include("aggregations.jl")
include("graphs.jl")
include("config.jl")
include("genome.jl")
include("species.jl")
include("stagnation.jl")
include("reproduction.jl")
include("reporting.jl")
include("feedforward.jl")
include("population.jl")

# Export main types
export Config, load_config
export GenomeConfig, SpeciesConfig, StagnationConfig, ReproductionConfig
export Genome, NodeGene, ConnectionGene
export Population, CompleteExtinctionException
export FeedForwardNetwork
export StdOutReporter

# Export main functions
export run!, activate!
export add_reporter!
export configure_new!, configure_crossover!, mutate!

# Export activation functions
export sigmoid_activation, tanh_activation, relu_activation
export get_activation_function, add_activation_function!

# Export aggregation functions
export sum_aggregation, product_aggregation, max_aggregation, min_aggregation
export get_aggregation_function, add_aggregation_function!

end # module NEAT

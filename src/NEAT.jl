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
include("statistics.jl")
include("feedforward.jl")
include("population.jl")

# Export main types
export Config, load_config
export GenomeConfig, SpeciesConfig, StagnationConfig, ReproductionConfig
export Genome, NodeGene, ConnectionGene
export Population, CompleteExtinctionException
export FeedForwardNetwork
export Reporter, StdOutReporter, StatisticsReporter

# Export main functions
export run!, activate!
export add_reporter!
export configure_new!, configure_crossover!, mutate!

# Export activation functions
export sigmoid_activation, tanh_activation, relu_activation
export sin_activation, gauss_activation, softplus_activation
export identity_activation, clamped_activation, inv_activation
export log_activation, exp_activation, abs_activation
export hat_activation, square_activation, cube_activation
export elu_activation, lelu_activation, selu_activation
export get_activation_function, add_activation_function!

# Export aggregation functions
export sum_aggregation, product_aggregation, max_aggregation, min_aggregation
export maxabs_aggregation, median_aggregation, mean_aggregation
export get_aggregation_function, add_aggregation_function!

# Export graph functions
export creates_cycle, required_for_output, feed_forward_layers

# Export reproduction functions
export compute_spawn

# Export statistics functions
export get_fitness_mean, get_fitness_stdev, get_fitness_median
export get_species_sizes, get_species_fitness
export best_genome, best_genomes, best_unique_genomes
export save_statistics

# Export visualization functions (implemented in extensions)
export plot_fitness, plot_species, plot_fitness_comparison
export draw_net, draw_net_comparison

# Function stubs for Plots.jl extension
"""
Plot fitness statistics (requires Plots.jl).
Load Plots.jl to use this function: `using Plots`
"""
function plot_fitness end

"""
Plot species sizes over time (requires Plots.jl).
Load Plots.jl to use this function: `using Plots`
"""
function plot_species end

"""
Plot fitness comparison between runs (requires Plots.jl).
Load Plots.jl to use this function: `using Plots`
"""
function plot_fitness_comparison end

# Function stubs for network visualization (also in Plots.jl extension)
"""
Draw a neural network genome as a graph (requires Plots.jl).
Load Plots.jl to use this function: `using Plots`

Shows network topology with:
- Input nodes (green), output nodes (blue), hidden nodes (white)
- Connections colored by weight (red=negative, green=positive)
- Connection thickness proportional to weight magnitude
- Disabled connections shown as dashed gray lines
- Layer-based automatic layout
"""
function draw_net end

"""
Draw multiple genomes for comparison (requires Plots.jl).
Load Plots.jl to use this function: `using Plots`

Creates a grid layout comparing multiple network structures.
"""
function draw_net_comparison end

end # module NEAT

"""
XOR Example with Visualization for NEAT.

This example evolves a neural network to solve the XOR problem
and generates visualizations of the evolution process.

Usage:
    julia --project evolve_with_visualization.jl

Requires Plots.jl to be installed:
    using Pkg; Pkg.add("Plots")
"""

using NEAT

# Load Plots for visualization
using Plots

# XOR inputs and expected outputs
const XOR_INPUTS = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
]

const XOR_OUTPUTS = [
    [0.0],
    [1.0],
    [1.0],
    [0.0]
]

"""
Fitness function for XOR problem.
"""
function eval_genomes(genomes, config)
    for (genome_id, genome) in genomes
        genome.fitness = 4.0
        net = FeedForwardNetwork(genome, config.genome_config)

        for (xi, xo) in zip(XOR_INPUTS, XOR_OUTPUTS)
            output = activate!(net, xi)
            genome.fitness -= (output[1] - xo[1])^2
        end
    end
end

function main()
    # Load configuration
    config_path = joinpath(@__DIR__, "config.toml")
    config = load_config(config_path)

    # Create population
    pop = Population(config)

    # Add reporters
    add_reporter!(pop, StdOutReporter(true))

    # Add statistics reporter for visualization
    stats = StatisticsReporter()
    add_reporter!(pop, stats)

    # Run evolution
    println("Starting XOR evolution with visualization...")
    println("This will evolve for up to 100 generations.\n")

    winner = run!(pop, eval_genomes, 100)

    # Display winning genome
    println("\n" * "="^70)
    println("EVOLUTION COMPLETE")
    println("="^70)
    println("\nBest genome:")
    println("  Key: $(winner.key)")
    println("  Fitness: $(round(winner.fitness, digits=5))")
    println("  Nodes: $(length(winner.nodes))")
    println("  Connections: $(length(winner.connections))")
    println("  Enabled connections: $(count(c.enabled for c in values(winner.connections)))")

    # Test the winner
    println("\nTesting winner network:")
    winner_net = FeedForwardNetwork(winner, config.genome_config)
    println("  Input      Expected   Output     Error")
    println("  " * "-"^45)

    total_error = 0.0
    for (xi, xo) in zip(XOR_INPUTS, XOR_OUTPUTS)
        output = activate!(winner_net, xi)
        error = abs(output[1] - xo[1])
        total_error += error
        println("  $(xi)  $(xo)    $(round.(output, digits=4))  $(round(error, digits=4))")
    end

    println("\n  Average error: $(round(total_error / length(XOR_INPUTS), digits=4))")

    # Generate visualizations
    println("\n" * "="^70)
    println("GENERATING VISUALIZATIONS")
    println("="^70 * "\n")

    # Save statistics to CSV
    save_statistics(stats, prefix="xor")

    # Plot fitness evolution
    plot_fitness(stats,
                 filename="xor_fitness.png",
                 title="XOR Fitness Evolution")

    # Plot species evolution
    plot_species(stats,
                 filename="xor_species.png",
                 title="XOR Species Evolution")

    # Draw winner network structure
    node_names = Dict(
        -1 => "x1",
        -2 => "x2",
        0 => "XOR"
    )
    draw_net(winner, config.genome_config,
             filename="xor_winner.png",
             node_names=node_names,
             show_disabled=true,
             prune_unused=false)

    # Plot activation heatmap showing what the network learned
    plot_activation_heatmap(winner, config.genome_config,
                           x_range=(-0.5, 1.5),
                           y_range=(-0.5, 1.5),
                           resolution=100,
                           filename="xor_activation_heatmap.png",
                           title="XOR Winner Activation Map")

    # Animate evolution (optional - can take a moment to generate)
    println("\nGenerating evolution animation...")
    animate_evolution(stats, config.genome_config,
                     filename="xor_evolution.gif",
                     fps=3,
                     node_names=node_names,
                     show_disabled=false)

    # Print statistics summary
    println("\nEvolution statistics:")
    println("  Generations: $(length(stats.most_fit_genomes))")
    println("  Final average fitness: $(round(get_fitness_mean(stats)[end], digits=4))")
    println("  Final best fitness: $(round(winner.fitness, digits=4))")

    species_sizes = get_species_sizes(stats)
    if !isempty(species_sizes)
        final_species = count(x -> x > 0, species_sizes[end])
        max_species = maximum(count(x -> x > 0, sizes) for sizes in species_sizes)
        println("  Final species count: $final_species")
        println("  Max species count: $max_species")
    end

    println("\nGenerated files:")
    println("  - xor_fitness.png: Fitness evolution plot")
    println("  - xor_species.png: Species evolution plot")
    println("  - xor_winner.png: Winner network structure diagram")
    println("  - xor_activation_heatmap.png: Activation heatmap showing network behavior")
    println("  - xor_evolution.gif: Animation of network evolution over time")
    println("  - xor_fitness.csv: Fitness statistics")
    println("  - xor_speciation.csv: Species sizes per generation")
    println("  - xor_species_fitness.csv: Species fitness per generation")

    println("\n" * "="^70)
    println("DONE")
    println("="^70)

    return winner, stats
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

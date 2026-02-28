"""
XOR Example for NEAT.

This example evolves a neural network to solve the XOR problem.
"""

using NeatEvolution

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

    # Add reporter
    add_reporter!(pop, StdOutReporter(true))

    # Run evolution
    println("Starting XOR evolution...")
    winner = run!(pop, eval_genomes)

    # Display winning genome
    println("\nBest genome:")
    println("  Key: $(winner.key)")
    println("  Fitness: $(winner.fitness)")
    println("  Nodes: $(length(winner.nodes))")
    println("  Connections: $(length(winner.connections))")

    # Test the winner
    println("\nOutput of winner network:")
    winner_net = FeedForwardNetwork(winner, config.genome_config)
    for (xi, xo) in zip(XOR_INPUTS, XOR_OUTPUTS)
        output = activate!(winner_net, xi)
        println("  input $(xi), expected $(xo), got $(round.(output, digits=3))")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

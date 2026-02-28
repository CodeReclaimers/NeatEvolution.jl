"""
Sequence Classification Example for NEAT.

Evolves a recurrent network that classifies whether a binary sequence
contains more 1s than 0s. This task requires memory, making RecurrentNetwork
necessary — a feed-forward network cannot solve it since inputs arrive
one at a time.

Usage:
    julia --project examples/sequence/evolve.jl
"""

using NeatEvolution
using Random

# Generate test sequences: pairs of (sequence, label)
# label = 1.0 if more 1s than 0s, else 0.0
function generate_sequences(rng::AbstractRNG; n_sequences=10, seq_length=5)
    sequences = Vector{Vector{Float64}}[]
    labels = Float64[]
    for _ in 1:n_sequences
        seq = [Float64(rand(rng, 0:1)) for _ in 1:seq_length]
        ones_count = sum(seq)
        label = ones_count > seq_length / 2.0 ? 1.0 : 0.0
        push!(sequences, [[x] for x in seq])  # wrap each element for single-input network
        push!(labels, label)
    end
    sequences, labels
end

# Fixed test data (deterministic for consistent fitness evaluation)
const RNG_SEED = 42
const SEQUENCES, LABELS = generate_sequences(Random.MersenneTwister(RNG_SEED))

function eval_genomes(genomes, config)
    for (genome_id, genome) in genomes
        net = RecurrentNetwork(genome, config.genome_config)
        fitness = 0.0

        for (seq, label) in zip(SEQUENCES, LABELS)
            reset!(net)
            output = [0.0]
            # Feed sequence elements one at a time
            for input in seq
                output = activate!(net, input)
            end
            # Score based on how close output is to correct label
            fitness += 1.0 - (output[1] - label)^2
        end

        genome.fitness = fitness
    end
end

function main()
    config_path = joinpath(@__DIR__, "config.toml")
    config = load_config(config_path)

    pop = Population(config)
    add_reporter!(pop, StdOutReporter(true))

    println("Sequence Classification with RecurrentNetwork")
    println("Task: Classify if binary sequence has more 1s than 0s")
    println("Sequences: $(length(SEQUENCES)), length: $(length(SEQUENCES[1]))")
    println()

    winner = run!(pop, eval_genomes, 300)

    println("\nBest genome:")
    println("  Key: $(winner.key)")
    println("  Fitness: $(round(winner.fitness, digits=3))")
    println("  Nodes: $(length(winner.nodes))")
    println("  Connections: $(length(winner.connections))")

    # Test the winner
    println("\nWinner network results:")
    net = RecurrentNetwork(winner, config.genome_config)
    correct = 0
    for (seq, label) in zip(SEQUENCES, LABELS)
        reset!(net)
        output = [0.0]
        for input in seq
            output = activate!(net, input)
        end
        prediction = output[1] > 0.5 ? 1.0 : 0.0
        match = prediction == label ? "✓" : "✗"
        if prediction == label
            correct += 1
        end
        seq_str = join([Int(x[1]) for x in seq], "")
        println("  $seq_str → $(round(output[1], digits=3)) (label=$label) $match")
    end
    println("Accuracy: $correct/$(length(LABELS))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

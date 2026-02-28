"""
Checkpoint Demo for NEAT.

Demonstrates saving and restoring evolution state using both:
1. Automatic checkpointing via the Checkpointer reporter
2. Manual checkpointing via save_checkpoint/restore_checkpoint

Uses the XOR task for simplicity.

Usage:
    julia --project examples/checkpoint_demo/evolve.jl
"""

using NEAT

const XOR_INPUTS = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
const XOR_OUTPUTS = [[0.0], [1.0], [1.0], [0.0]]

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
    config_path = joinpath(@__DIR__, "config.toml")
    config = load_config(config_path)
    checkpoint_prefix = joinpath(tempdir(), "neat-checkpoint-demo")

    # --- Part 1: Automatic checkpointing via reporter ---
    println("=" ^ 60)
    println("Part 1: Automatic Checkpointing (via Checkpointer reporter)")
    println("=" ^ 60)

    pop = Population(config)
    add_reporter!(pop, StdOutReporter(false))
    add_reporter!(pop, Checkpointer(generation_interval=10, filename_prefix=checkpoint_prefix))

    # Run for 20 generations — checkpoints saved at generation 10 and 20
    println("\nRunning 20 generations with checkpoint every 10...")
    winner = run!(pop, eval_genomes, 20)
    println("Best fitness after 20 generations: $(round(winner.fitness, digits=4))")

    # Check what checkpoint files were created
    checkpoint_10 = "$(checkpoint_prefix)-10"
    checkpoint_20 = "$(checkpoint_prefix)-20"
    println("\nCheckpoint files created:")
    println("  $(checkpoint_10): $(isfile(checkpoint_10) ? "exists" : "missing")")
    println("  $(checkpoint_20): $(isfile(checkpoint_20) ? "exists" : "missing")")

    # --- Part 2: Restore from checkpoint and continue ---
    println("\n" * "=" ^ 60)
    println("Part 2: Restore from Checkpoint and Continue")
    println("=" ^ 60)

    if isfile(checkpoint_10)
        println("\nRestoring from generation 10 checkpoint...")
        restored_pop = restore_checkpoint(checkpoint_10)
        add_reporter!(restored_pop, StdOutReporter(false))

        println("Running 20 more generations from restored state...")
        winner2 = run!(restored_pop, eval_genomes, 20)
        println("Best fitness after restore + 20 generations: $(round(winner2.fitness, digits=4))")
    end

    # --- Part 3: Manual checkpointing ---
    println("\n" * "=" ^ 60)
    println("Part 3: Manual Checkpointing")
    println("=" ^ 60)

    pop2 = Population(config)
    add_reporter!(pop2, StdOutReporter(false))

    # Run for 10 generations
    println("\nRunning 10 generations...")
    winner3 = run!(pop2, eval_genomes, 10)
    println("Best fitness after 10 generations: $(round(winner3.fitness, digits=4))")

    # Manually save checkpoint
    manual_checkpoint = joinpath(tempdir(), "neat-manual-checkpoint")
    save_checkpoint(manual_checkpoint, pop2)
    println("Manual checkpoint saved to: $(manual_checkpoint)")

    # Restore and continue
    println("\nRestoring from manual checkpoint...")
    restored_pop2 = restore_checkpoint(manual_checkpoint)
    add_reporter!(restored_pop2, StdOutReporter(false))

    println("Running 10 more generations...")
    winner4 = run!(restored_pop2, eval_genomes, 10)
    println("Best fitness after restore + 10 generations: $(round(winner4.fitness, digits=4))")

    # Cleanup
    for f in [checkpoint_10, checkpoint_20, manual_checkpoint]
        isfile(f) && rm(f)
    end
    println("\nCheckpoint files cleaned up.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

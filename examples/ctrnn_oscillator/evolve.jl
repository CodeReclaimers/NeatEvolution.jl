"""
CTRNN Oscillator Example for NEAT.

Evolves a Continuous-Time Recurrent Neural Network that produces oscillating
output (alternating high/low) when given a constant input. This demonstrates
CTRNN dynamics — a feed-forward or standard recurrent network with constant
input produces constant output, but a CTRNN can oscillate due to its
continuous temporal dynamics.

Fitness rewards output variance (oscillation amplitude) over time.

Usage:
    julia --project examples/ctrnn_oscillator/evolve.jl
"""

using NEAT

const SIM_TIME = 2.0      # total simulation time (seconds)
const TIME_STEP = 0.01     # integration step (seconds)
const SAMPLE_INTERVAL = 0.05  # how often to sample output
const CONSTANT_INPUT = [1.0]

function eval_genomes(genomes, config)
    for (genome_id, genome) in genomes
        net = CTRNNNetwork(genome, config.genome_config)

        # Collect output samples over time
        samples = Float64[]
        t = 0.0
        next_sample = 0.0

        while t < SIM_TIME
            step = min(TIME_STEP, SIM_TIME - t)
            output = advance!(net, CONSTANT_INPUT, step, step)
            t += step

            if t >= next_sample
                push!(samples, output[1])
                next_sample += SAMPLE_INTERVAL
            end
        end

        if length(samples) < 2
            genome.fitness = 0.0
            continue
        end

        # Fitness = output variance (rewards oscillation)
        # Normalize to [0, 1] range using tanh
        mean_val = sum(samples) / length(samples)
        variance = sum((s - mean_val)^2 for s in samples) / length(samples)

        # Also reward having multiple sign changes (actual oscillation, not drift)
        sign_changes = 0
        centered = samples .- mean_val
        for i in 2:length(centered)
            if centered[i] * centered[i-1] < 0
                sign_changes += 1
            end
        end

        # Combined fitness: variance component + oscillation component
        var_score = tanh(variance * 10.0)  # saturates around 1.0 for large variance
        osc_score = min(1.0, sign_changes / 5.0)  # reward up to 5 sign changes
        genome.fitness = 0.5 * var_score + 0.5 * osc_score
    end
end

function main()
    config_path = joinpath(@__DIR__, "config.toml")
    config = load_config(config_path)

    pop = Population(config)
    add_reporter!(pop, StdOutReporter(true))

    println("CTRNN Oscillator Evolution")
    println("Task: Evolve oscillating output from constant input")
    println("Simulation: $(SIM_TIME)s at dt=$(TIME_STEP)s")
    println()

    winner = run!(pop, eval_genomes, 300)

    println("\nBest genome:")
    println("  Key: $(winner.key)")
    println("  Fitness: $(round(winner.fitness, digits=4))")
    println("  Nodes: $(length(winner.nodes))")
    println("  Connections: $(length(winner.connections))")

    # Show winner output trajectory
    println("\nWinner output over time:")
    net = CTRNNNetwork(winner, config.genome_config)
    t = 0.0
    while t < SIM_TIME
        step = min(TIME_STEP, SIM_TIME - t)
        output = advance!(net, CONSTANT_INPUT, step, step)
        t += step
        if t % 0.1 < TIME_STEP  # print every ~0.1s
            println("  t=$(round(t, digits=2))s → output=$(round(output[1], digits=4))")
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

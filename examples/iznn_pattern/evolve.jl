"""
Izhikevich Spiking Network Example for NEAT.

Evolves a spiking network that produces a specific spike pattern in response
to constant input current. The target pattern is a regular spiking rhythm.
Fitness measures how closely the network's spike timing matches the target.

Usage:
    julia --project examples/iznn_pattern/evolve.jl
"""

using NeatEvolution

const SIM_DURATION_MS = 100.0  # simulation duration in milliseconds
const DT_MS = 1.0             # time step in milliseconds
const INPUT_CURRENT = [10.0]  # constant input current

# Target spike pattern: regular spiking at approximately every 20ms
# Target spike times (in ms)
const TARGET_SPIKES = [4.0, 24.0, 44.0, 64.0, 84.0]
const SPIKE_WINDOW = 5.0  # tolerance window (ms) for spike matching

function eval_genomes(genomes, config)
    for (genome_id, genome) in genomes
        net = IZNNNetwork(genome, config.genome_config)
        set_inputs!(net, INPUT_CURRENT)

        # Record spike times
        spike_times = Float64[]
        n_steps = Int(SIM_DURATION_MS / DT_MS)

        for step in 1:n_steps
            output = advance!(net, DT_MS)
            if output[1] > 0.5  # spike detected
                push!(spike_times, step * DT_MS)
            end
        end

        # Compute fitness based on spike timing match
        # For each target spike, find the closest actual spike
        matched = 0
        for target_t in TARGET_SPIKES
            for spike_t in spike_times
                if abs(spike_t - target_t) <= SPIKE_WINDOW
                    matched += 1
                    break
                end
            end
        end

        # Penalize extra spikes (we want a clean pattern)
        extra_spikes = max(0, length(spike_times) - length(TARGET_SPIKES))
        extra_penalty = 0.1 * extra_spikes

        # Fitness: fraction of target spikes matched, minus penalty for extras
        genome.fitness = max(0.0, matched / length(TARGET_SPIKES) - extra_penalty)
    end
end

function main()
    config_path = joinpath(@__DIR__, "config.toml")
    config = load_config(config_path)

    pop = Population(config)
    add_reporter!(pop, StdOutReporter(true))

    println("Izhikevich Spiking Network Pattern Evolution")
    println("Task: Match target spike pattern $(TARGET_SPIKES) ms")
    println("Simulation: $(SIM_DURATION_MS)ms at dt=$(DT_MS)ms")
    println("Input current: $(INPUT_CURRENT[1])")
    println()

    winner = run!(pop, eval_genomes, 300)

    println("\nBest genome:")
    println("  Key: $(winner.key)")
    println("  Fitness: $(round(winner.fitness, digits=4))")
    println("  Nodes: $(length(winner.nodes))")
    println("  Connections: $(length(winner.connections))")

    # Show winner spike pattern
    println("\nWinner spike pattern:")
    net = IZNNNetwork(winner, config.genome_config)
    set_inputs!(net, INPUT_CURRENT)

    spike_times = Float64[]
    n_steps = Int(SIM_DURATION_MS / DT_MS)

    for step in 1:n_steps
        output = advance!(net, DT_MS)
        if output[1] > 0.5
            push!(spike_times, step * DT_MS)
        end
    end

    println("  Target:  $(TARGET_SPIKES)")
    println("  Actual:  $(spike_times)")
    println("  Count:   $(length(spike_times)) spikes (target: $(length(TARGET_SPIKES)))")

    # Show Izhikevich parameters of the output neuron
    for (nid, neuron) in net.neurons
        println("  Neuron $nid: a=$(round(neuron.a, digits=4)), b=$(round(neuron.b, digits=4)), " *
                "c=$(round(neuron.c, digits=1)), d=$(round(neuron.d, digits=2))")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

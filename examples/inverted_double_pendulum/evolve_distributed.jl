"""
Inverted Double Pendulum Example with Distributed Computing.

This version uses Julia's Distributed computing for parallelization,
which can scale across multiple machines or processes.

Requirements:
- All requirements from evolve.jl
- Distributed.jl (stdlib)

Setup:
```bash
# Run with multiple processes (e.g., 8 processes)
julia -p 8 --project examples/inverted_double_pendulum/evolve_distributed.jl

# Or auto-detect CPU cores
julia -p auto --project examples/inverted_double_pendulum/evolve_distributed.jl
```

Note: Distributed computing has more overhead than multi-threading but can:
- Scale to multiple machines
- Isolate processes (useful if crashes occur)
- Handle very large populations
"""

using Distributed

# Load required packages on all workers
@everywhere using NeatEvolution
@everywhere using PyCall

# Import gymnasium on all workers
@everywhere const gym = PyNULL()

@everywhere function __init__()
    copy!(gym, pyimport("gymnasium"))
end

# Evaluate a single genome in the InvertedDoublePendulum environment.
# This function runs on worker processes.
@everywhere function eval_genome_worker(genome, config, episodes::Int=5)
    net = FeedForwardNetwork(genome, config.genome_config)

    total_reward = 0.0
    for episode in 1:episodes
        env = gym.make("InvertedDoublePendulum-v5")
        observation, info = env.reset()
        episode_reward = 0.0
        done = false
        truncated = false

        while !done && !truncated
            obs = collect(Float64, observation)
            output = activate!(net, obs)
            action = clamp(output[1], -1.0, 1.0)
            observation, reward, done, truncated, info = env.step([action])
            episode_reward += reward
        end

        env.close()
        total_reward += episode_reward
    end

    return total_reward / episodes
end

"""
Fitness function using distributed computing.
Evaluates genomes in parallel across worker processes.
"""
function eval_genomes_distributed(genomes, config)
    # Create a list of (genome_id, genome) pairs
    genome_list = collect(genomes)

    # Distribute evaluation across workers
    # pmap automatically handles load balancing
    fitnesses = pmap(genome_list) do (genome_id, genome)
        eval_genome_worker(genome, config, 5)
    end

    # Assign fitness values back to genomes
    for (i, (genome_id, genome)) in enumerate(genome_list)
        genome.fitness = fitnesses[i]
    end
end

"""
Test a genome by visualizing its performance.
"""
function visualize_genome(genome, config, episodes::Int=5)
    println("\nVisualizing genome performance...")
    println("  Genome key: $(genome.key)")
    println("  Fitness: $(genome.fitness)")
    println("  Nodes: $(length(genome.nodes))")
    println("  Connections: $(length(genome.connections))")

    total_reward = 0.0
    for ep in 1:episodes
        env = gym.make("InvertedDoublePendulum-v5", render_mode="human")
        observation, info = env.reset()
        episode_reward = 0.0
        done = false
        truncated = false

        net = FeedForwardNetwork(genome, config.genome_config)

        while !done && !truncated
            obs = collect(Float64, observation)
            output = activate!(net, obs)
            action = clamp(output[1], -1.0, 1.0)
            observation, reward, done, truncated, info = env.step([action])
            episode_reward += reward
        end

        env.close()
        total_reward += episode_reward
        println("  Episode $ep reward: $episode_reward")
    end

    avg_reward = total_reward / episodes
    println("  Average reward: $avg_reward")

    return avg_reward
end

function main()
    # Initialize Python imports on all workers
    @everywhere __init__()

    println("="^70)
    println("NEAT Inverted Double Pendulum Evolution (Distributed)")
    println("="^70)
    println("Workers: $(nworkers())")
    println("Goal: Evolve a neural network to balance two poles on a cart")
    println("="^70)

    # Load configuration
    config_path = joinpath(@__DIR__, "config.toml")
    config = load_config(config_path)

    # Create population
    pop = Population(config)

    # Add reporter
    add_reporter!(pop, StdOutReporter(true))

    # Run evolution with distributed evaluation
    println("\nStarting evolution...")
    winner = run!(pop, eval_genomes_distributed, 300)

    # Display winning genome
    println("\n" * "="^70)
    println("Evolution complete!")
    println("="^70)
    println("Best genome found:")
    println("  Key: $(winner.key)")
    println("  Fitness: $(winner.fitness)")
    println("  Nodes: $(length(winner.nodes))")
    println("  Connections: $(length(winner.connections))")

    # Test winner more thoroughly
    println("\nEvaluating winner over 100 episodes...")
    winner_net = FeedForwardNetwork(winner, config.genome_config)
    test_fitness = eval_genome_worker(winner, config, 100)
    println("  Average reward over 100 episodes: $test_fitness")

    if test_fitness >= 8000.0
        println("\n🎉 Excellent solution! (average reward >= 8000)")
    elseif test_fitness >= 5000.0
        println("\n✓ Good solution! (average reward >= 5000)")
    elseif test_fitness >= 2000.0
        println("\n+ Decent solution (average reward >= 2000)")
    else
        println("\n  Solution needs improvement (target: 5000+)")
    end

    # Ask user if they want to visualize
    println("\nVisualize the winner? (requires display and mujoco) [y/N]: ")
    response = lowercase(strip(readline()))

    if response == "y" || response == "yes"
        try
            visualize_genome(winner, config, 5)
        catch e
            println("Visualization failed: $e")
            println("Make sure you have MuJoCo and display available.")
        end
    end

    println("\n" * "="^70)
    println("Run complete!")
    println("="^70)

    return winner
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

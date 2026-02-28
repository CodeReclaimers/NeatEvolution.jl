"""
Inverted Double Pendulum Example with TRUE Parallelization.

This version uses process-based parallelism to avoid Python's GIL.
Each worker process has its own Python interpreter, avoiding GIL contention.

Setup:
```bash
# Install if needed
julia -e 'using Pkg; Pkg.add("Distributed")'

# Run with multiple processes
julia -p 8 --project examples/inverted_double_pendulum/evolve_parallel.jl
```

Key insight: Threading doesn't work with PyCall due to Python's GIL.
We need separate processes, each with its own Python interpreter.
"""

using Distributed

# Add worker processes if not already added
if nworkers() == 1 && nprocs() == 1
    println("Warning: Running with single process. Use: julia -p N --project evolve_parallel.jl")
    println("Starting 4 worker processes automatically...")
    addprocs(4)
end

println("Active workers: $(nworkers())")

# Load packages on all workers
@everywhere begin
    using NeatEvolution
    using PyCall

    # Each worker gets its own Python interpreter instance
    const gym = PyNULL()

    function __init__()
        copy!(gym, pyimport("gymnasium"))
    end

    # Initialize immediately
    __init__()
end

# Evaluate a single genome across multiple episodes.
# This runs on worker processes, each with its own Python interpreter.
@everywhere function eval_single_genome(genome, genome_config, episodes::Int=5)
    net = FeedForwardNetwork(genome, genome_config)

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
Parallel fitness evaluation using distributed computing.
Each genome is evaluated on a separate worker process.
"""
function eval_genomes(genomes, config)
    genome_list = collect(genomes)

    # Distribute work across workers
    # pmap handles load balancing automatically
    fitnesses = pmap(genome_list) do (genome_id, genome)
        eval_single_genome(genome, config.genome_config, 5)
    end

    # Assign fitness back to genomes
    for (i, (genome_id, genome)) in enumerate(genome_list)
        genome.fitness = fitnesses[i]
    end
end

"""
Visualize a genome's performance.
"""
function visualize_genome(genome, config, episodes::Int=5)
    println("\nVisualizing genome performance...")
    println("  Genome key: $(genome.key)")
    println("  Fitness: $(genome.fitness)")
    println("  Nodes: $(length(genome.nodes))")
    println("  Connections: $(length(genome.connections))")

    net = FeedForwardNetwork(genome, config.genome_config)

    total_reward = 0.0
    for ep in 1:episodes
        env = gym.make("InvertedDoublePendulum-v5", render_mode="human")
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
        println("  Episode $ep reward: $episode_reward")
    end

    avg_reward = total_reward / episodes
    println("  Average reward: $avg_reward")

    return avg_reward
end

function main()
    println("="^70)
    println("NEAT Inverted Double Pendulum Evolution (Parallel)")
    println("="^70)
    println("Active workers: $(nworkers())")
    println("Total processes: $(nprocs())")
    println("="^70)

    # Load configuration
    config_path = joinpath(@__DIR__, "config.toml")
    config = load_config(config_path)

    # Create population
    pop = Population(config)

    # Add reporter
    add_reporter!(pop, StdOutReporter(true))

    # Run evolution
    println("\nStarting parallel evolution...")
    println("Each genome will be evaluated on a separate worker process")
    println("This avoids Python's GIL limitations")
    println()

    winner = run!(pop, eval_genomes, 500)

    # Display results
    println("\n" * "="^70)
    println("Evolution complete!")
    println("="^70)
    println("Best genome found:")
    println("  Key: $(winner.key)")
    println("  Fitness: $(winner.fitness)")
    println("  Nodes: $(length(winner.nodes))")
    println("  Connections: $(length(winner.connections))")

    # Test winner thoroughly
    println("\nEvaluating winner over 100 episodes...")
    test_fitness = eval_single_genome(winner, config.genome_config, 100)
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

    # Visualization option
    println("\nVisualize the winner? (requires display and mujoco) [y/N]: ")
    response = lowercase(strip(readline()))

    if response == "y" || response == "yes"
        try
            visualize_genome(winner, config, 5)
        catch e
            println("Visualization failed: $e")
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

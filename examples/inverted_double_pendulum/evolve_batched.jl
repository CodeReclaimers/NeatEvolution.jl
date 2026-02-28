"""
Inverted Double Pendulum with Batched Parallel Evaluation.

This version reduces communication overhead by evaluating genomes in batches.
Each worker evaluates multiple genomes per task, reducing serialization costs.

Usage:
```bash
julia -p 16 --project examples/inverted_double_pendulum/evolve_batched.jl
```

This is optimized for large populations on many-core systems.
"""

using Distributed

if nworkers() == 1 && nprocs() == 1
    println("Warning: Running with single process. Use: julia -p N --project")
    println("Starting workers automatically...")
    addprocs(min(16, Sys.CPU_THREADS ÷ 2))
end

println("Active workers: $(nworkers())")

@everywhere begin
    using NeatEvolution
    using PyCall

    const gym = PyNULL()

    function __init__()
        copy!(gym, pyimport("gymnasium"))
    end

    __init__()
end

# Evaluate a single genome (runs on worker)
@everywhere function eval_single_genome(genome, genome_config, episodes::Int)
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

# Evaluate a batch of genomes (runs on worker)
@everywhere function eval_genome_batch(genome_batch, genome_config, episodes::Int)
    fitnesses = Float64[]
    for (genome_id, genome) in genome_batch
        fitness = eval_single_genome(genome, genome_config, episodes)
        push!(fitnesses, fitness)
    end
    return fitnesses
end

"""
Batched fitness evaluation with automatic batch sizing.
Reduces communication overhead by sending batches of genomes to workers.
"""
function eval_genomes(genomes, config)
    genome_list = collect(genomes)
    n_genomes = length(genome_list)
    n_workers = nworkers()

    # Determine optimal batch size
    # Larger batches = less overhead, but less load balancing
    # Target: ~4 batches per worker for good load balancing
    batches_per_worker = 4
    target_batches = n_workers * batches_per_worker
    batch_size = max(1, n_genomes ÷ target_batches)

    # Create batches
    batches = []
    for i in 1:batch_size:n_genomes
        batch_end = min(i + batch_size - 1, n_genomes)
        push!(batches, genome_list[i:batch_end])
    end

    # Distribute batches across workers
    batch_results = pmap(batches) do batch
        eval_genome_batch(batch, config.genome_config, 10)
    end

    # Flatten results and assign fitness
    idx = 1
    for batch_fitnesses in batch_results
        for fitness in batch_fitnesses
            genome_list[idx][2].fitness = fitness
            idx += 1
        end
    end
end

"""
Visualize genome performance.
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
    println("NEAT Inverted Double Pendulum (Batched Parallel)")
    println("="^70)
    println("Workers: $(nworkers())")
    println("Strategy: Batch processing to reduce communication overhead")
    println("="^70)

    config_path = joinpath(@__DIR__, "config_aggressive.toml")
    config = load_config(config_path)

    pop = Population(config)
    add_reporter!(pop, StdOutReporter(true))

    println("\nStarting batched parallel evolution...")
    println("Genomes are evaluated in batches to minimize overhead")
    println()

    # Time the first generation to estimate throughput
    start_time = time()
    winner = run!(pop, eval_genomes, 500)

    println("\n" * "="^70)
    println("Evolution complete!")
    println("="^70)
    println("Best genome:")
    println("  Key: $(winner.key)")
    println("  Fitness: $(winner.fitness)")
    println("  Nodes: $(length(winner.nodes))")
    println("  Connections: $(length(winner.connections))")

    println("\nTesting winner over 100 episodes...")
    test_fitness = eval_single_genome(winner, config.genome_config, 100)
    println("  Average reward: $test_fitness")

    if test_fitness >= 8000.0
        println("\n🎉 Excellent solution! (>= 8000)")
    elseif test_fitness >= 5000.0
        println("\n✓ Good solution! (>= 5000)")
    elseif test_fitness >= 2000.0
        println("\n+ Decent solution (>= 2000)")
    else
        println("\n  Needs improvement (target: 5000+)")
    end

    println("\nVisualize? [y/N]: ")
    if lowercase(strip(readline())) in ["y", "yes"]
        try
            visualize_genome(winner, config, 5)
        catch e
            println("Visualization failed: $e")
        end
    end

    println("\n" * "="^70)
    println("Complete!")
    println("="^70)

    return winner
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

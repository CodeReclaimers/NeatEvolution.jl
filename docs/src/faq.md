# Frequently Asked Questions (FAQ)

This document answers common questions about using NeatEvolution.jl for neuroevolution.

## Table of Contents

1. [General Questions](#general-questions)
2. [When to Use NEAT](#when-to-use-neat)
3. [Performance and Expectations](#performance-and-expectations)
4. [Network Architecture](#network-architecture)
5. [Training and Evolution](#training-and-evolution)
6. [Integration and Deployment](#integration-and-deployment)
7. [Comparison with Other Methods](#comparison-with-other-methods)

---

## General Questions

### What is NEAT?

**NEAT** (NeuroEvolution of Augmenting Topologies) is a genetic algorithm for evolving artificial neural networks. Unlike traditional neural network training methods (like backpropagation), NEAT:

- **Evolves both topology and weights** - Networks start simple and grow more complex
- **Uses genetic algorithms** - No gradient computation required
- **Discovers novel architectures** - Evolution finds optimal network structures
- **Works without labeled data** - Only requires a fitness function

**Original paper**: Stanley, K. O., & Miikkulainen, R. (2002). "Evolving Neural Networks through Augmenting Topologies"

### Is NeatEvolution.jl ready for production use?

**Yes**, NeatEvolution.jl v0.1.0 is a complete, tested implementation suitable for:

- ✅ Research and experimentation
- ✅ Game AI and control tasks
- ✅ Evolutionary robotics
- ✅ Function approximation without gradients
- ✅ Problems with sparse or delayed rewards

**Not recommended for**:
- ❌ Large-scale image classification (use CNNs instead)
- ❌ Natural language processing (use transformers instead)
- ❌ Problems where deep learning excels with abundant labeled data

### What are the main differences from neat-python?

NeatEvolution.jl is a faithful Julia port of neat-python with these key differences:

**Similarities**:
- Same NEAT algorithm implementation
- Compatible configuration format (TOML)
- Same species/stagnation mechanisms
- Similar API design

**Differences**:
- **Language**: Julia instead of Python (typically faster execution)
- **Type system**: Leverages Julia's type system for performance
- **Visualization**: Uses Plots.jl and GraphMakie instead of matplotlib/graphviz
- **Extensions**: Uses Julia's package extension system for optional dependencies
- **Network types**: Supports feed-forward, recurrent (`RecurrentNetwork`), continuous-time recurrent (`CTRNNNetwork`), and Izhikevich spiking (`IZNNNetwork`) networks

### How do I cite NeatEvolution.jl in academic work?

Cite both the original NEAT paper and this implementation:

```bibtex
@article{stanley2002evolving,
  title={Evolving neural networks through augmenting topologies},
  author={Stanley, Kenneth O and Miikkulainen, Risto},
  journal={Evolutionary computation},
  volume={10},
  number={2},
  pages={99--127},
  year={2002},
  publisher={MIT Press}
}

@software{neat_jl,
  title={NeatEvolution.jl: NeuroEvolution of Augmenting Topologies in Julia},
  author={CodeReclaimers},
  year={2025},
  url={https://github.com/CodeReclaimers/NeatEvolution.jl}
}
```

---

## When to Use NEAT

### When should I use NEAT instead of deep learning?

Use **NEAT** when:

✅ **No gradient available**
- Fitness function is non-differentiable
- Reward signal is sparse or delayed
- Evaluation involves simulation/games

✅ **Small to medium networks sufficient**
- Problem doesn't require thousands of parameters
- Interpretability matters
- Want minimal network complexity

✅ **Topology matters**
- Network structure is part of the solution
- Want to discover novel architectures
- Recurrent connections may help

✅ **Limited data**
- No large labeled dataset
- Only have fitness/reward function
- Reinforcement learning scenario

Use **Deep Learning** when:
- Large labeled datasets available
- Problem proven to work with standard architectures (ResNet, BERT, etc.)
- Need very deep networks (100+ layers)
- Gradient-based optimization works well

### Can NEAT solve reinforcement learning problems?

**Yes!** NEAT is excellent for RL tasks, especially:

- **Continuous control**: Cart-pole, bipedal walker, robot control
- **Game playing**: Simple games, Atari (with feature extraction), board games
- **Navigation**: Maze solving, path planning
- **Optimization**: Function optimization, parameter tuning

**Advantages over traditional RL**:
- No gradient computation needed
- Handles sparse rewards naturally
- Evolves network structure to fit problem
- Good for problems with delayed rewards

**Example**: See `examples/cartpole.jl` for OpenAI Gym integration

### What types of problems is NEAT good at?

NEAT excels at:

1. **Control Tasks**
   - Robot control
   - Vehicle control
   - Game controllers
   - Balancing tasks (cart-pole, inverted pendulum)

2. **Game AI**
   - Strategy games
   - Action games
   - NPCs and opponents
   - Procedural behavior

3. **Function Approximation**
   - XOR and logical functions
   - Symbolic regression
   - Pattern classification (small datasets)
   - Time series prediction

4. **Optimization Problems**
   - Parameter optimization
   - Structural design
   - Multi-objective optimization

5. **Creative Applications**
   - Generative art
   - Music generation
   - Procedural content generation

### What problems is NEAT NOT good for?

Avoid NEAT for:

❌ **Large-scale vision tasks**
- Image classification (ImageNet-scale)
- Object detection
- Semantic segmentation
→ Use CNNs instead

❌ **Natural Language Processing**
- Machine translation
- Language modeling
- Text generation
→ Use transformers instead

❌ **Very deep architectures needed**
- Problems requiring 100+ layers
- Tasks proven to need deep networks
→ Use deep learning frameworks

❌ **Abundant labeled data available**
- When you have millions of labeled examples
- When supervised learning works well
→ Use standard neural networks with backprop

❌ **Real-time learning required**
- Need to learn from single examples
- Online learning scenarios
→ Use online learning algorithms

---

## Performance and Expectations

### How many generations does NEAT typically need?

Typical generation counts by problem complexity:

**Simple problems** (10-100 generations):
- XOR: 50-200 generations
- Simple control: 50-150 generations
- Basic function approximation: 50-200 generations

**Medium problems** (100-500 generations):
- Cart-pole: 100-300 generations
- Simple games: 200-500 generations
- Pattern classification: 150-400 generations

**Complex problems** (500-5000+ generations):
- Complex control tasks: 500-2000 generations
- Difficult games: 1000-5000 generations
- Multi-objective problems: 1000+ generations

**Factors affecting convergence**:
- Problem difficulty
- Population size
- Fitness function design
- Mutation rates
- Species settings

### How large should my population be?

**General guidelines**:

```toml
# Simple problems (XOR, basic control)
pop_size = 50-150

# Medium complexity
pop_size = 150-500

# Complex problems
pop_size = 500-1000

# Very complex problems
pop_size = 1000-5000
```

**Trade-offs**:
- **Larger populations**:
  - ✅ More genetic diversity
  - ✅ Better exploration
  - ✅ More stable evolution
  - ❌ Slower per generation
  - ❌ More memory usage

- **Smaller populations**:
  - ✅ Faster generations
  - ✅ Less memory
  - ❌ Less diversity
  - ❌ May get stuck in local optima
  - ❌ Higher extinction risk

**Rule of thumb**: Start with 150, increase if you see complete extinction or poor diversity.

### Why is evolution so slow?

If evolution takes too long, check:

1. **Fitness function is slow**
   ```julia
   # Profile your fitness function
   @time eval_genomes(genomes, config)
   ```
   - Is evaluation the bottleneck?
   - Can you vectorize operations?
   - Can you reduce test cases?

2. **Population too large**
   ```toml
   pop_size = 1000  # Try reducing to 150-500
   ```

3. **Networks too complex**
   - Enable `single_structural_mutation = true`
   - Increase deletion rates
   - Add complexity penalty to fitness

4. **Inefficient network construction**
   ```julia
   # Bad - reconstructs every time
   for input in test_cases
       net = FeedForwardNetwork(genome, config)
       output = activate!(net, input)
   end

   # Good - construct once
   net = FeedForwardNetwork(genome, config)
   for input in test_cases
       output = activate!(net, input)
   end
   ```

### How do I know if evolution is working?

**Good signs**:
- ✅ Best fitness increasing over time (check with `plot_fitness`)
- ✅ Fitness standard deviation > 0 (genomes differ in performance)
- ✅ Multiple species maintained (typically 5-20)
- ✅ Species sizes relatively balanced
- ✅ Network complexity increasing gradually

**Bad signs**:
- ❌ Fitness stuck at same value for 50+ generations
- ❌ All genomes have identical fitness
- ❌ CompleteExtinctionException
- ❌ Too many species (>30 with pop_size=150)
- ❌ Network complexity explosion

**Monitor progress**:
```julia
# Add reporters
add_reporter!(pop, StdOutReporter(true))  # Detailed output
stats = StatisticsReporter()
add_reporter!(pop, stats)

# Run evolution
winner = run!(pop, eval_genomes, 100)

# Visualize
using Plots
plot_fitness(stats, filename="fitness.png")
plot_species(stats, filename="species.png")
```

---

## Network Architecture

### Should I use feed-forward or recurrent networks?

**Feed-forward networks** (`feed_forward = true`):

Use for:
- ✅ Static input → output mappings
- ✅ Pattern classification
- ✅ Function approximation
- ✅ Problems without temporal dependencies

Advantages:
- Faster evaluation (no recurrent loops)
- Simpler structure
- Easier to visualize
- Guaranteed to compute in finite time

**Recurrent networks** (`feed_forward = false`):

Use for:
- ✅ Temporal/sequential problems
- ✅ Memory requirements
- ✅ Time-series prediction
- ✅ Control tasks with state

Advantages:
- Can maintain internal state
- Model temporal patterns
- More expressive
- Better for control tasks

**Start with feed-forward** unless you know you need recurrency.

### What activation functions should I use?

**Common choices**:

```toml
# For outputs in [0, 1]: use sigmoid
activation_default = "sigmoid"
activation_options = ["sigmoid"]

# For outputs in [-1, 1]: use tanh
activation_default = "tanh"
activation_options = ["tanh"]

# For unbounded outputs: use identity or relu
activation_default = "relu"
activation_options = ["relu", "identity"]

# Let evolution choose (recommended for exploration):
activation_default = "sigmoid"
activation_options = ["sigmoid", "tanh", "relu", "sin", "gauss"]
activation_mutate_rate = 0.1
```

**Activation function guide**:

- **sigmoid**: Output in [0, 1], smooth, good for probabilities
- **tanh**: Output in [-1, 1], smooth, centered at 0
- **relu**: Output in [0, ∞), sparse, widely used in deep learning
- **identity**: Linear, output = input
- **sin**: Periodic, useful for oscillatory behaviors
- **gauss**: Bell curve, useful for localized responses

**Pro tip**: Start with a single activation function for simplicity, then allow multiple options if evolution struggles.

### How many hidden nodes should I start with?

**Recommended**: Start with **zero hidden nodes**

```toml
[DefaultGenome]
num_hidden = 0
initial_connection = "full_nodirect"  # or "full_direct"
```

**Why?**
- NEAT's strength is evolving topology
- Networks start minimal and grow as needed
- Prevents unnecessary complexity
- Let evolution discover optimal structure

**Alternative approaches**:

```toml
# Start with direct connections (no hidden layer)
num_hidden = 0
initial_connection = "full_direct"

# Start with complete connections through hidden layer
num_hidden = 0
initial_connection = "full_nodirect"

# Start with no connections (fully minimal)
num_hidden = 0
initial_connection = "unconnected"
```

**Rarely needed**: Starting with hidden nodes (only if you know the problem requires them)

### What does "initial_connection" mean?

Controls how genomes are initialized:

**Options**:

1. **`"full_direct"`** - All inputs directly connected to outputs
   ```
   Input → Output (no hidden nodes)
   ```
   - Fast initial evaluation
   - Good for simple problems
   - Evolution adds complexity as needed

2. **`"full_nodirect"`** - Complete connectivity but no direct input→output
   ```
   Input → [Hidden] → Output
   ```
   - Forces at least one hidden layer
   - Better for problems requiring transformation
   - Still allows evolution to grow

3. **`"unconnected"`** - No initial connections
   ```
   Input    Output (disconnected)
   ```
   - Fully minimal start
   - Evolution must build everything
   - Slowest initial progress

4. **`"partial"`** - Randomly connected (with probability)
   ```
   Some Input → Output connections
   ```
   - Varied initial population
   - Balance between minimal and connected

**Recommendation**: Use `"full_direct"` for most problems.

---

## Training and Evolution

### What's a good fitness function?

**Key principles**:

1. **Always return positive values**
   ```julia
   # Bad
   g.fitness = prediction - target  # Can be negative!

   # Good
   g.fitness = max(0.001, 1.0 - abs(prediction - target))
   ```

2. **Provide smooth gradient**
   ```julia
   # Bad - binary fitness
   g.fitness = correct ? 1.0 : 0.0  # No guidance

   # Good - continuous fitness
   g.fitness = 1.0 - abs(output - target)  # Smooth gradient
   ```

3. **Differentiate genomes**
   - Ensure different genomes get different fitness values
   - Check fitness standard deviation > 0

4. **Higher is better**
   ```julia
   # Error-based: subtract from maximum
   error = calculate_error(net)
   g.fitness = 1.0 - error

   # Reward-based: accumulate rewards
   total_reward = simulate(net)
   g.fitness = total_reward
   ```

5. **Normalize to reasonable range**
   ```julia
   # Scale fitness to [0, 1] or [0, 100]
   raw_fitness = evaluate(net)
   g.fitness = raw_fitness / max_possible_fitness
   ```

**Examples**:

```julia
# Classification
function eval_genomes(genomes, config)
    for (_, g) in genomes
        net = FeedForwardNetwork(g, config.genome_config)

        correct = 0
        for (input, target) in training_data
            output = activate!(net, input)[1]
            predicted = output > 0.5 ? 1 : 0
            if predicted == target
                correct += 1
            end
        end

        # Fitness proportional to accuracy
        g.fitness = correct / length(training_data)
    end
end

# Control task (maximize reward)
function eval_genomes(genomes, config)
    for (_, g) in genomes
        net = FeedForwardNetwork(g, config.genome_config)

        total_reward = 0.0
        for episode in 1:num_episodes
            total_reward += run_simulation(net)
        end

        g.fitness = total_reward / num_episodes
    end
end
```

### How do I handle stochastic fitness functions?

If fitness varies between evaluations:

**Solution 1: Average multiple evaluations**
```julia
function eval_genomes(genomes, config)
    for (_, g) in genomes
        net = FeedForwardNetwork(g, config.genome_config)

        # Run multiple episodes with different random seeds
        fitnesses = Float64[]
        for trial in 1:num_trials
            push!(fitnesses, evaluate_with_seed(net, trial))
        end

        # Use mean fitness
        g.fitness = mean(fitnesses)
    end
end
```

**Solution 2: Use fixed random seeds**
```julia
function eval_genomes(genomes, config)
    # Use same seeds for all genomes (fair comparison)
    seeds = [123, 456, 789]

    for (_, g) in genomes
        net = FeedForwardNetwork(g, config.genome_config)

        total = 0.0
        for seed in seeds
            total += evaluate_with_seed(net, seed)
        end

        g.fitness = total / length(seeds)
    end
end
```

**Trade-off**: More trials = more stable but slower evaluation

### Why does fitness go down sometimes?

**This is normal!** Fitness can decrease due to:

1. **Fitness sharing** - Fitness adjusted based on species size
2. **Species formation** - New species split populations
3. **Mutation exploration** - Some mutations hurt performance
4. **Stochastic fitness** - Random variation in evaluation

**When to worry**:
- Fitness never improves for 50+ generations
- Fitness drops to near zero
- Best fitness decreasing consistently

**Monitor overall trends**:
```julia
using Plots
plot_fitness(stats, filename="fitness_trend.png")
# Look for upward trend in best fitness
```

### Can I parallelize fitness evaluation?

**Yes!** NeatEvolution.jl fitness evaluation is embarrassingly parallel:

```julia
using Distributed

# Add worker processes
addprocs(4)

@everywhere using NeatEvolution
@everywhere include("my_fitness.jl")

function eval_genomes_parallel(genomes, config)
    genome_list = collect(genomes)

    # Evaluate in parallel
    fitnesses = pmap(genome_list) do (genome_id, genome)
        net = FeedForwardNetwork(genome, config.genome_config)
        evaluate(net)
    end

    # Assign fitnesses back
    for (i, (genome_id, genome)) in enumerate(genome_list)
        genome.fitness = fitnesses[i]
    end
end

# Use parallel evaluation
winner = run!(pop, eval_genomes_parallel, 100)
```

**Expected speedup**: Nearly linear with number of cores for expensive fitness functions.

---

## Integration and Deployment

### How do I export trained networks?

NeatEvolution.jl provides JSON export for framework-agnostic model sharing:

```julia
using NeatEvolution

# After training
winner = run!(pop, eval_genomes, 100)

# Export single network
export_network_json(winner, config.genome_config, "winner.json")

# Export top 10 genomes
export_population_json(pop.population, config.genome_config,
                       "top10.json", top_n=10)

# Import later
imported = import_network_json("winner.json", config.genome_config)
net = FeedForwardNetwork(imported, config.genome_config)
```

**JSON format** is framework-agnostic and can be loaded by:
- Other Julia code
- Python implementations
- JavaScript (for web deployment)
- Any language with JSON support

### Can I integrate NeatEvolution.jl with OpenAI Gym?

**Yes!** See `examples/gym_integration.jl` for a complete example:

```julia
using NeatEvolution
using PythonCall

gym = pyimport("gym")

function eval_genomes(genomes, config)
    for (_, g) in genomes
        net = FeedForwardNetwork(g, config.genome_config)

        # Create environment
        env = gym.make("CartPole-v1")
        obs = env.reset()

        total_reward = 0.0
        done = false

        while !done
            # Get network output
            output = activate!(net, pyconvert(Vector{Float64}, obs))
            action = output[1] > 0.5 ? 1 : 0

            # Step environment
            obs, reward, done, info = env.step(action)
            total_reward += reward
        end

        env.close()
        g.fitness = total_reward
    end
end
```

### How do I use a trained network in production?

Two options:

**Option 1: Use NeatEvolution.jl directly**
```julia
using NeatEvolution

# Load saved network
config = load_config("config.toml")
genome = import_network_json("winner.json", config.genome_config)
net = FeedForwardNetwork(genome, config.genome_config)

# Use in production
function predict(input_data)
    output = activate!(net, input_data)
    return output
end
```

**Option 2: Export to standalone format**
```julia
# Export to JSON
export_network_json(winner, config.genome_config, "model.json")

# Implement lightweight interpreter in production language
# JSON contains all nodes, connections, weights, activation functions
```

### Can I continue evolution from a checkpoint?

**Yes!** NeatEvolution.jl has built-in checkpointing:

```julia
# Add checkpointer during evolution
add_reporter!(pop, Checkpointer(generation_interval=10))

# Later: restore and continue
pop = restore_checkpoint("neat-checkpoint-50")
add_reporter!(pop, StdOutReporter())
winner = run!(pop, eval_genomes, 50)  # run 50 more generations
```

You can also manually save/restore:

```julia
save_checkpoint("my_checkpoint", pop)
pop = restore_checkpoint("my_checkpoint")
```

---

## Comparison with Other Methods

### NEAT vs Backpropagation?

**Backpropagation** (gradient descent):
- ✅ Very efficient with abundant data
- ✅ Proven architectures exist (ResNet, BERT, etc.)
- ✅ Fast training on GPUs
- ❌ Requires differentiable fitness
- ❌ Needs fixed architecture
- ❌ Can get stuck in local minima

**NEAT**:
- ✅ No gradient needed
- ✅ Evolves architecture
- ✅ Works with sparse rewards
- ✅ Handles non-differentiable fitness
- ❌ Slower than backprop (for same data)
- ❌ Requires many evaluations
- ❌ Less sample efficient

**Use backprop if**: You have lots of labeled data and standard architectures work

**Use NEAT if**: Fitness is non-differentiable or you need to discover architecture

### NEAT vs Other Neuroevolution Methods?

**NEAT advantages**:
- Evolves topology (not just weights)
- Speciation protects innovation
- Historical markings enable crossover
- Start minimal, grow complexity

**Alternatives**:

- **HyperNEAT**: NEAT + geometric patterns (for spatial problems)
- **ES-HyperNEAT**: + evolvable substrate
- **CPPN-NEAT**: Compositional pattern producing networks
- **CoDeepNEAT**: Evolves deep learning architectures

**NEAT is best for**: General-purpose neuroevolution with moderate network sizes

### NEAT vs Reinforcement Learning (PPO, DQN)?

**Modern RL (PPO, DQN, etc.)**:
- ✅ State-of-the-art on many benchmarks
- ✅ Sample efficient (with experience replay)
- ✅ Proven on Atari, robotics
- ❌ Requires careful hyperparameter tuning
- ❌ Can be unstable
- ❌ Needs differentiable components

**NEAT**:
- ✅ Simpler to set up
- ✅ More stable evolution
- ✅ No backprop needed
- ✅ Good for simpler control tasks
- ❌ Less sample efficient
- ❌ Not state-of-the-art on complex benchmarks

**Use modern RL if**: You have computational resources and need state-of-the-art results

**Use NEAT if**: You want simplicity and stability for moderate-complexity tasks

---

## See Also

- [Getting Started Guide](getting_started.md) - Tutorial and basic usage
- [Troubleshooting Guide](troubleshooting.md) - Common problems and solutions
- [Configuration Reference](config_file.md) - All configuration parameters
- [Algorithm Internals](algorithm_internals.md) - How NEAT works under the hood

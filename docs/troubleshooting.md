# Troubleshooting Guide

This guide addresses common problems and errors you may encounter when using NEAT.jl.

## Table of Contents

1. [Evolution Not Progressing](#evolution-not-progressing)
2. [All Species Go Extinct](#all-species-go-extinct)
3. [Network Complexity Explosion](#network-complexity-explosion)
4. [Fitness Function Issues](#fitness-function-issues)
5. [Configuration Errors](#configuration-errors)
6. [Performance Problems](#performance-problems)
7. [Visualization Issues](#visualization-issues)

---

## Evolution Not Progressing

### Symptoms
- Fitness stuck at same value for many generations
- Average fitness not improving
- Best fitness not increasing

### Common Causes and Solutions

#### 1. **Fitness Function Not Differentiating**

**Problem**: All genomes get same or very similar fitness values.

**Diagnosis**:
```julia
stats = StatisticsReporter()
add_reporter!(pop, stats)
winner = run!(pop, eval_genomes, 10)

# Check fitness variance
mean_fitness = get_fitness_mean(stats)
stdev_fitness = get_fitness_stdev(stats)
println("Mean: $mean_fitness")
println("Stdev: $stdev_fitness")  # Should be > 0
```

**Solutions**:
- Ensure fitness function returns different values for different genomes
- Add more granularity to fitness calculation
- Check that your evaluation correctly reflects performance differences
- Verify input data is being properly fed to networks

**Example Fix**:
```julia
# Bad - not enough differentiation
function eval_genomes(genomes, config)
    for (_, g) in genomes
        net = FeedForwardNetwork(g, config.genome_config)
        # Only checks if output is close to target
        g.fitness = abs(activate!(net, [1.0])[1] - 0.5) < 0.1 ? 1.0 : 0.0
    end
end

# Good - smooth gradient
function eval_genomes(genomes, config)
    for (_, g) in genomes
        net = FeedForwardNetwork(g, config.genome_config)
        output = activate!(net, [1.0])[1]
        # Continuous fitness based on distance
        g.fitness = 1.0 - abs(output - 0.5)
    end
end
```

#### 2. **Population Too Small**

**Problem**: Not enough genetic diversity to explore solution space.

**Solution**: Increase `pop_size` in config:
```toml
[NEAT]
pop_size = 150  # Increase from 50
```

**Guidelines**:
- Simple problems (XOR): 50-150
- Medium problems: 150-500
- Complex problems: 500-1000+

#### 3. **Compatibility Threshold Too Low**

**Problem**: Too many species, each too small to evolve effectively.

**Diagnosis**:
```julia
# Add StdOutReporter to see species count
add_reporter!(pop, StdOutReporter(true))
```

**Solution**: Increase `compatibility_threshold`:
```toml
[DefaultSpeciesSet]
compatibility_threshold = 4.0  # Increase from 3.0
```

If you see >20 species with population of 150, threshold is too low.

#### 4. **Wrong Network Type**

**Problem**: Using feed-forward network for problem requiring recurrency.

**Solution**: Try recurrent networks:
```toml
[DefaultGenome]
feed_forward = false  # Allow recurrent connections
```

Use recurrent networks for:
- Temporal/sequential problems
- Memory requirements
- Time-series prediction
- Control tasks with state

#### 5. **Activation Functions Don't Fit Problem**

**Problem**: Activation functions don't match output range needed.

**Solutions**:
```toml
# For outputs in [0, 1]: use sigmoid
activation_default = "sigmoid"

# For outputs in [-1, 1]: use tanh
activation_default = "tanh"

# For unbounded outputs: use identity or relu
activation_default = "identity"

# Or let evolution choose:
activation_options = ["sigmoid", "tanh", "relu"]
activation_mutate_rate = 0.1
```

---

## All Species Go Extinct

### Symptoms
```
Error: CompleteExtinctionException()
```

All species removed due to stagnation, leaving no genomes.

### Common Causes and Solutions

#### 1. **Negative or Zero Fitness Values**

**Problem**: NEAT requires positive fitness values for fitness sharing.

**Bad Example**:
```julia
function eval_genomes(genomes, config)
    for (_, g) in genomes
        # Error prone fitness can be negative!
        g.fitness = prediction - target  # Can be negative
    end
end
```

**Good Example**:
```julia
function eval_genomes(genomes, config)
    for (_, g) in genomes
        error = abs(prediction - target)
        # Ensure positive fitness
        g.fitness = max(0.001, 1.0 - error)
    end
end
```

**Quick Fix**: Add offset to ensure positive values:
```julia
g.fitness = raw_fitness + 1000.0  # Ensure positive
```

#### 2. **Population Too Small**

**Problem**: Small populations lose all species quickly.

**Solution**: Increase population size:
```toml
[NEAT]
pop_size = 150  # Minimum for stability
```

#### 3. **max_stagnation Too Low**

**Problem**: Species removed before they can improve.

**Solution**: Increase patience:
```toml
[DefaultStagnation]
max_stagnation = 20  # Increase from 15
```

#### 4. **Compatibility Threshold Too Low**

**Problem**: Each genome becomes its own species, all species small and vulnerable.

**Solution**: Increase threshold to create fewer, larger species:
```toml
[DefaultSpeciesSet]
compatibility_threshold = 4.0  # Increase
```

#### 5. **Enable Extinction Reset**

**Solution**: Allow automatic population reset on extinction:
```toml
[NEAT]
reset_on_extinction = true
```

This creates a new random population if extinction occurs.

---

## Network Complexity Explosion

### Symptoms
- Networks grow to hundreds of nodes
- Connections explode exponentially
- Evaluation becomes very slow
- Visualization shows incomprehensible networks

### Causes and Solutions

#### 1. **Structural Mutation Rates Too High**

**Problem**: Adding nodes/connections faster than deleting them.

**Solution**: Balance mutation rates:
```toml
[DefaultGenome]
# Reduce addition rates
conn_add_prob = 0.3  # Reduce from 0.5
node_add_prob = 0.2  # Reduce from 0.2

# Increase deletion rates
conn_delete_prob = 0.2  # Increase from 0.05
node_delete_prob = 0.1  # Increase from 0.03
```

#### 2. **Enable Single Structural Mutation**

**Solution**: Allow only one structural mutation per genome per generation:
```toml
[DefaultGenome]
single_structural_mutation = true
```

This prevents complexity from growing too fast.

#### 3. **No Initial Connections**

**Problem**: Starting with no hidden nodes forces evolution to add many.

**Solution**: Start simpler:
```toml
[DefaultGenome]
initial_connection = "full_nodirect"  # Or "full_direct"
num_hidden = 0  # Start with no hidden nodes
```

#### 4. **Add Complexity Penalty to Fitness**

**Solution**: Penalize large networks:
```julia
function eval_genomes(genomes, config)
    for (_, g) in genomes
        net = FeedForwardNetwork(g, config.genome_config)

        # Your normal fitness calculation
        base_fitness = evaluate(net)

        # Add complexity penalty
        num_nodes = length(g.nodes)
        num_connections = count(c -> c.enabled, values(g.connections))
        complexity_penalty = 0.001 * (num_nodes + num_connections)

        g.fitness = base_fitness - complexity_penalty
    end
end
```

---

## Fitness Function Issues

### Problem 1: Fitness Not Set

**Error**: Some genomes have `fitness = nothing`

**Cause**: Fitness function doesn't set fitness for all genomes.

**Solution**:
```julia
function eval_genomes(genomes, config)
    for (genome_id, genome) in genomes
        # MUST set fitness for every genome
        net = FeedForwardNetwork(genome, config.genome_config)
        genome.fitness = evaluate(net)

        # Ensure it's set
        @assert genome.fitness !== nothing "Fitness not set for genome $genome_id"
    end
end
```

### Problem 2: Fitness Goes Down Over Time

**This is Normal!** Fitness sharing causes this:

- Species adjust fitness based on size
- New species form, splitting populations
- Species can temporarily lose members

**Don't worry unless**:
- Fitness never improves for 50+ generations
- Fitness drops to near zero

**Monitor with**:
```julia
using Plots
plot_fitness(stats, filename="fitness_check.png")
```

Look for overall upward trend in best fitness.

### Problem 3: Fitness Always Zero or Same Value

**Cause**: Logic error in fitness function.

**Debug**:
```julia
function eval_genomes(genomes, config)
    for (genome_id, genome) in genomes
        net = FeedForwardNetwork(genome, config.genome_config)

        # Debug: Print outputs for first genome
        if genome_id == 1
            outputs = [activate!(net, input) for input in test_inputs]
            println("Outputs: $outputs")
        end

        genome.fitness = calculate_fitness(net)

        # Debug: Check fitness values
        println("Genome $genome_id: fitness = $(genome.fitness)")
    end
end
```

---

## Configuration Errors

### Error: "Unknown configuration parameter"

**Cause**: Typo in config file or unsupported parameter.

**Solution**: Check spelling against [Configuration Reference](config_file.md).

Common typos:
```toml
# Wrong
compatability_threshold = 3.0  # Missing 'i'

# Correct
compatibility_threshold = 3.0
```

### Error: "Parameter X must be positive"

**Cause**: Invalid parameter value.

**Solution**: Check parameter constraints in config reference.

Example:
```toml
# Wrong
pop_size = -10

# Correct
pop_size = 150
```

### Error: "Missing required parameter"

**Cause**: Required parameter not specified in config.

**Solution**: Add missing parameter. Check error message for which parameter.

---

## Performance Problems

### Problem: Evolution Very Slow

#### Solution 1: Profile Fitness Function

Your fitness function is likely the bottleneck:

```julia
using Profile

function eval_genomes(genomes, config)
    @time begin  # Time the entire evaluation
        for (_, g) in genomes
            net = FeedForwardNetwork(g, config.genome_config)
            g.fitness = evaluate(net)  # Profile this
        end
    end
end
```

#### Solution 2: Reduce Population or Generations

```toml
[NEAT]
pop_size = 100  # Reduce from 500
```

#### Solution 3: Simplify Network Construction

If you're creating networks multiple times, cache them:

```julia
# Bad - reconstructs network each time
function evaluate(genome, config, test_inputs)
    total_error = 0.0
    for input in test_inputs
        net = FeedForwardNetwork(genome, config)  # Expensive!
        output = activate!(net, input)
        total_error += calculate_error(output)
    end
    return -total_error
end

# Good - construct once
function evaluate(genome, config, test_inputs)
    net = FeedForwardNetwork(genome, config)  # Once
    total_error = 0.0
    for input in test_inputs
        output = activate!(net, input)
        total_error += calculate_error(output)
    end
    return -total_error
end
```

### Problem: Memory Usage Too High

**Solution**: Export only needed genomes:
```julia
# Export only winner, not entire population
export_network_json(winner, config.genome_config, "winner.json")

# Or top 10
export_population_json(pop.population, config.genome_config,
                       "top10.json", top_n=10)
```

---

## Visualization Issues

### Problem: Plots.jl Not Found

**Solution**: Install Plots.jl:
```julia
using Pkg
Pkg.add("Plots")
```

Then load it:
```julia
using NEAT
using Plots  # Must come after NEAT

plot_fitness(stats, filename="fitness.png")
```

### Problem: GraphMakie Tests Skipped in CI

**This is expected!** GraphMakie requires display capabilities not available in CI.

Tests run locally but skip in CI environments.

### Problem: Network Diagram Too Cluttered

**Solution 1**: Prune unused nodes:
```julia
draw_net(genome, config.genome_config,
         prune_unused=true,
         show_disabled=false)
```

**Solution 2**: Export to JSON and visualize elsewhere:
```julia
export_network_json(genome, config.genome_config, "network.json")
# Use external visualization tools
```

---

## Debugging Checklist

When evolution isn't working, check these in order:

1. ✓ **Fitness function**
   - Returns positive values?
   - Sets fitness for ALL genomes?
   - Shows variation (stdev > 0)?
   - Higher fitness = better performance?

2. ✓ **Configuration**
   - Population size adequate? (≥150)
   - Compatibility threshold reasonable? (2.0-4.0)
   - Mutation rates balanced?
   - Correct network type (feed-forward vs recurrent)?

3. ✓ **Problem representation**
   - Inputs normalized?
   - Outputs match activation range?
   - Enough test cases?
   - Evaluation deterministic?

4. ✓ **Monitor progress**
   - Use `StdOutReporter(true)` for details
   - Use `StatisticsReporter()` for analysis
   - Generate fitness plots
   - Check species count and sizes

5. ✓ **Try simpler first**
   - Test on XOR before your problem
   - Reduce problem complexity
   - Verify setup works on known problems

---

## Getting Help

If you're still stuck:

1. **Check the documentation**:
   - [Getting Started Guide](getting_started.md)
   - [Configuration Reference](config_file.md)
   - [Algorithm Internals](algorithm_internals.md)
   - [FAQ](faq.md)

2. **Enable detailed logging**:
   ```julia
   add_reporter!(pop, StdOutReporter(true))  # Verbose output
   ```

3. **Create minimal reproducible example**:
   - Strip down to simplest case that shows the problem
   - Include config file and fitness function
   - Show error message or unexpected behavior

4. **Open an issue** on GitHub with:
   - Julia version (`VERSION`)
   - NEAT.jl version
   - Complete error message
   - Minimal reproducible code
   - What you've already tried

---

## See Also

- [FAQ](faq.md) - Frequently Asked Questions
- [Getting Started Guide](getting_started.md) - Basic usage
- [Configuration Reference](config_file.md) - All parameters
- [Algorithm Internals](algorithm_internals.md) - How NEAT works

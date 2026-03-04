# XOR Example Walkthrough

## Overview

This tutorial walks through the complete XOR example located in `examples/xor/`. The XOR (exclusive OR) problem is a classic benchmark for neural networks, demonstrating NEAT's ability to evolve both network topology and weights to solve a non-linearly separable problem.

## The XOR Problem

XOR is a boolean function with two inputs and one output:

| Input 1 | Input 2 | Output |
|---------|---------|--------|
| 0       | 0       | 0      |
| 0       | 1       | 1      |
| 1       | 0       | 1      |
| 1       | 1       | 0      |

The challenge: this function cannot be solved by a single-layer perceptron (it requires a hidden layer or non-linear topology).

## File Structure

```
examples/xor/
├── config.toml                              # NEAT configuration
├── evolve.jl                                # Basic evolution script
├── evolve_with_visualization.jl             # Evolution + static visualization
└── evolve_with_interactive_visualization.jl # Evolution + interactive visualization
```

## Step-by-Step Implementation

### Step 1: Define Test Cases

```julia
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
```

### Step 2: Implement Fitness Function

The fitness function evaluates how well a genome solves XOR:

```julia
function eval_genomes(genomes, config)
    for (genome_id, genome) in genomes
        # Start with perfect fitness
        genome.fitness = 4.0

        # Create neural network from genome
        net = FeedForwardNetwork(genome, config.genome_config)

        # Test on all XOR cases
        for (xi, xo) in zip(XOR_INPUTS, XOR_OUTPUTS)
            output = activate!(net, xi)
            # Subtract squared error
            genome.fitness -= (output[1] - xo[1])^2
        end
    end
end
```

**Fitness Calculation:**
- Start with fitness = 4.0 (perfect score)
- For each test case, subtract (predicted - expected)²
- Perfect solution: fitness = 4.0
- Random guess: fitness ≈ 2.0

**Why this works:**
- Higher fitness = better performance
- Smooth gradient guides evolution
- Zero error achieves maximum fitness

### Step 3: Create Configuration File

Create `config.toml` with these key settings:

```toml
[NEAT]
pop_size = 150          # 150 genomes per generation
fitness_criterion = "max"
fitness_threshold = 3.9  # Stop when fitness reaches 3.9

[DefaultGenome]
num_inputs = 2           # Two XOR inputs
num_outputs = 1          # One XOR output
num_hidden = 0           # Start with no hidden nodes
feed_forward = true      # No recurrent connections needed
initial_connection = "full"  # Fully connected initially

# Allow network to grow
node_add_prob = 0.2      # Can add hidden nodes
conn_add_prob = 0.5      # Can add connections
```

See [Configuration Reference](config_file.md) for all parameters.

### Step 4: Run Evolution

```julia
function main()
    # Load configuration
    config_path = joinpath(@__DIR__, "config.toml")
    config = load_config(config_path)

    # Create population
    pop = Population(config)

    # Add progress reporter
    add_reporter!(pop, StdOutReporter(true))

    # Run evolution for up to 100 generations
    println("Starting XOR evolution...")
    winner = run!(pop, eval_genomes, 100)

    # Display results
    println("\nBest genome:")
    println("  Key: $(winner.key)")
    println("  Fitness: $(round(winner.fitness, digits=5))")
    println("  Nodes: $(length(winner.nodes))")
    println("  Connections: $(length(winner.connections))")

    return winner
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
```

## Running the Example

### Basic Version

```bash
cd examples/xor
julia --project=../.. evolve.jl
```

**Expected Output:**
```
Starting XOR evolution...

****** Running generation 0 ******

Population's average fitness: 2.4521 stdev: 0.8342
Best fitness: 3.6234 - size: (3, 4) - species 1 - id 42

****** Running generation 1 ******
...

Best genome:
  Key: 42
  Fitness: 3.9876
  Nodes: 3
  Connections: 5
```

### With Visualization

```bash
julia --project=../.. evolve_with_visualization.jl
```

This generates:
- `xor_fitness.png` - Fitness evolution plot
- `xor_species.png` - Species evolution plot
- `xor_winner.png` - Winner network diagram
- `xor_activation_heatmap.png` - Decision boundary visualization
- `xor_evolution.gif` - Animation of network evolution
- CSV files with statistics

## Understanding the Results

### Testing the Winner

```julia
# Test the winning network
winner_net = FeedForwardNetwork(winner, config.genome_config)

println("Testing winner network:")
println("  Input      Expected   Output     Error")
println("  " * "-"^45)

for (xi, xo) in zip(XOR_INPUTS, XOR_OUTPUTS)
    output = activate!(winner_net, xi)
    error = abs(output[1] - xo[1])
    println("  $(xi)  $(xo)    $(round.(output, digits=4))  $(round(error, digits=4))")
end
```

**Example output:**
```
Testing winner network:
  Input      Expected   Output     Error
  ---------------------------------------------
  [0.0, 0.0]  [0.0]    [0.0123]  0.0123
  [0.0, 1.0]  [1.0]    [0.9891]  0.0109
  [1.0, 0.0]  [1.0]    [0.9876]  0.0124
  [1.0, 1.0]  [0.0]    [0.0145]  0.0145
```

### Typical Network Structure

A solution typically contains:
- **Nodes:** 3-5 (2 inputs + 1 output + 0-2 hidden)
- **Connections:** 4-8
- **Topology:** Often evolves a hidden node to capture non-linearity

**Example evolved structure:**
```
Input 1 ──┬──> Hidden ──┐
          │             ├──> Output
Input 2 ──┴─────────────┘
```

## Visualization Examples

### Fitness Evolution

*Generate with `plot_fitness(stats, filename="xor_fitness.png")` (requires Plots.jl)*

Shows:
- **Red line**: Best fitness per generation
- **Blue line**: Average fitness
- **Green band**: ±1 standard deviation

### Species Evolution

*Generate with `plot_species(stats, filename="xor_species.png")` (requires Plots.jl)*

Shows:
- Colored areas: Different species
- Y-axis: Number of genomes per species
- Demonstrates species formation and competition

### Network Diagram

*Generate with `draw_net(winner, config.genome_config, filename="xor_winner.png")` (requires Plots.jl)*

Shows:
- **Green nodes**: Inputs (x1, x2)
- **Blue nodes**: Outputs (XOR)
- **White nodes**: Hidden nodes
- **Green lines**: Positive weights (thicker = stronger)
- **Red lines**: Negative weights
- **Gray dashed**: Disabled connections

### Activation Heatmap

*Generate with `plot_activation_heatmap(winner, config.genome_config, filename="xor_heatmap.png")` (requires Plots.jl)*

Visualizes the decision boundary:
- X-axis: Input 1
- Y-axis: Input 2
- Color: Network output
- Shows clear XOR pattern (opposite corners activated)

## Typical Evolution Timeline

**Generations 0-10:**
- Random initial population
- Fitness gradually improving
- Multiple species forming

**Generations 10-30:**
- Some genomes discover partial solutions
- Hidden nodes begin to appear
- Species competition intensifies

**Generations 30-50:**
- Network topology refinement
- Weights optimizing
- Approaching fitness threshold

**Generation 50-100:**
- Solution found (fitness > 3.9)
- Final weight tuning
- Evolution terminates

## Common Patterns

### Fast Convergence

If evolution solves XOR in < 20 generations:
- Good initial luck with connections
- Effective mutation rates
- Proper species protection

### Slow Convergence

If taking > 100 generations:
- Increase `pop_size` to 200-300
- Adjust `compatibility_threshold`
- Increase `conn_add_prob` or `node_add_prob`
- Check `weight_mutate_power` isn't too large

### Stagnation

If stuck at fitness ~3.5:
- Increase `max_stagnation` to give species more time
- Increase population diversity
- Check for premature species extinction

## Advanced Usage

### Using Statistics Reporter

```julia
# Add statistics reporter
stats = StatisticsReporter()
add_reporter!(pop, stats)

# Run evolution
winner = run!(pop, eval_genomes, 100)

# Access statistics
println("Final average fitness: ", get_fitness_mean(stats)[end])
println("Best fitness over time: ", [g.fitness for g in stats.most_fit_genomes])

# Get species information
species_sizes = get_species_sizes(stats)
println("Species count per generation: ", [count(x > 0, sizes) for sizes in species_sizes])
```

### Custom Node Names

```julia
node_names = Dict(
    -1 => "Input A",
    -2 => "Input B",
    0 => "XOR Output"
)

draw_net(winner, config.genome_config,
         filename="xor_custom.png",
         node_names=node_names)
```

### Saving/Loading Genomes

The preferred approaches for saving and loading genomes are JSON export and checkpointing:

```julia
# JSON export (portable, neat-python compatible)
export_network_json(winner, config.genome_config, "winner.json")
imported = import_network_json("winner.json", config.genome_config)

# Checkpointing (saves full population state)
add_reporter!(pop, Checkpointer(generation_interval=10))
# Later: restore and continue evolution
pop = restore_checkpoint("neat-checkpoint-50")
```

You can also use Julia's built-in serialization:

```julia
using Serialization
serialize("winner.jls", winner)
winner = deserialize("winner.jls")
```

## Troubleshooting

### "Fitness not assigned" Error

Make sure your fitness function sets `genome.fitness`:
```julia
genome.fitness = computed_fitness  # Don't forget this!
```

### Networks Always Output 0.5

Check:
- Initial connection strategy
- Weight initialization parameters
- Activation functions (sigmoid should be default)

### Evolution Too Slow

Try:
- Increase `pop_size`
- Set `initial_connection = "full"`
- Increase `survival_threshold`

### Networks Too Complex

Set:
- `single_structural_mutation = true`
- Lower `conn_add_prob` and `node_add_prob`

## Next Steps

- Try modifying the fitness function
- Experiment with different configuration parameters
- Adapt the example for your own problem
- Explore other examples in `examples/` directory

## Related Documentation

- [Getting Started Guide](getting_started.md)
- [Configuration Reference](config_file.md)
- [Activation Functions](activation_functions.md)
- [Visualization Guide](visualization_guide.md)

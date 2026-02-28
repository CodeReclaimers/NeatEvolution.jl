# Getting Started with NeatEvolution.jl

## Overview

NeatEvolution.jl is a Julia implementation of **NEAT** (NeuroEvolution of Augmenting Topologies), an evolutionary algorithm that creates artificial neural networks. This guide will help you get started with using NeatEvolution.jl in your projects.

## What is NEAT?

NEAT is a method developed by Kenneth O. Stanley for evolving arbitrary neural networks through genetic algorithms. Unlike traditional neural network training methods that use gradient descent, NEAT evolves both the network topology and weights simultaneously through mutation and crossover operations.

### Key Features

- **Topology Evolution**: Networks start simple and grow more complex over generations
- **Historical Markings**: Innovation numbers track gene origins for proper crossover
- **Speciation**: Protects innovative structures from premature elimination
- **Minimal Assumptions**: Works without domain-specific network architecture design

## Installation

### From Julia REPL

```julia
using Pkg
Pkg.add(url="https://github.com/CodeReclaimers/NEAT.git")
```

### For Development

```bash
git clone https://github.com/CodeReclaimers/NEAT.git
cd NEAT
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## Quick Start: Solving XOR

Here's a minimal example that evolves a neural network to solve the XOR problem:

### 1. Define Your Fitness Function

```julia
using NeatEvolution

# XOR test cases
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

function eval_genomes(genomes, config)
    for (genome_id, genome) in genomes
        # Start with maximum fitness
        genome.fitness = 4.0

        # Create a neural network from the genome
        net = FeedForwardNetwork(genome, config.genome_config)

        # Test on all XOR cases
        for (xi, xo) in zip(XOR_INPUTS, XOR_OUTPUTS)
            output = activate!(net, xi)
            # Penalize squared error
            genome.fitness -= (output[1] - xo[1])^2
        end
    end
end
```

### 2. Create a Configuration File

Create `config.toml`:

```toml
[NEAT]
pop_size = 150
fitness_criterion = "max"
fitness_threshold = 3.9
reset_on_extinction = false

[DefaultGenome]
# Network structure
num_inputs = 2
num_outputs = 1
num_hidden = 0
feed_forward = true
initial_connection = "full"

# Compatibility coefficients
compatibility_excess_coefficient = 1.0
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient = 0.4

# Structural mutation probabilities
conn_add_prob = 0.5
conn_delete_prob = 0.5
node_add_prob = 0.2
node_delete_prob = 0.2

# Node activation
activation_default = "sigmoid"
activation_mutate_rate = 0.0
activation_options = ["sigmoid"]

# Node aggregation
aggregation_default = "sum"
aggregation_mutate_rate = 0.0
aggregation_options = ["sum"]

# Connection weight options
weight_init_mean = 0.0
weight_init_stdev = 1.0
weight_max_value = 30.0
weight_min_value = -30.0
weight_mutate_power = 0.5
weight_mutate_rate = 0.8
weight_replace_rate = 0.1

# Connection enabled options
enabled_default = true
enabled_mutate_rate = 0.01

# Node bias options
bias_init_mean = 0.0
bias_init_stdev = 1.0
bias_max_value = 30.0
bias_min_value = -30.0
bias_mutate_power = 0.5
bias_mutate_rate = 0.7
bias_replace_rate = 0.1

# Node response options
response_init_mean = 1.0
response_init_stdev = 0.0
response_max_value = 30.0
response_min_value = -30.0
response_mutate_power = 0.0
response_mutate_rate = 0.0
response_replace_rate = 0.0

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = "max"
max_stagnation = 20
species_elitism = 2

[DefaultReproduction]
elitism = 2
survival_threshold = 0.2
min_species_size = 1
```

### 3. Run Evolution

```julia
# Load configuration
config = load_config("config.toml")

# Create population
pop = Population(config)

# Add a reporter to see progress
add_reporter!(pop, StdOutReporter(true))

# Run evolution for up to 100 generations
winner = run!(pop, eval_genomes, 100)

# Display results
println("\nBest genome:")
println("  Fitness: ", winner.fitness)
println("  Nodes: ", length(winner.nodes))
println("  Connections: ", length(winner.connections))
```

## Understanding the Core Concepts

### Genomes

A **genome** in NEAT represents an artificial neural network through two types of genes:

1. **Node Genes**: Define neurons with properties like:
   - Bias
   - Response multiplier
   - Activation function
   - Aggregation function

2. **Connection Genes**: Define weighted links between neurons with:
   - Input and output node IDs
   - Weight
   - Enabled/disabled status
   - Innovation number (for tracking historical origins)

### Evolution Process

1. **Initialization**: Create a population of random genomes
2. **Evaluation**: Your fitness function scores each genome
3. **Selection**: Better genomes are more likely to reproduce
4. **Reproduction**: Genomes undergo crossover and mutation
5. **Speciation**: Similar genomes are grouped into species
6. **Iteration**: Repeat steps 2-5 until solution found or generation limit reached

### Fitness Function

The fitness function is the heart of NEAT evolution. It must:

- Accept `(genomes, config)` as parameters
- Set `genome.fitness` for each genome (higher is better)
- Evaluate how well the genome solves your problem

Example patterns:

```julia
# Minimize error
function eval_genomes(genomes, config)
    for (id, genome) in genomes
        net = FeedForwardNetwork(genome, config.genome_config)
        error = compute_error(net)
        genome.fitness = 1.0 / (1.0 + error)  # Higher fitness = lower error
    end
end

# Maximize score
function eval_genomes(genomes, config)
    for (id, genome) in genomes
        net = FeedForwardNetwork(genome, config.genome_config)
        score = play_game(net)
        genome.fitness = score  # Higher fitness = higher score
    end
end
```

## Visualization

NeatEvolution.jl provides optional visualization capabilities through Plots.jl:

```julia
using NeatEvolution
using Plots  # Load Plots to enable visualization

# ... run evolution with StatisticsReporter ...

stats = StatisticsReporter()
add_reporter!(pop, stats)
winner = run!(pop, eval_genomes, 100)

# Plot fitness over generations
plot_fitness(stats, filename="fitness.png")

# Plot species evolution
plot_species(stats, filename="species.png")

# Draw the winner network
draw_net(winner, config.genome_config,
         filename="winner.png",
         node_names=Dict(-1=>"x1", -2=>"x2", 0=>"out"))
```

See [Visualization Guide](visualization_guide.md) for advanced visualization features.

## Next Steps

- **[Configuration Reference](config_file.md)**: Detailed parameter descriptions
- **[XOR Example](xor_example.md)**: Complete walkthrough of the XOR problem
- **[Activation Functions](activation_functions.md)**: Available activation functions
- **[API Reference](api_reference.md)**: Complete API documentation
- **Examples**: Check the `examples/` directory for more use cases

## Common Patterns

### Using Multiple Outputs

```julia
[DefaultGenome]
num_inputs = 2
num_outputs = 3  # Multiple output neurons

# In fitness function:
net = FeedForwardNetwork(genome, config.genome_config)
outputs = activate!(net, inputs)  # Returns vector of 3 values
```

### Recurrent Networks

```julia
[DefaultGenome]
feed_forward = false  # Allow recurrent connections

# Networks can now have cycles - useful for temporal problems
```

### Custom Activation Functions

```julia
using NeatEvolution

# Define custom activation
my_activation(x) = x^3

# Register it
add_activation_function!(:cubic, my_activation)

# Use in config:
# activation_options = ["sigmoid", "cubic"]
```

### Population Seeding for Transfer Learning

You can seed populations with pre-evolved networks imported from JSON files. This is useful for transfer learning, cross-library experiments, or checkpointing:

```julia
using NeatEvolution

config = load_config("config.toml")

# Import genomes from JSON (neat-python or NeatEvolution.jl format)
imported_genomes = [
    import_network_json("winner1.json", config.genome_config),
    import_network_json("winner2.json", config.genome_config)
]

# Create population with imported genomes
# The system automatically adjusts genome IDs, node IDs, and innovation numbers
pop = Population(config, imported_genomes)

# Continue evolution from these seeds
add_reporter!(pop, StdOutReporter(true))
winner = run!(pop, eval_genomes, 100)
```

**Parameters:**
- `fill_remaining=true` (default): Fills to configured `pop_size` with random genomes
- `fill_remaining=false`: Population contains only the imported genomes

**Use cases:**
- **Transfer learning**: Bootstrap with networks from a related problem
- **Cross-library**: Import neat-python networks, continue in Julia
- **Checkpointing**: Save/restore evolution state across sessions
- **Ensemble seeding**: Start with multiple good solutions

The imported genomes preserve their original IDs and all internal counters are automatically adjusted to ensure no conflicts with newly created genomes during evolution.

## Troubleshooting

### Evolution Stagnates

- Increase `pop_size` (try 200-300)
- Increase `compatibility_threshold` to allow more diversity
- Adjust mutation rates (increase `conn_add_prob`, `node_add_prob`)
- Reduce `max_stagnation` to eliminate stagnant species faster

### Networks Too Complex

- Decrease `conn_add_prob` and `node_add_prob`
- Set `single_structural_mutation = true`
- Start with `initial_connection = "unconnected"`

### Convergence Too Slow

- Increase `survival_threshold` (more aggressive selection)
- Increase `elitism` to preserve best individuals
- Adjust fitness function to provide better gradient

## Resources

- **Original Paper**: [Evolving Neural Networks through Augmenting Topologies](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
- **NEAT-Python**: [Documentation](https://neat-python.readthedocs.io/)
- **Examples**: See `examples/` directory in this repository

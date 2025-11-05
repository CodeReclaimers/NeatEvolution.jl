# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NEAT is a Julia package implementing NeuroEvolution of Augmenting Topologies (NEAT). This library is designed as:
- A platform for running ad-hoc evolution experiments to determine feasibility
- A starting point for building custom neuroevolution solutions

**Not intended as:** A quick-start platform for game development.

The package is based on the neat-python implementation and provides equivalent functionality in pure Julia.

## Common Commands

### Testing
```bash
# Run all tests
julia --project -e 'using Pkg; Pkg.test()'

# Run tests interactively (for development)
julia --project -e 'using Pkg; Pkg.test("NEAT"; test_args=["--verbose"])'
```

### Running Examples
```bash
# Run the XOR example
julia --project examples/xor/evolve.jl
```

### Development Setup
```bash
# Activate the package environment
julia --project

# Install dependencies
julia --project -e 'using Pkg; Pkg.instantiate()'

# Load package in REPL for interactive development
julia --project -e 'using NEAT'
```

### Documentation
```bash
# Build documentation locally
julia --project=docs docs/make.jl
```

### CI/CD
The project uses GitHub Actions for CI, testing on:
- Julia versions: 1.10, 1.11, nightly
- Operating systems: Ubuntu, Windows, macOS

## Architecture

### Core Components

The NEAT implementation is organized into the following modules:

#### 1. **attributes.jl**
Handles gene attributes (weights, biases, etc.) with:
- `FloatAttribute`: Floating-point parameters (weights, biases)
- `BoolAttribute`: Boolean parameters (enabled connections)
- `StringAttribute`: Categorical parameters (activation functions)

Each attribute type handles initialization, mutation, and validation.

#### 2. **genes.jl**
Defines gene representations:
- `NodeGene`: Represents neurons with bias, response, activation, and aggregation
- `ConnectionGene`: Represents connections with weight and enabled status

Supports mutation, crossover, and distance calculations for speciation.

#### 3. **genome.jl**
Core genome representation and operations:
- `Genome`: Container for nodes and connections
- Initialization strategies (full, partial connectivity)
- Structural mutations (add/delete nodes and connections)
- Crossover between parent genomes
- Genetic distance computation for speciation

#### 4. **config.jl**
Configuration system using TOML format:
- `Config`: Main configuration container
- `GenomeConfig`: Genome structure and mutation parameters
- `SpeciesConfig`: Speciation parameters
- `StagnationConfig`: Stagnation detection parameters
- `ReproductionConfig`: Reproduction and elitism parameters

#### 5. **activations.jl**
Activation functions for nodes:
- sigmoid, tanh, relu, elu, lelu, selu
- sin, gauss, softplus, identity, clamped
- inv, log, exp, abs, hat, square, cube

Extensible: add custom functions with `add_activation_function!`

#### 6. **aggregations.jl**
Aggregation functions for combining inputs:
- sum, product, max, min, maxabs, median, mean

Extensible: add custom functions with `add_aggregation_function!`

#### 7. **graphs.jl**
Graph algorithms for network structure:
- `creates_cycle`: Detects cycles for feed-forward networks
- `required_for_output`: Finds nodes needed for output computation
- `feed_forward_layers`: Computes evaluation order

#### 8. **species.jl**
Speciation system:
- `Species`: Groups genomes by genetic similarity
- `SpeciesSet`: Manages all species
- `GenomeDistanceCache`: Caches distance computations for efficiency

#### 9. **stagnation.jl**
Tracks species improvement:
- Marks species as stagnant if no fitness improvement
- Protects top species (elitism)
- Uses configurable fitness functions (max, mean, etc.)

#### 10. **reproduction.jl**
Genome reproduction and population management:
- Creates offspring through crossover and mutation
- Implements elitism (preserves best genomes)
- Computes spawn amounts based on adjusted fitness
- Manages survival threshold for breeding selection

#### 11. **population.jl**
Main evolution loop:
- Coordinates evaluation, reproduction, and speciation
- Handles extinction and reset
- Checks termination criteria
- Integrates with reporters for progress tracking

#### 12. **feedforward.jl**
Neural network evaluation:
- `FeedForwardNetwork`: Evaluates genomes as neural networks
- Computes feed-forward layers for efficient evaluation
- `activate!`: Runs network with given inputs

#### 13. **reporting.jl**
Progress reporting:
- `StdOutReporter`: Prints generation statistics
- Tracks fitness, species, and population size
- Reports solutions and extinction events

#### 14. **utils.jl**
Utility functions:
- Statistical functions (mean, median, stdev, variance)
- Stat function registry for configuration

### File Structure
```
src/
├── NEAT.jl              # Main module with exports
├── attributes.jl        # Attribute system
├── genes.jl            # Node and Connection genes
├── genome.jl           # Genome with mutation/crossover
├── species.jl          # Speciation logic
├── reproduction.jl     # Reproduction and elitism
├── stagnation.jl       # Species stagnation tracking
├── population.jl       # Main evolution loop
├── config.jl           # Configuration system (TOML)
├── activations.jl      # Activation functions
├── aggregations.jl     # Aggregation functions
├── graphs.jl           # Graph algorithms
├── feedforward.jl      # Feed-forward network evaluation
├── reporting.jl        # Progress reporting
└── utils.jl            # Math utilities

examples/
└── xor/
    ├── config.toml     # XOR configuration
    └── evolve.jl       # XOR evolution script

test/
└── runtests.jl         # Test suite with XOR test
```

### Package Metadata
- UUID: `fcc92617-70eb-4c43-847c-323c44b5224c`
- Dependencies: Random, Statistics, TOML (all stdlib)

## Configuration Files

Configuration files use TOML format with the following sections:

### [NEAT]
- `pop_size`: Population size
- `fitness_criterion`: "max", "min", or "mean"
- `fitness_threshold`: Target fitness for solution
- `reset_on_extinction`: Whether to reset population on extinction
- `no_fitness_termination`: Run for fixed generations

### [DefaultGenome]
Network structure and mutation parameters:
- `num_inputs`, `num_outputs`, `num_hidden`
- `feed_forward`: true for feed-forward networks
- `initial_connection`: "full", "partial", etc.
- Mutation probabilities: `conn_add_prob`, `node_add_prob`, etc.
- Attribute parameters for bias, response, weight, activation, etc.

### [DefaultSpeciesSet]
- `compatibility_threshold`: Distance threshold for speciation

### [DefaultStagnation]
- `species_fitness_func`: "max", "mean", etc.
- `max_stagnation`: Generations before marking stagnant
- `species_elitism`: Number of top species to protect

### [DefaultReproduction]
- `elitism`: Number of best genomes to preserve
- `survival_threshold`: Fraction of species for breeding
- `min_species_size`: Minimum species size

## Development Notes

### Adding New Activation Functions
```julia
using NEAT
NEAT.add_activation_function!(:custom, x -> tanh(x^2))
```

### Adding New Aggregation Functions
```julia
using NEAT
NEAT.add_aggregation_function!(:custom, xs -> sum(xs.^2))
```

### Creating Custom Fitness Functions
Fitness functions receive a list of (genome_id, genome) pairs and config:
```julia
function eval_genomes(genomes, config)
    for (gid, genome) in genomes
        # Evaluate genome
        net = FeedForwardNetwork(genome, config.genome_config)
        # ... compute fitness ...
        genome.fitness = fitness_value
    end
end
```

### Architecture Philosophy
- Uses Julia structs instead of Python classes
- Leverages multiple dispatch for polymorphic behavior
- TOML configuration (Julia-native format)
- Follows neat-python's core algorithms for compatibility
- Pure Julia implementation with no external dependencies

When implementing new functionality, follow Julia package conventions and ensure compatibility with Julia 1.10+ (LTS) through nightly builds.

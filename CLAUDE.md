# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeatEvolution is a Julia package implementing NeuroEvolution of Augmenting Topologies (NEAT). This library is designed as:
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
julia --project -e 'using Pkg; Pkg.test("NeatEvolution"; test_args=["--verbose"])'
```

### Running Examples
```bash
# Run the XOR example
julia --project examples/xor/evolve.jl

# Other examples: cartpole, checkpoint_demo, ctrnn_oscillator,
# inverted_pendulum, inverted_double_pendulum, iznn_pattern,
# lorenz_ctrnn, sequence
julia --project examples/cartpole/evolve.jl
```

### Development Setup
```bash
# Activate the package environment
julia --project

# Install dependencies
julia --project -e 'using Pkg; Pkg.instantiate()'

# Load package in REPL for interactive development
julia --project -e 'using NeatEvolution'
```

### Documentation
```bash
# Build documentation locally
julia --project=docs docs/make.jl
```

### CI/CD
The project uses GitHub Actions for CI, testing on:
- Julia versions: 1.10, 1.11, 1.12, nightly
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
- Supports initialization with imported genomes (auto-adjusts ID counters)

#### 12. **feedforward.jl**
Neural network evaluation:
- `FeedForwardNetwork`: Evaluates genomes as neural networks
- Computes feed-forward layers for efficient evaluation
- `activate!`: Runs network with given inputs

#### 13. **recurrent.jl**
Recurrent neural network evaluation:
- `RecurrentNetwork`: Handles cyclic connections with internal state
- Double-buffered values (current/previous timestep)
- `activate!`: Runs one timestep; `reset!`: Clears state

#### 14. **ctrnn.jl**
Continuous-Time Recurrent Neural Network:
- `CTRNNNetwork`: Forward Euler integration with per-node time constants
- `advance!`: Integrates forward by specified time with configurable step size
- `set_node_value!`: Injects state into network buffers

#### 15. **iznn.jl**
Izhikevich spiking neural network:
- `IZNNNetwork`: Biologically realistic spiking dynamics (a, b, c, d parameters)
- Spike-based communication (binary 0/1)
- Named presets for common neuron types (regular spiking, fast spiking, etc.)

#### 16. **reporting.jl**
Progress reporting:
- `StdOutReporter`: Prints generation statistics
- `StatisticsReporter`: Collects per-generation fitness and species data for analysis
- Tracks fitness, species, and population size
- Reports solutions and extinction events

#### 17. **statistics.jl**
Statistics collection and export:
- `StatisticsReporter`: Records fitness statistics and species data each generation
- `save_statistics`: Exports collected data to CSV files
- `best_genome`, `best_genomes`, `best_unique_genomes`: Query best performers

#### 18. **checkpointer.jl**
Evolution checkpointing:
- `Checkpointer`: Reporter that saves population state at configurable intervals
- `save_checkpoint` / `restore_checkpoint`: Manual save/restore via Serialization
- Supports generation-based and time-based intervals

#### 19. **export.jl**
JSON export/import for model sharing:
- `export_network_json` / `import_network_json`: Single genome serialization
- `export_population_json`: Batch export of top genomes
- Compatible with neat-python JSON format

#### 20. **validation.jl**
Configuration validation:
- Validates TOML config files against known parameter names
- Detects typos and unknown parameters
- Warns about missing recommended parameters

#### 21. **utils.jl**
Utility functions:
- Statistical functions (mean, median, stdev, variance)
- `tmean`: Trimmed mean; `softmax`: Softmax normalization
- Stat function registry for configuration

### File Structure
```
src/
├── NeatEvolution.jl     # Main module with exports
├── attributes.jl        # Attribute system
├── genes.jl             # Node and Connection genes
├── genome.jl            # Genome with mutation/crossover
├── config.jl            # Configuration system (TOML)
├── validation.jl        # Config validation and typo detection
├── species.jl           # Speciation logic
├── reproduction.jl      # Reproduction and elitism
├── stagnation.jl        # Species stagnation tracking
├── population.jl        # Main evolution loop
├── activations.jl       # Activation functions
├── aggregations.jl      # Aggregation functions
├── graphs.jl            # Graph algorithms
├── feedforward.jl       # Feed-forward network evaluation
├── recurrent.jl         # Recurrent network evaluation
├── ctrnn.jl             # Continuous-time recurrent network
├── iznn.jl              # Izhikevich spiking network
├── reporting.jl         # Progress reporting
├── statistics.jl        # Statistics collection and CSV export
├── checkpointer.jl      # Evolution checkpointing
├── export.jl            # JSON export/import
└── utils.jl             # Math utilities

ext/
├── NeatEvolutionVisualizationExt.jl   # Plots.jl visualization extension
└── NeatEvolutionGraphMakieExt.jl      # GraphMakie interactive visualization

examples/
├── xor/                 # Classic XOR benchmark
├── cartpole/            # Cart-pole balancing
├── sequence/            # Sequence prediction (recurrent)
├── ctrnn_oscillator/    # CTRNN oscillator
├── iznn_pattern/        # Izhikevich spiking patterns
├── checkpoint_demo/     # Checkpointing demonstration
├── inverted_pendulum/   # Inverted pendulum control
├── inverted_double_pendulum/  # Double pendulum control
└── lorenz_ctrnn/        # Lorenz attractor with CTRNN

test/
├── runtests.jl              # Test runner
├── test_attributes.jl       # Attribute system tests
├── test_genes.jl            # Gene tests
├── test_genome_operations.jl # Genome operation tests
├── test_config.toml         # Test configuration
├── test_speciation.jl       # Speciation tests
├── test_stagnation.jl       # Stagnation tests
├── test_reproduction.jl     # Reproduction tests
├── test_recurrent.jl        # Recurrent network tests
├── test_ctrnn.jl            # CTRNN tests
├── test_iznn.jl             # Izhikevich network tests
├── test_ctrnn_iznn_genes.jl # CTRNN/IZNN gene tests
├── test_checkpointer.jl     # Checkpointing tests
├── test_validation.jl       # Validation tests
├── test_population_seeding.jl # Population seeding tests
└── test_misc_coverage.jl    # Additional coverage tests
```

### Package Metadata
- UUID: `fcc92617-70eb-4c43-847c-323c44b5224c`
- Stdlib dependencies: Dates, Random, Serialization, Statistics, TOML
- External dependencies: FunctionWrappers, JSON

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
using NeatEvolution
NeatEvolution.add_activation_function!(:custom, x -> tanh(x^2))
```

### Adding New Aggregation Functions
```julia
using NeatEvolution
NeatEvolution.add_aggregation_function!(:custom, xs -> sum(xs.^2))
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

### Importing and Seeding Populations
Import evolved networks from JSON (neat-python or NeatEvolution.jl format) and use them to seed new populations:

```julia
using NeatEvolution

config = load_config("config.toml")

# Import genomes from JSON files
imported_genomes = [
    import_network_json("winner1.json", config.genome_config),
    import_network_json("winner2.json", config.genome_config)
]

# Create population seeded with imported genomes
# The constructor automatically adjusts genome IDs, node IDs, and innovation numbers
# to prevent conflicts with newly created genomes
pop = Population(config, imported_genomes)

# The population now contains:
# - All imported genomes (with their original IDs preserved)
# - Newly created random genomes to fill remaining slots
# - Properly adjusted counters ensuring no ID conflicts

# Continue evolution
winner = run!(pop, eval_genomes, n_generations)
```

Options:
- `fill_remaining=true` (default): Fill population to configured size with random genomes
- `fill_remaining=false`: Population contains only the imported genomes

This is useful for:
- **Transfer learning**: Start evolution with pre-trained networks from a related task
- **Cross-library experiments**: Import neat-python networks and continue evolution in Julia
- **Population seeding**: Bootstrap evolution with known good solutions
- **Checkpoint/resume**: Save and restore evolution state across sessions

### Architecture Philosophy
- Uses Julia structs instead of Python classes
- Leverages multiple dispatch for polymorphic behavior
- TOML configuration (Julia-native format)
- Follows neat-python's core algorithms for compatibility
- Pure Julia implementation (external deps: JSON, FunctionWrappers only)
- Optional visualization via package extensions (Plots.jl, GraphMakie)

When implementing new functionality, follow Julia package conventions and ensure compatibility with Julia 1.10+ (LTS) through nightly builds.

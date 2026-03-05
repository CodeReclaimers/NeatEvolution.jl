# NeatEvolution.jl

[![CI](https://github.com/CodeReclaimers/NeatEvolution.jl/actions/workflows/CI.yaml/badge.svg)](https://github.com/CodeReclaimers/NeatEvolution.jl/actions/workflows/CI.yaml)
[![codecov](https://codecov.io/gh/CodeReclaimers/NeatEvolution.jl/graph/badge.svg?token=BMK6EVEC48)](https://codecov.io/gh/CodeReclaimers/NeatEvolution.jl)

A pure Julia implementation of **NEAT** (NeuroEvolution of Augmenting Topologies), the evolutionary algorithm that creates artificial neural networks.

## Overview

NEAT is a method developed by Kenneth O. Stanley for evolving arbitrary neural networks. This implementation is compliant with the original [NEAT paper](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) and provides a robust platform for neuroevolution experiments.

**What this library is:**
- A faithful implementation of the NEAT algorithm (v0.1.0)
- A flexible platform for neuroevolution experiments
- A starting point for custom neuroevolution solutions

**What this library is not:**
- A game development framework
- A replacement for gradient-based deep learning

## Features

✅ **Paper-Compliant NEAT Algorithm**
- Innovation number tracking for proper gene alignment
- Correct compatibility distance formula (Equation 1 from paper)
- 75% crossover disable rule per original specification
- Speciation with genomic distance

✅ **Complete Functionality**
- Four network types: feed-forward, recurrent, CTRNN (continuous-time), and Izhikevich spiking
- 18 built-in activation functions
- 7 aggregation functions
- Multiple initial connection strategies
- Comprehensive mutation operators
- JSON export/import for model sharing (neat-python compatible)
- Population seeding with imported genomes for transfer learning
- Checkpointing for saving and restoring evolution state
- StatisticsReporter for fitness/species tracking and CSV export

✅ **Visualization** (Optional)
- Fitness evolution plots
- Species dynamics visualization
- Network topology diagrams
- Activation heatmaps for 2D problems
- Evolution animations (GIF)

✅ **Quality Assurance**
- Comprehensive test suite
- Code coverage reporting
- Continuous integration
- Comprehensive troubleshooting guide

## Quick Start

### Installation

```julia
using Pkg
Pkg.add(url="https://github.com/CodeReclaimers/NeatEvolution.jl.git")
```

### Basic Example

```julia
using NeatEvolution

# Define your fitness function
function eval_genomes(genomes, config)
    for (genome_id, genome) in genomes
        net = FeedForwardNetwork(genome, config.genome_config)
        # Evaluate your problem
        score = your_evaluation_function(net)
        genome.fitness = score
    end
end

# Load configuration
config = load_config("config.toml")

# Create and run population
pop = Population(config)
add_reporter!(pop, StdOutReporter(true))
winner = run!(pop, eval_genomes, 100)

# Use the winner
net = FeedForwardNetwork(winner, config.genome_config)
output = activate!(net, [1.0, 0.0])
```

See [examples/xor/](examples/xor/) for a complete working example.

### Advanced: Population Seeding for Transfer Learning

You can seed populations with pre-trained networks from JSON (from neat-python or NeatEvolution.jl):

```julia
using NeatEvolution

config = load_config("config.toml")

# Import evolved networks
imported_genomes = [
    import_network_json("winner1.json", config.genome_config),
    import_network_json("winner2.json", config.genome_config)
]

# Create population seeded with imported genomes
# Counters are automatically adjusted to prevent ID conflicts
pop = Population(config, imported_genomes)
winner = run!(pop, eval_genomes, 100)
```

This is useful for:
- **Transfer learning** - Start with networks from a related task
- **Cross-library experiments** - Import neat-python networks, continue in Julia
- **Checkpointing** - Save and restore evolution state

## Documentation

- **[Getting Started Guide](docs/getting_started.md)** - Learn the basics
- **[Configuration Reference](docs/config_file.md)** - All configuration parameters
- **[XOR Example Walkthrough](docs/xor_example.md)** - Complete tutorial
- **[API Reference](docs/api_reference.md)** - Complete API documentation
- **[Activation Functions](docs/activation_functions.md)** - Available activation functions
- **[Aggregation Functions](docs/aggregation_functions.md)** - Aggregation function reference
- **[Algorithm Internals](docs/algorithm_internals.md)** - Deep dive into NEAT mechanics
- **[Visualization Guide](docs/visualization_guide.md)** - Complete visualization tutorial
- **[Troubleshooting Guide](docs/troubleshooting.md)** - Common problems and solutions
- **[FAQ](docs/faq.md)** - Frequently asked questions

## Example: Solving XOR

```julia
using NeatEvolution

# XOR test cases
const XOR_INPUTS = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
const XOR_OUTPUTS = [[0.0], [1.0], [1.0], [0.0]]

function eval_genomes(genomes, config)
    for (genome_id, genome) in genomes
        genome.fitness = 4.0
        net = FeedForwardNetwork(genome, config.genome_config)
        for (xi, xo) in zip(XOR_INPUTS, XOR_OUTPUTS)
            output = activate!(net, xi)
            genome.fitness -= (output[1] - xo[1])^2
        end
    end
end

config = load_config("examples/xor/config.toml")
pop = Population(config)
add_reporter!(pop, StdOutReporter(true))
winner = run!(pop, eval_genomes, 100)

println("Solution found! Fitness: ", winner.fitness)
```

## Visualization

### Static Visualization (Plots.jl)

Optional visualization support through Plots.jl:

```julia
using NeatEvolution
using Plots  # Enables static visualization

stats = StatisticsReporter()
add_reporter!(pop, stats)
winner = run!(pop, eval_genomes, 100)

# Generate static visualizations
plot_fitness(stats, filename="fitness.png")
plot_species(stats, filename="species.png")
draw_net(winner, config.genome_config, filename="network.png")
plot_activation_heatmap(winner, config.genome_config, filename="heatmap.png")
animate_evolution(stats, config.genome_config, filename="evolution.gif")
```

### Interactive Visualization (GraphMakie)

For interactive 3D network visualization with rotation, zoom, and pan:

```julia
using NeatEvolution
using GLMakie, GraphMakie, Graphs  # Enables interactive visualization

# Create interactive network visualization
fig = draw_network_interactive(winner, config.genome_config,
    layout=:spring,
    title="Interactive Network"
)
display(fig)  # Opens interactive window

# Compare multiple networks interactively
top3 = best_genomes(stats, 3)
fig = draw_network_comparison_interactive(top3, config.genome_config,
    labels=["Best", "2nd", "3rd"]
)
display(fig)
```

See the [Visualization Guide](docs/visualization_guide.md) for complete details.

## Version History

**v0.1.0** (Current) - NEAT Paper Compliance + Enhancements
- ✅ Innovation numbers implemented
- ✅ Crossover disable rule fixed (75%)
- ✅ Compatibility distance formula corrected
- ✅ All 3 phases of visualization complete
- ✅ JSON export/import for model sharing
- ✅ Stricter configuration validation with typo detection
- ✅ Comprehensive troubleshooting guide and FAQ

**v0.0.x** - Initial Implementation
- Basic NEAT algorithm
- Statistics and visualization (Phases 1-3)

## Running Tests

```julia
using Pkg
Pkg.test("NeatEvolution")
```

All tests should pass!

## Examples

The `examples/` directory contains complete working examples:

| Directory | Description |
|-----------|-------------|
| `xor/` | Classic XOR benchmark (feed-forward) |
| `cartpole/` | Cart-pole balancing control task |
| `sequence/` | Sequence prediction with recurrent networks |
| `ctrnn_oscillator/` | CTRNN oscillator demonstration |
| `iznn_pattern/` | Izhikevich spiking network patterns |
| `checkpoint_demo/` | Checkpointing save/restore demonstration |
| `inverted_pendulum/` | Inverted pendulum control |
| `inverted_double_pendulum/` | Double inverted pendulum control |
| `lorenz_ctrnn/` | Lorenz attractor with CTRNN |

Run any example with:
```bash
julia --project examples/xor/evolve.jl
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Citing

If you use NeatEvolution.jl in a publication, you can cite it using the following BibTeX entry:

```bibtex
@software{McIntyre_NeatEvolution_jl,
  author = {McIntyre, Alan},
  title = {{NeatEvolution.jl}},
  url = {https://github.com/CodeReclaimers/NeatEvolution.jl}
}
```

## License

This project is inspired by and follows the design of [neat-python](https://github.com/CodeReclaimers/neat-python).

## Acknowledgments

- Kenneth O. Stanley and Risto Miikkulainen for the original NEAT algorithm
- The neat-python project for design inspiration
- Julia community for the excellent language and ecosystem

## See Also

- **Original NEAT Paper**: [Evolving Neural Networks through Augmenting Topologies](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
- **neat-python**: [GitHub](https://github.com/CodeReclaimers/neat-python) | [Documentation](https://neat-python.readthedocs.io/)
- **Kenneth Stanley's Website**: [Homepage](https://www.kenstanley.net)

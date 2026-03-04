# NeatEvolution.jl

A pure Julia implementation of **NEAT** (NeuroEvolution of Augmenting Topologies), the evolutionary algorithm that creates artificial neural networks.

## Overview

NEAT is a method developed by Kenneth O. Stanley for evolving arbitrary neural networks. This implementation is compliant with the original [NEAT paper](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) and is based on the [neat-python](https://github.com/CodeReclaimers/neat-python) implementation.

**What this library is:**
- A faithful implementation of the NEAT algorithm
- A flexible platform for neuroevolution experiments
- A starting point for custom neuroevolution solutions

**What this library is not:**
- A game development framework
- A replacement for gradient-based deep learning

## Features

- **Paper-compliant NEAT algorithm** with innovation number tracking, compatibility distance (Equation 1), and 75% crossover disable rule
- **Multiple network types:** Feed-forward, Recurrent, CTRNN (continuous-time), and Izhikevich spiking networks
- **18 activation functions** and **7 aggregation functions**, both extensible
- **JSON export/import** for model sharing (neat-python compatible)
- **Population seeding** with imported genomes for transfer learning
- **Checkpointing** for saving and restoring evolution state
- **Optional visualization** via Plots.jl and GraphMakie extensions
- **Pure Julia** with no external dependencies beyond stdlib + JSON + FunctionWrappers

## Quick Start

```julia
using Pkg
Pkg.add(url="https://github.com/CodeReclaimers/NeatEvolution.jl.git")
```

```julia
using NeatEvolution

config = load_config("config.toml")
pop = Population(config)

function eval_genomes(genomes, config)
    for (gid, genome) in genomes
        net = FeedForwardNetwork(genome, config.genome_config)
        # ... evaluate network and set genome.fitness ...
    end
end

winner = run!(pop, eval_genomes, 300)
```

See the [Getting Started](getting_started.md) guide and [XOR Example](xor_example.md) for detailed walkthroughs.

## Documentation

- **[Getting Started](getting_started.md)** — Installation, first experiment, and basic concepts
- **[XOR Example](xor_example.md)** — Complete walkthrough of the classic XOR benchmark
- **[Configuration](config_file.md)** — TOML configuration file reference
- **[API Reference](api_reference.md)** — Complete function and type documentation
- **[Activation Functions](activation_functions.md)** — Built-in and custom activation functions
- **[Aggregation Functions](aggregation_functions.md)** — Built-in and custom aggregation functions
- **[Visualization](visualization_guide.md)** — Plotting fitness, species, and network topology
- **[Algorithm Internals](algorithm_internals.md)** — How NEAT works under the hood
- **[FAQ](faq.md)** — Frequently asked questions
- **[Troubleshooting](troubleshooting.md)** — Common issues and solutions

# AGENTS.md — NeatEvolution.jl

## Project Overview

NeatEvolution.jl is a Julia implementation of NEAT (NeuroEvolution of Augmenting Topologies),
a method developed by Kenneth O. Stanley for evolving arbitrary neural networks. It is a port
of neat-python by the same author, designed to take advantage of Julia's performance
characteristics while maintaining the same algorithmic fidelity to the original NEAT paper.

- **License:** 3-clause BSD
- **Julia compatibility:** 1.11+
- **Zenodo DOI:** https://doi.org/10.5281/zenodo.19025463
- **Related project:** https://github.com/CodeReclaimers/neat-python

## Repository Structure

```
src/
  NeatEvolution.jl     # Module definition and public API
  activations.jl       # Activation functions (sigmoid, tanh, relu, etc.)
  aggregations.jl      # Aggregation functions (sum, product, max, etc.)
  attributes.jl        # Gene attribute system (Float, Bool, String)
  checkpointer.jl      # Evolution checkpointing (save/restore)
  config.jl            # Configuration parsing (TOML format)
  export.jl            # JSON export/import for model sharing
  feedforward.jl       # FeedForwardNetwork
  genes.jl             # Node and Connection gene representations
  genome.jl            # Genome representation (node genes, connection genes)
  graphs.jl            # Graph algorithms (cycle detection, layer computation)
  iznn.jl              # IZNNNetwork (Izhikevich spiking neural network)
  population.jl        # Population management and evolution loop
  recurrent.jl         # RecurrentNetwork
  reporting.jl         # Progress reporting (StdOutReporter)
  reproduction.jl      # Crossover and mutation operators
  species.jl           # Speciation by genomic distance
  stagnation.jl        # Species stagnation tracking
  statistics.jl        # StatisticsReporter and CSV export
  ctrnn.jl             # CTRNNNetwork (continuous-time recurrent, per-node time constants)
  utils.jl             # Math utilities (mean, median, stdev, etc.)
  validation.jl        # Configuration validation and typo detection
ext/
  NeatEvolutionVisualizationExt.jl   # Plots.jl visualization extension
  NeatEvolutionGraphMakieExt.jl      # GraphMakie interactive visualization
examples/              # Runnable examples (XOR, Gymnasium environments via PyCall, etc.)
test/                  # Test suite
```

## Build and Test Commands

```julia
# Install from source
using Pkg
Pkg.add(url="https://github.com/CodeReclaimers/NeatEvolution.jl.git")

# Run tests
Pkg.test("NeatEvolution")

# Or from the repository root
julia --project -e 'using Pkg; Pkg.test()'
```

```bash
# Run an example
julia --project examples/xor/evolve.jl
```

## Coding Conventions

- Follow standard Julia conventions: lowercase function names, CamelCase types.
- Core dependencies: FunctionWrappers, JSON, plus Julia standard library
  (Dates, Random, Serialization, Statistics, TOML).
- Gymnasium-based examples use PyCall.
- Configuration is file-driven (TOML format), compatible with the neat-python
  config file format where applicable.
- Public API functions and types should have docstrings.

## Key Design Decisions

- **Per-node time constants:** Like neat-python 2.0, CTRNN nodes each have their own
  evolvable time constant rather than sharing a global value. This was validated by
  systematic experiments on the Lorenz attractor showing ~2x performance degradation
  with fixed time constants.
- **Algorithm fidelity:** The implementation follows the original NEAT paper closely,
  including speciation by genomic distance, fitness sharing, and structural mutation
  (add node, add connection).
- **Port of neat-python:** The design mirrors neat-python's architecture. If you are
  familiar with neat-python's `DefaultGenome`, `DefaultReproduction`, etc., the Julia
  equivalents follow the same patterns.

## Common Tasks

### Adding a new example
Place it in `examples/<name>/` with its own config file and a Julia script as entry point.
Examples should be self-contained.

### Working with CTRNN networks
Create via `CTRNNNetwork(genome, config)`, then simulate with `advance!(network, inputs, dt, steps)`.
Each node has its own time constant that evolves alongside weights and topology.

## Important Warnings

- The `main` branch is the primary development branch.
- Do not add heavy dependencies to the core package — keep it lightweight.
- Gymnasium examples require a working Python environment accessible via PyCall.
- This is a younger project than neat-python; the API may still evolve between minor versions.

## Citation

If referencing this project in generated text or code comments:

```bibtex
@software{McIntyre_NeatEvolution_jl,
  author = {McIntyre, Alan},
  title = {{NeatEvolution.jl}},
  url = {https://github.com/CodeReclaimers/NeatEvolution.jl},
  doi = {10.5281/zenodo.19025463},
  version = {0.1.1},
  year = {2026}
}
```

# NEAT.jl Visualization Plan

## Overview
This document outlines a plan to port visualization capabilities from neat-python to NEAT.jl, enabling users to visualize evolution progress, network topologies, and species dynamics.

## Current neat-python Visualization Features

### 1. Statistics Tracking (`neat.statistics.StatisticsReporter`)
- Tracks best genome per generation
- Collects fitness statistics (mean, stdev, median)
- Records species sizes and fitness per generation
- Exports data to CSV files

### 2. Fitness Plots (`visualize.plot_stats`)
- Line plot showing:
  - Best fitness per generation (red)
  - Average fitness per generation (blue)
  - ±1 standard deviation bands (green)
- Optional log scale for fitness axis
- Saves to SVG/PNG

### 3. Speciation Plots (`visualize.plot_species`)
- Stacked area chart showing species sizes over generations
- Each species gets a different color
- Shows species emergence and extinction

### 4. Network Topology (`visualize.draw_net`)
- Uses Graphviz to render neural network structure
- Features:
  - Input nodes (gray boxes)
  - Output nodes (blue circles)
  - Hidden nodes (white circles)
  - Connection weights (line thickness and color)
  - Disabled connections (dotted lines)
  - Optional pruning of unused nodes
  - Customizable node names and colors

## Julia Visualization Ecosystem

### Option 1: Plots.jl (Recommended for Phase 1)
**Pros:**
- Simple, high-level API
- Multiple backends (GR, PlotlyJS, PyPlot)
- Good for fitness and species plots
- Well-documented, stable
- Easy to install

**Cons:**
- Not ideal for network graphs
- Less interactive than modern alternatives

### Option 2: Makie.jl (Recommended for Phase 2)
**Pros:**
- Modern, GPU-accelerated
- Beautiful, publication-quality graphics
- Highly interactive (GLMakie, WGLMakie)
- GraphMakie.jl for network visualization
- Very flexible

**Cons:**
- Heavier dependency
- Steeper learning curve
- Longer compilation times

### Option 3: GraphPlot.jl + Compose.jl (For Network Graphs)
**Pros:**
- Lightweight
- Specifically designed for graph visualization
- Works well with Graphs.jl
- SVG export

**Cons:**
- Limited styling options
- Less actively maintained

### Option 4: TikzGraphs.jl (For Publications)
**Pros:**
- Publication-quality LaTeX/TikZ output
- Perfect control over appearance
- Integrates with academic workflows

**Cons:**
- Requires LaTeX installation
- Not interactive
- Overkill for quick visualization

## Recommended Implementation Plan

### Phase 1: Statistics Collection & Basic Plots (Essential)
**Priority: HIGH - Week 1-2**

#### 1.1 Statistics Reporter
Create `src/statistics.jl`:
```julia
mutable struct StatisticsReporter <: Reporter
    most_fit_genomes::Vector{Genome}
    generation_statistics::Vector{Dict{Int, Dict{Int, Float64}}}
    # species_id => genome_id => fitness
end
```

**Methods to implement:**
- `post_evaluate!(reporter, config, population, species, best_genome)`
- `get_fitness_mean(reporter)::Vector{Float64}`
- `get_fitness_stdev(reporter)::Vector{Float64}`
- `get_fitness_median(reporter)::Vector{Float64}`
- `get_species_sizes(reporter)::Vector{Vector{Int}}`
- `get_species_fitness(reporter)::Vector{Vector{Float64}}`
- `best_genome(reporter)::Genome`
- `best_genomes(reporter, n::Int)::Vector{Genome}`
- `save_statistics(reporter, prefix="neat_stats")`

#### 1.2 Fitness Plots
Create `src/visualization/fitness.jl`:
```julia
using Plots

function plot_fitness(reporter::StatisticsReporter;
                      ylog=false,
                      filename="fitness_history.png")
    # Plot best, average, ±1 std dev
end

function plot_fitness_comparison(reporters::Vector{StatisticsReporter},
                                  labels::Vector{String};
                                  filename="fitness_comparison.png")
    # Compare multiple runs
end
```

**Dependencies:** `Plots.jl`

### Phase 2: Species Visualization (Important)
**Priority: MEDIUM - Week 3**

Create `src/visualization/species.jl`:
```julia
function plot_species(reporter::StatisticsReporter;
                      filename="speciation.png")
    # Stacked area plot of species sizes
end

function plot_species_fitness(reporter::StatisticsReporter;
                              filename="species_fitness.png")
    # Line plots of each species' average fitness
end
```

### Phase 3: Network Topology Visualization (Nice-to-have)
**Priority: MEDIUM - Week 4-5**

Create `src/visualization/network.jl`:

**Option A: Using GraphPlot.jl (Simpler)**
```julia
using Graphs, GraphPlot, Colors

function draw_network(genome::Genome, config::GenomeConfig;
                      filename="network.svg",
                      node_names=Dict{Int,String}(),
                      show_disabled=true,
                      prune_unused=false)
    # Create directed graph
    # Style nodes by type (input/output/hidden)
    # Style edges by weight and enabled status
    # Save to SVG
end
```

**Option B: Using GraphMakie.jl (More powerful)**
```julia
using GraphMakie, GLMakie, Graphs

function draw_network_interactive(genome::Genome, config::GenomeConfig;
                                   layout=:spring)
    # Interactive network visualization
    # Can zoom, pan, hover for details
    # Animated weight changes
end
```

### Phase 4: Advanced Visualizations (Optional)
**Priority: LOW - Future**

#### 4.1 Evolution Animation
```julia
function animate_evolution(reporter::StatisticsReporter;
                           filename="evolution.gif",
                           fps=5)
    # Animate network topology changes over generations
    # Show fitness improvement
end
```

#### 4.2 Genome Comparison
```julia
function compare_genomes(genome1::Genome, genome2::Genome, config::GenomeConfig;
                        filename="genome_comparison.png")
    # Side-by-side or diff view
    # Highlight differences in structure
end
```

#### 4.3 Behavior Heatmaps
```julia
function plot_activation_heatmap(genome::Genome, config::GenomeConfig,
                                 input_ranges;
                                 filename="activation_heatmap.png")
    # For 2D inputs, show network output as heatmap
    # Useful for XOR, classification problems
end
```

## Implementation Details

### Package Structure
```
NEAT.jl/
├── src/
│   ├── statistics.jl          # Statistics collection
│   └── visualization/
│       ├── visualization.jl    # Module file
│       ├── fitness.jl         # Fitness plots
│       ├── species.jl         # Species plots
│       └── network.jl         # Network topology
├── examples/
│   └── xor/
│       ├── evolve.jl
│       └── visualize.jl       # Visualization example
└── docs/
    └── visualization.md        # User guide
```

### Dependencies to Add

**Phase 1 (Required):**
```toml
[deps]
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"

[compat]
Plots = "1.38"
```

**Phase 2 (Optional):**
```toml
[weakdeps]
Graphs = "86223c79-3864-5bf0-83f7-82e725a168b6"
GraphPlot = "a2cc645c-3eea-5389-862e-a155d0052231"
GraphMakie = "1ecd5474-83a3-4783-bb4f-06765db800d2"
Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"

[extensions]
NEATGraphsExt = "Graphs"
NEATGraphPlotExt = ["Graphs", "GraphPlot"]
NEATMakieExt = ["Graphs", "GraphMakie", "Makie"]
```

This uses Julia's package extensions feature (Julia 1.9+) to avoid hard dependencies on visualization libraries.

### Testing Strategy

Create `test/test_visualization.jl`:
```julia
@testset "Visualization" begin
    @testset "Statistics Collection" begin
        # Test reporter collects data correctly
        # Test statistics calculations
    end

    @testset "Plot Generation" begin
        # Test plots can be created without errors
        # Test file outputs exist
        # Don't display (headless testing)
    end

    @testset "Network Rendering" begin
        # Test various genome topologies
        # Test with/without disabled connections
        # Test pruning
    end
end
```

## Example Usage

### Basic Workflow
```julia
using NEAT

# Setup
config = load_config("config.toml")
pop = Population(config)

# Add statistics reporter
stats = StatisticsReporter()
add_reporter!(pop, stats)

# Evolution
winner = run!(pop, eval_genomes, 100)

# Visualize results
using NEAT.Visualization

plot_fitness(stats, filename="fitness.png")
plot_species(stats, filename="species.png")
draw_network(winner, config.genome_config, filename="winner.svg")

# Save statistics for later analysis
save_statistics(stats, prefix="xor_run1")
```

### Advanced Analysis
```julia
# Compare multiple runs
runs = [run_experiment() for _ in 1:5]
stats_list = [r.statistics for r in runs]

plot_fitness_comparison(stats_list,
                       ["Run $i" for i in 1:5],
                       filename="comparison.png")

# Interactive network exploration (if GraphMakie available)
fig = draw_network_interactive(winner, config.genome_config)
display(fig)
```

## Benefits of This Approach

1. **Modular**: Visualization is optional, won't break core functionality
2. **Flexible**: Users can choose their preferred plotting library
3. **Extensible**: Easy to add new visualization types
4. **Compatible**: Matches neat-python API where sensible
5. **Julia-idiomatic**: Uses Julia's package extension system
6. **Lightweight**: Core package doesn't depend on heavy plotting libraries

## Migration Path from neat-python

For users familiar with neat-python:

| neat-python | NEAT.jl |
|-------------|---------|
| `neat.StatisticsReporter()` | `NEAT.StatisticsReporter()` |
| `p.add_reporter(stats)` | `add_reporter!(pop, stats)` |
| `visualize.plot_stats(stats)` | `plot_fitness(stats)` |
| `visualize.plot_species(stats)` | `plot_species(stats)` |
| `visualize.draw_net(config, genome)` | `draw_network(genome, config.genome_config)` |

## Timeline Estimate

- **Phase 1** (Statistics + Basic Plots): 1-2 weeks
- **Phase 2** (Species Visualization): 1 week
- **Phase 3** (Network Topology): 1-2 weeks
- **Phase 4** (Advanced Features): 2-4 weeks (ongoing)

**Total for MVP (Phases 1-2):** 2-3 weeks of development time

## Next Steps

1. ✅ Create this plan document
2. ⬜ Gather feedback from maintainers/users
3. ⬜ Implement Phase 1: Statistics collection
4. ⬜ Add tests for statistics
5. ⬜ Implement Phase 1: Basic fitness plots
6. ⬜ Create example visualization script for XOR
7. ⬜ Document visualization features
8. ⬜ Proceed to Phase 2 if Phase 1 is successful

## Open Questions

1. Should visualization be in core package or separate `NEATViz.jl` package?
   - **Recommendation**: Core package with weak deps/extensions

2. Which plotting backend should be the default?
   - **Recommendation**: Plots.jl with GR backend (fastest, most compatible)

3. Should we support interactive plots (Pluto/Jupyter)?
   - **Recommendation**: Yes, but as secondary priority

4. How to handle animations/GIFs?
   - **Recommendation**: Phase 4, use Plots.jl animation framework

## References

- neat-python visualize.py: Examples from XOR and other demos
- Plots.jl docs: https://docs.juliaplots.org
- GraphPlot.jl: https://github.com/JuliaGraphs/GraphPlot.jl
- GraphMakie.jl: https://github.com/MakieOrg/GraphMakie.jl
- Julia Package Extensions: https://pkgdocs.julialang.org/v1/creating-packages/#Conditional-loading-of-code-in-packages-(Extensions)

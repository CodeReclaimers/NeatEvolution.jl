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

### Phase 1: Statistics Collection & Basic Plots ✅ COMPLETED
**Priority: HIGH - Week 1-2**
**Status: Implemented and tested (commit a5b6812)**

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

**Dependencies:** `Plots.jl` (added as weak dependency with package extension)

**Implementation Details:**
- Created `src/statistics.jl` with StatisticsReporter type
- All methods implemented and tested
- Added to `ext/NEATVisualizationExt.jl` as package extension
- `plot_fitness()` shows best, average, and ±1σ bands
- `plot_species()` creates stacked area chart
- `plot_fitness_comparison()` compares multiple runs
- CSV export via `save_statistics()`
- 15 tests covering all functionality
- Updated XOR example with visualization

### Phase 2: Network Structure Visualization ✅ COMPLETED
**Priority: MEDIUM - Week 3**
**Status: Implemented and tested (commit e8b4460)**

**Implementation Details:**
- Implemented `draw_net()` function in `ext/NEATVisualizationExt.jl`
- Uses Plots.jl instead of Graphviz for simpler dependency management
- Layer-based automatic layout using `feed_forward_layers()`
- Color-coded nodes: input (green), output (blue), hidden (white)
- Connection styling by weight: red (negative), green (positive)
- Line thickness proportional to weight magnitude
- Disabled connections shown as dashed gray lines
- Support for custom node names, colors, and pruning
- `draw_net_comparison()` for side-by-side genome comparison
- 10 tests covering all network visualization features
- Updated XOR example to generate winner network diagram

### Phase 3: Advanced Visualizations (Optional)
**Priority: LOW - Future**

#### 3.1 Evolution Animation
```julia
function animate_evolution(reporter::StatisticsReporter;
                           filename="evolution.gif",
                           fps=5)
    # Animate network topology changes over generations
    # Show fitness improvement
end
```

#### 3.2 Interactive Network Visualization
```julia
using GraphMakie, GLMakie, Graphs

function draw_network_interactive(genome::Genome, config::GenomeConfig;
                                   layout=:spring)
    # Interactive network visualization with GraphMakie
    # Can zoom, pan, hover for details
    # Animated weight changes
end
```

#### 3.3 Behavior Heatmaps
```julia
function plot_activation_heatmap(genome::Genome, config::GenomeConfig,
                                 input_ranges;
                                 filename="activation_heatmap.png")
    # For 2D inputs, show network output as heatmap
    # Useful for XOR, classification problems
end
```

## Implementation Details

### Package Structure (Actual Implementation)
```
NEAT.jl/
├── src/
│   ├── NEAT.jl                # Main module with function stubs
│   └── statistics.jl          # ✅ Statistics collection
├── ext/
│   └── NEATVisualizationExt.jl  # ✅ Package extension with all viz
├── examples/
│   └── xor/
│       ├── evolve.jl
│       └── evolve_with_visualization.jl  # ✅ Complete example
├── test/
│   └── runtests.jl            # ✅ 25 visualization tests
└── docs/
    └── VISUALIZATION_PLAN.md  # This document
```

**Note:** Using Julia's package extensions (Julia 1.9+) keeps visualization as optional dependency.

### Dependencies (Actual Implementation) ✅

**Current Project.toml:**
```toml
[weakdeps]
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"

[extensions]
NEATVisualizationExt = "Plots"

[compat]
Plots = "1.38"
julia = "1.9"

[extras]
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"

[targets]
test = ["Test", "Plots"]
```

**Benefits of this approach:**
- Plots.jl is only loaded when user explicitly does `using Plots`
- Core NEAT.jl has no visualization dependencies
- Single extension handles all visualization (fitness, species, networks)
- Tests include Plots to verify visualization works

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

| neat-python | NEAT.jl (Implemented) |
|-------------|---------|
| `neat.StatisticsReporter()` | `NEAT.StatisticsReporter()` ✅ |
| `p.add_reporter(stats)` | `add_reporter!(pop, stats)` ✅ |
| `visualize.plot_stats(stats)` | `plot_fitness(stats)` ✅ |
| `visualize.plot_species(stats)` | `plot_species(stats)` ✅ |
| `visualize.draw_net(config, genome)` | `draw_net(genome, config.genome_config)` ✅ |

**Note:** Requires `using Plots` to load the visualization extension.

## Timeline Estimate

- **Phase 1** (Statistics + Fitness/Species Plots): ✅ Completed
- **Phase 2** (Network Structure Visualization): ✅ Completed
- **Phase 3** (Advanced Features): ⬜ Future work (optional)

**Total implementation time:** Completed in Phase 1 & 2

## Completion Summary

### What Was Implemented ✅

1. **Statistics Collection** (src/statistics.jl:216)
   - StatisticsReporter with full tracking
   - Fitness statistics (mean, stdev, median)
   - Species tracking and sizes
   - CSV export functionality
   - 15 comprehensive tests

2. **Fitness & Species Visualization** (ext/NEATVisualizationExt.jl:18-185)
   - plot_fitness() with best, average, ±1σ bands
   - plot_species() with stacked area charts
   - plot_fitness_comparison() for multi-run analysis
   - 7 tests covering all plot types

3. **Network Structure Visualization** (ext/NEATVisualizationExt.jl:187-403)
   - draw_net() with layer-based layout
   - Color-coded nodes and weight-based connections
   - draw_net_comparison() for genome comparisons
   - 10 tests covering all network features

4. **Documentation & Examples**
   - Updated XOR example with full visualization
   - examples/xor/evolve_with_visualization.jl
   - Comprehensive inline documentation

### Test Results
- **Total tests:** 226 (all passing)
- **Visualization tests:** 25
- **Test coverage:** Statistics, fitness plots, species plots, network diagrams

## Next Steps

1. ✅ Create this plan document
2. ✅ Implement Phase 1: Statistics collection
3. ✅ Add tests for statistics
4. ✅ Implement Phase 1: Fitness and species plots
5. ✅ Implement Phase 2: Network structure visualization
6. ✅ Create example visualization script for XOR
7. ✅ Document visualization features (inline docs)
8. ⬜ Add user guide documentation (optional)
9. ⬜ Consider Phase 3: Advanced features (animations, interactive plots)

## Implementation Decisions ✅

1. **Should visualization be in core package or separate `NEATViz.jl` package?**
   - ✅ **Decision**: Core package with weak dependencies and package extensions
   - **Result**: Clean integration, no hard dependencies, users opt-in with `using Plots`

2. **Which plotting backend should be the default?**
   - ✅ **Decision**: Plots.jl with GR backend (fastest, most compatible)
   - **Result**: Works well, fast rendering, no issues

3. **Network visualization approach?**
   - ✅ **Decision**: Use Plots.jl instead of Graphviz/GraphPlot
   - **Rationale**: Single dependency, layer-based layout works well, simpler for users
   - **Result**: Clean visualizations with automatic layout

4. **Future considerations:**
   - Interactive plots (Pluto/Jupyter) - can be added using show_plot=true parameter
   - Animations/GIFs - potential Phase 3 using Plots.jl animation framework
   - Alternative backends (Makie, GraphMakie) - could add as additional extensions

## References

- neat-python visualize.py: Examples from XOR and other demos
- Plots.jl docs: https://docs.juliaplots.org
- GraphPlot.jl: https://github.com/JuliaGraphs/GraphPlot.jl
- GraphMakie.jl: https://github.com/MakieOrg/GraphMakie.jl
- Julia Package Extensions: https://pkgdocs.julialang.org/v1/creating-packages/#Conditional-loading-of-code-in-packages-(Extensions)

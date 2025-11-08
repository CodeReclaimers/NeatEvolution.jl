# Visualization Guide

## Overview

NEAT.jl provides comprehensive visualization capabilities to help you understand and debug the evolutionary process. This guide covers all visualization features with practical examples and best practices.

## Setup

### Installation

Visualization is an optional feature that requires the Plots.jl package:

```julia
using Pkg
Pkg.add("Plots")
```

### Enabling Visualization

Simply load Plots.jl after loading NEAT.jl:

```julia
using NEAT
using Plots  # This automatically loads the visualization extension

# Now visualization functions are available
config = load_config("config.toml")
pop = Population(config)
stats = StatisticsReporter()
add_reporter!(pop, stats)
```

The visualization extension uses Julia's package extension system (Julia 1.9+), so Plots.jl is only loaded when you explicitly use it.

## Statistics Collection

### StatisticsReporter

Before you can visualize anything, you need to collect statistics during evolution:

```julia
using NEAT

config = load_config("config.toml")
pop = Population(config)

# Create statistics reporter
stats = StatisticsReporter()
add_reporter!(pop, stats)

# Run evolution
winner = run!(pop, eval_genomes, 100)

# Now stats contains all the data you need for visualization
```

**What it tracks:**
- Best genome per generation
- Fitness statistics (mean, standard deviation, median)
- Species sizes and fitness values
- Complete population snapshots

### Accessing Statistics Programmatically

```julia
# Get fitness data
mean_fitness = get_fitness_mean(stats)      # Vector{Float64}
stdev_fitness = get_fitness_stdev(stats)    # Vector{Float64}
median_fitness = get_fitness_median(stats)  # Vector{Float64}

# Get species data
species_sizes = get_species_sizes(stats)    # Vector{Vector{Int}}
species_fitness = get_species_fitness(stats) # Vector{Vector{Float64}}

# Get best genomes
best = best_genome(stats)                    # Single best genome
top5 = best_genomes(stats, 5)               # Top 5 genomes

# Export to CSV for external analysis
save_statistics(stats, prefix="experiment1")
# Creates: experiment1_fitness.csv
#          experiment1_speciation.csv
#          experiment1_species_fitness.csv
```

## Fitness Visualization

### Basic Fitness Plot

Shows how fitness evolves over generations:

```julia
using Plots

plot_fitness(stats, filename="fitness.png")
```

**What it shows:**
- **Red line**: Best fitness per generation
- **Blue line**: Average fitness per generation
- **Green dashed lines**: ±1 standard deviation bands

![Example fitness plot](../examples/xor/fitness.png)

### Customizing Fitness Plots

```julia
plot_fitness(stats,
    ylog=true,                          # Logarithmic y-axis
    filename="fitness_log.png",
    title="XOR Evolution Progress",
    show_plot=true                      # Display interactively
)
```

**When to use log scale:**
- Wide fitness ranges (e.g., 0.001 to 1000)
- Exponential improvement
- Early generation details are important

### Comparing Multiple Runs

Compare fitness across different experiments:

```julia
# Run multiple experiments
stats1 = run_experiment(config1)
stats2 = run_experiment(config2)
stats3 = run_experiment(config3)

# Compare them
plot_fitness_comparison(
    [stats1, stats2, stats3],
    ["Standard", "High Mutation", "Large Population"],
    filename="comparison.png",
    title="Parameter Comparison"
)
```

**Use cases:**
- Testing different hyperparameters
- Comparing initialization strategies
- Evaluating fitness function variants
- Reproducibility checks across runs

## Species Visualization

### Species Over Time

Visualize how species emerge, grow, and go extinct:

```julia
plot_species(stats,
    filename="species.png",
    title="Species Dynamics",
    show_plot=false
)
```

**What it shows:**
- Stacked area chart with each species as a colored band
- Species height = population size
- New species appear as new colors
- Extinct species disappear

**Interpreting the plot:**

```
Good patterns:
- Multiple species maintained (diversity)
- Gradual species turnover (exploration)
- Some long-lived species (exploitation)

Warning signs:
- Single species dominates (premature convergence)
- Too many species (threshold too high)
- Rapid species turnover (instability)
```

### Species Analysis Tips

```julia
# Get detailed species information
species_sizes = get_species_sizes(stats)

# How many species per generation?
num_species_per_gen = length.(species_sizes)
println("Max species: ", maximum(num_species_per_gen))
println("Min species: ", minimum(num_species_per_gen))

# Average species size
avg_species_sizes = [isempty(s) ? 0.0 : mean(s) for s in species_sizes]
```

## Network Topology Visualization

### Drawing Networks

Visualize the structure of evolved neural networks:

```julia
# Draw the winner network
draw_net(winner, config.genome_config,
    filename="winner_network.png"
)
```

**Features:**
- **Green boxes**: Input nodes
- **Blue boxes**: Output nodes
- **White circles**: Hidden nodes
- **Line color**: Red (negative), green (positive) weights
- **Line thickness**: Proportional to weight magnitude
- **Dashed lines**: Disabled connections

### Customizing Network Diagrams

```julia
# Add custom node names
node_names = Dict(
    -1 => "Input 1",
    -2 => "Input 2",
    0 => "Output",
    1 => "Hidden A"
)

# Custom node colors
node_colors = Dict(
    1 => :yellow,  # Highlight important hidden node
    0 => :orange   # Different output color
)

draw_net(winner, config.genome_config,
    filename="network_labeled.png",
    node_names=node_names,
    node_colors=node_colors,
    show_disabled=false,    # Hide disabled connections
    prune_unused=true,      # Remove unconnected nodes
    show_plot=true
)
```

### Comparing Network Structures

Compare multiple genomes side-by-side:

```julia
# Get top 3 genomes
top3 = best_genomes(stats, 3)

draw_net_comparison(
    top3,
    config.genome_config,
    labels=["Best", "2nd Best", "3rd Best"],
    filename="top3_comparison.png",
    show_disabled=false
)
```

**Use cases:**
- Compare different solutions to the same problem
- Understand structural diversity in population
- Visualize convergence (or lack thereof)
- Debug why certain genomes perform better

## Activation Heatmaps

### 2D Input Space Visualization

For problems with 2 inputs, visualize network behavior across input space:

```julia
# XOR example
plot_activation_heatmap(winner, config.genome_config,
    x_range=(-0.5, 1.5),      # Input 1 range
    y_range=(-0.5, 1.5),      # Input 2 range
    resolution=100,            # Grid resolution
    output_index=1,            # Which output to plot
    filename="xor_heatmap.png",
    title="XOR Decision Boundary"
)
```

**What it shows:**
- Color intensity = network output value
- X-axis = first input
- Y-axis = second input
- Decision boundaries visible as color transitions

**Example interpretation for XOR:**
```
Expected pattern:
- Dark corners at (0,0) and (1,1) → output ≈ 0
- Bright corners at (0,1) and (1,0) → output ≈ 1
- Clear diagonal boundary
```

### Comparing Activation Patterns

Compare how different genomes solve the same problem:

```julia
# Compare multiple solutions
candidates = best_genomes(stats, 4)

plot_activation_comparison(
    candidates,
    config.genome_config,
    labels=["Gen 10", "Gen 25", "Gen 50", "Final"],
    filename="evolution_heatmaps.png",
    resolution=50
)
```

**Use cases:**
- Visualize solution convergence over evolution
- Compare different evolved strategies
- Debug fitness function issues
- Understand network decision-making

## Evolution Animation

### Creating Evolution GIFs

Animate network topology changes over time:

```julia
animate_evolution(stats, config.genome_config,
    filename="evolution.gif",
    fps=2,                     # Frames per second
    show_disabled=false,
    node_names=Dict(-1=>"x1", -2=>"x2", 0=>"out")
)
```

**What it shows:**
- Network structure at each generation (or sampled)
- Fitness progression in title
- Structural changes over time

**Performance notes:**
- Automatically samples up to 50 frames for large runs
- Adjust `fps` for faster/slower playback
- Lower resolution for smaller file sizes

### Animation Best Practices

```julia
# For long runs, animation auto-samples generations
# Manual control:
total_gens = length(stats.most_fit_genomes)
sample_every = max(1, total_gens ÷ 50)  # ~50 frames

# For presentation:
animate_evolution(stats, config.genome_config,
    filename="presentation.gif",
    fps=1,              # Slower for presentations
    show_disabled=false # Cleaner visualization
)

# For analysis:
animate_evolution(stats, config.genome_config,
    filename="detailed.gif",
    fps=5,              # Faster playback
    show_disabled=true  # See all structure
)
```

## Complete Workflow Example

### Typical Visualization Pipeline

```julia
using NEAT
using Plots

# 1. Setup
config = load_config("config.toml")
pop = Population(config)

# 2. Add reporters
stdout_reporter = StdOutReporter(true)
stats = StatisticsReporter()
add_reporter!(pop, stdout_reporter)
add_reporter!(pop, stats)

# 3. Run evolution
println("Starting evolution...")
winner = run!(pop, eval_genomes, 100)
println("Evolution complete!")

# 4. Generate all visualizations
println("Generating visualizations...")

# Fitness evolution
plot_fitness(stats,
    filename="output/fitness.png",
    title="Fitness Over Time"
)

# Species dynamics
plot_species(stats,
    filename="output/species.png",
    title="Species Evolution"
)

# Winner network
draw_net(winner, config.genome_config,
    filename="output/winner.png",
    prune_unused=true
)

# For 2D problems: activation heatmap
plot_activation_heatmap(winner, config.genome_config,
    x_range=(-0.5, 1.5),
    y_range=(-0.5, 1.5),
    filename="output/heatmap.png"
)

# Evolution animation
animate_evolution(stats, config.genome_config,
    filename="output/evolution.gif",
    fps=3
)

# 5. Export data for external analysis
save_statistics(stats, prefix="output/experiment1")

println("All visualizations saved to output/")
```

### Multi-Run Comparison

```julia
using NEAT
using Plots

function run_experiment(config, name)
    pop = Population(config)
    stats = StatisticsReporter()
    add_reporter!(pop, stats)
    add_reporter!(pop, StdOutReporter(false))  # Quiet

    winner = run!(pop, eval_genomes, 100)

    return (stats=stats, winner=winner, name=name)
end

# Run experiments
results = []

# Experiment 1: Baseline
config1 = load_config("config_baseline.toml")
push!(results, run_experiment(config1, "Baseline"))

# Experiment 2: High mutation
config2 = load_config("config_high_mutation.toml")
push!(results, run_experiment(config2, "High Mutation"))

# Experiment 3: Large population
config3 = load_config("config_large_pop.toml")
push!(results, run_experiment(config3, "Large Population"))

# Compare fitness
plot_fitness_comparison(
    [r.stats for r in results],
    [r.name for r in results],
    filename="comparison_fitness.png"
)

# Compare final solutions
draw_net_comparison(
    [r.winner for r in results],
    config1.genome_config,
    labels=[r.name for r in results],
    filename="comparison_networks.png"
)

# Compare activation patterns
plot_activation_comparison(
    [r.winner for r in results],
    config1.genome_config,
    labels=[r.name for r in results],
    filename="comparison_heatmaps.png"
)
```

## Headless Rendering (CI/Servers)

### Running Without Display

For automated testing or server environments:

```julia
# Set GR backend to non-interactive mode
ENV["GKSwstype"] = "100"

using NEAT
using Plots

# All visualization functions work normally
# Files are saved without trying to display
plot_fitness(stats, filename="fitness.png", show_plot=false)
```

### CI/CD Integration

```julia
# In your CI script
using NEAT
using Plots

# Suppress all display attempts
ENV["GKSwstype"] = "100"
gr()  # Use GR backend

# Run and visualize
stats = run_evolution()
plot_fitness(stats, filename="artifacts/fitness.png")
plot_species(stats, filename="artifacts/species.png")

# Save for later review
save_statistics(stats, prefix="artifacts/run")
```

## Troubleshooting

### Common Issues

**Problem: "Plots.jl not loaded" error**
```julia
# Solution: Load Plots.jl after NEAT.jl
using NEAT
using Plots  # Must come after NEAT
```

**Problem: Blank or empty plots**
```julia
# Solution: Check that statistics were collected
@assert !isempty(stats.most_fit_genomes) "No evolution data"

# Run evolution first
winner = run!(pop, eval_genomes, 100)

# Then visualize
plot_fitness(stats)
```

**Problem: Plot window doesn't appear**
```julia
# Solution: Use show_plot=true
plot_fitness(stats, show_plot=true)

# Or display manually
p = plot_fitness(stats)
display(p)
```

**Problem: Out of memory with large runs**
```julia
# Solution: Sample generations for animation
# (Automatically done, but you can control it)
animate_evolution(stats, config.genome_config,
    filename="evolution.gif",
    fps=5  # Faster playback = fewer frames in memory
)

# Or create plots selectively
plot_fitness(stats)  # Light
# Skip animate_evolution for very long runs
```

**Problem: Network diagram too cluttered**
```julia
# Solution: Prune unused nodes and hide disabled connections
draw_net(genome, config.genome_config,
    prune_unused=true,
    show_disabled=false,
    filename="clean_network.png"
)
```

### Performance Tips

1. **Use GR backend** (default) for fastest rendering:
   ```julia
   gr()  # Already default, but explicit
   ```

2. **Disable interactive display** for batch processing:
   ```julia
   plot_fitness(stats, show_plot=false)  # Default
   ```

3. **Sample large datasets**:
   ```julia
   # For 1000+ generations, sample for visualization
   every_n = 10
   sampled_stats = sample_statistics(stats, every_n)
   plot_fitness(sampled_stats)
   ```

4. **Reduce resolution** for faster heatmaps:
   ```julia
   plot_activation_heatmap(genome, config,
       resolution=25  # Instead of 100
   )
   ```

## Customization Examples

### Custom Color Schemes

```julia
# Define custom colors for species plot
using Plots
my_colors = palette(:tab20)  # Use a specific palette

plot_species(stats,
    filename="species_custom.png"
)
# Note: Color selection is automatic, but you can modify
# the plot object returned for full customization
```

### Publication-Quality Figures

```julia
using Plots
using Plots.PlotMeasures  # For margin control

# High DPI settings
default(dpi=300, size=(800, 600))

# Larger fonts
default(
    titlefontsize=16,
    guidefontsize=14,
    tickfontsize=12,
    legendfontsize=12
)

# Generate plots
plot_fitness(stats,
    filename="paper_fitness.png",
    title="NEAT Evolution on XOR Problem"
)

# Manual customization
p = plot_fitness(stats, show_plot=false)
plot!(p,
    margin=5mm,
    background_color=:white,
    foreground_color=:black
)
savefig(p, "publication_fitness.pdf")  # Vector format
```

### Interactive Notebooks

```julia
# In Jupyter/Pluto notebooks
using NEAT
using Plots

# Evolution
stats = run_evolution()

# Interactive display (automatically shown in notebooks)
plot_fitness(stats, show_plot=true)
plot_species(stats, show_plot=true)

# Network with interaction
p = draw_net(winner, config.genome_config)
display(p)  # Can zoom, pan in notebook
```

## Advanced Techniques

### Custom Visualizations

Build on top of the statistics data:

```julia
using Plots

# Custom plot: Complexity over time
function plot_complexity(stats::StatisticsReporter)
    generations = 1:length(stats.most_fit_genomes)
    num_nodes = [length(g.nodes) for g in stats.most_fit_genomes]
    num_connections = [length(g.connections) for g in stats.most_fit_genomes]

    p = plot(generations, num_nodes,
        label="Nodes",
        xlabel="Generation",
        ylabel="Count",
        title="Network Complexity Evolution",
        linewidth=2
    )

    plot!(p, generations, num_connections,
        label="Connections",
        linewidth=2
    )

    savefig(p, "complexity.png")
    return p
end

plot_complexity(stats)
```

### Animated Comparisons

```julia
# Create side-by-side animation comparing two runs
using Plots

function animate_comparison(stats1, stats2, config)
    anim = @animate for i in 1:length(stats1.most_fit_genomes)
        g1 = stats1.most_fit_genomes[i]
        g2 = stats2.most_fit_genomes[i]

        p1 = draw_net(g1, config, show_plot=false)
        title!(p1, "Method 1 - Gen $i")

        p2 = draw_net(g2, config, show_plot=false)
        title!(p2, "Method 2 - Gen $i")

        plot(p1, p2, layout=(1, 2), size=(1200, 400))
    end

    gif(anim, "comparison.gif", fps=2)
end
```

### 3D Fitness Landscapes

For problems with 2 parameters:

```julia
using Plots

function plot_fitness_landscape(genome, config, param1_range, param2_range)
    x = range(param1_range[1], param1_range[2], length=50)
    y = range(param2_range[1], param2_range[2], length=50)

    net = FeedForwardNetwork(genome, config)

    z = [activate!(net, [xi, yi])[1] for yi in y, xi in x]

    surface(x, y, z,
        xlabel="Parameter 1",
        ylabel="Parameter 2",
        zlabel="Output",
        title="Fitness Landscape",
        camera=(45, 30)
    )

    savefig("landscape_3d.png")
end
```

## Best Practices

### What to Visualize When

**During Development:**
- `plot_fitness()` - Track if evolution is working
- `plot_species()` - Check diversity is maintained
- Console output with `StdOutReporter(true)` for real-time feedback

**For Debugging:**
- `draw_net()` - Inspect specific genome structures
- `plot_activation_heatmap()` - Verify network behavior
- `animate_evolution()` - See when/how solutions emerge

**For Analysis:**
- `plot_fitness_comparison()` - Compare hyperparameters
- `draw_net_comparison()` - Understand solution diversity
- `save_statistics()` - Export for statistical analysis

**For Presentation:**
- High-DPI plots with custom titles
- Animations showing evolution process
- Heatmaps demonstrating learned behavior
- Clean networks with `prune_unused=true`

### Organizing Output

```julia
# Create organized output directory structure
mkpath("output/plots")
mkpath("output/networks")
mkpath("output/data")
mkpath("output/animations")

# Save systematically
plot_fitness(stats, filename="output/plots/fitness.png")
plot_species(stats, filename="output/plots/species.png")
draw_net(winner, config, filename="output/networks/winner.png")
animate_evolution(stats, config, filename="output/animations/evolution.gif")
save_statistics(stats, prefix="output/data/run")
```

## See Also

- [Getting Started Guide](getting_started.md) - Basic NEAT.jl usage
- [XOR Example](xor_example.md) - Complete example with visualization
- [API Reference](api_reference.md) - Function signatures and details
- [Configuration Reference](config_file.md) - Tuning parameters that affect evolution
- [Plots.jl Documentation](https://docs.juliaplots.org/) - Advanced plotting customization

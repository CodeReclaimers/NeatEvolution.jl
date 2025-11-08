# Interactive Visualization Example for XOR Problem
# This example demonstrates the interactive visualization capabilities using GraphMakie

using NEAT

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

# Load configuration
config_path = joinpath(@__DIR__, "config.toml")
config = load_config(config_path)

# Create population with reporters
pop = Population(config)
add_reporter!(pop, StdOutReporter(true))

stats = StatisticsReporter()
add_reporter!(pop, stats)

println("Starting XOR evolution...")
println("=" ^ 50)

# Run evolution
winner = run!(pop, eval_genomes, 100)

println("=" ^ 50)
println("Evolution complete!")
println("Winner fitness: ", winner.fitness)
println()

# ============================================================================
# Static Visualizations (using Plots.jl)
# ============================================================================

println("Generating static visualizations with Plots.jl...")

using Plots

# Create output directory
mkpath("output")

# 1. Fitness evolution plot
plot_fitness(stats,
    filename="output/fitness_interactive.png",
    title="XOR Evolution - Fitness Over Time"
)
println("✓ Saved fitness plot to output/fitness_interactive.png")

# 2. Species dynamics
plot_species(stats,
    filename="output/species_interactive.png",
    title="XOR Evolution - Species Dynamics"
)
println("✓ Saved species plot to output/species_interactive.png")

# 3. Winner network (static)
node_names = Dict(
    config.genome_config.input_keys[1] => "x₁",
    config.genome_config.input_keys[2] => "x₂",
    config.genome_config.output_keys[1] => "XOR"
)

draw_net(winner, config.genome_config,
    filename="output/winner_static.png",
    node_names=node_names,
    prune_unused=true
)
println("✓ Saved static winner network to output/winner_static.png")

# 4. Activation heatmap
plot_activation_heatmap(winner, config.genome_config,
    x_range=(-0.5, 1.5),
    y_range=(-0.5, 1.5),
    resolution=100,
    filename="output/xor_heatmap_interactive.png",
    title="XOR Solution - Decision Boundary"
)
println("✓ Saved activation heatmap to output/xor_heatmap_interactive.png")

# 5. Evolution animation
animate_evolution(stats, config.genome_config,
    filename="output/evolution_interactive.gif",
    fps=3,
    node_names=node_names,
    show_disabled=false
)
println("✓ Saved evolution animation to output/evolution_interactive.gif")

println()

# ============================================================================
# Interactive Visualizations (using GraphMakie + GLMakie)
# ============================================================================

println("=" ^ 50)
println("Interactive Visualization (GraphMakie)")
println("=" ^ 50)
println()

try
    using GLMakie
    using GraphMakie
    using Graphs

    println("Creating interactive network visualization...")
    println()

    # Interactive visualization of the winner network
    println("1. Interactive Winner Network")
    println("   Features:")
    println("   - Rotate view: Left-click + drag")
    println("   - Pan: Right-click + drag")
    println("   - Zoom: Scroll wheel")
    println()

    fig_winner = draw_network_interactive(
        winner,
        config.genome_config,
        layout=:spring,          # Try :spring, :stress, :shell, :spectral, :circular
        node_size=30.0,
        edge_width_scale=3.0,
        show_disabled=false,
        prune_unused=true,
        node_names=node_names,
        title="XOR Winner Network (Interactive)",
        resolution=(1200, 800)
    )

    println("   → Displaying interactive winner network...")
    println("      (Close the window to continue)")
    display(fig_winner)

    # Save a screenshot
    save("output/winner_interactive_screenshot.png", fig_winner)
    println("   ✓ Saved screenshot to output/winner_interactive_screenshot.png")
    println()

    # Compare top 3 networks
    if length(stats.most_fit_genomes) >= 3
        println("2. Interactive Comparison of Top 3 Networks")
        println("   Comparing evolution progress across generations")
        println()

        # Get networks from different stages
        n_gens = length(stats.most_fit_genomes)
        indices = [1, n_gens ÷ 2, n_gens]  # First, middle, last
        top3_genomes = [stats.most_fit_genomes[i] for i in indices]

        fig_comparison = draw_network_comparison_interactive(
            top3_genomes,
            config.genome_config,
            labels=["Gen $(indices[1])", "Gen $(indices[2])", "Gen $(indices[3])"],
            layout=:spring,
            node_size=25.0,
            prune_unused=true,
            resolution=(1800, 600)
        )

        println("   → Displaying network comparison...")
        println("      (Close the window to continue)")
        display(fig_comparison)

        save("output/comparison_interactive_screenshot.png", fig_comparison)
        println("   ✓ Saved screenshot to output/comparison_interactive_screenshot.png")
        println()
    end

    # Demonstrate different layouts
    println("3. Different Layout Algorithms")
    println("   GraphMakie supports multiple layout algorithms:")
    println()

    layouts = [:spring, :stress, :circular]
    for (idx, layout_name) in enumerate(layouts)
        println("   $(idx). Layout: $layout_name")

        fig_layout = draw_network_interactive(
            winner,
            config.genome_config,
            layout=layout_name,
            node_size=25.0,
            prune_unused=true,
            title="XOR Network - $layout_name layout",
            resolution=(800, 800)
        )

        # Save screenshot
        save("output/winner_$(layout_name)_layout.png", fig_layout)
        println("      ✓ Saved to output/winner_$(layout_name)_layout.png")

        # Optional: display each one
        # display(fig_layout)
    end

    println()
    println("=" ^ 50)
    println("All interactive visualizations complete!")
    println("=" ^ 50)
    println()
    println("Output files created:")
    println("  Static (Plots.jl):")
    println("    - output/fitness_interactive.png")
    println("    - output/species_interactive.png")
    println("    - output/winner_static.png")
    println("    - output/xor_heatmap_interactive.png")
    println("    - output/evolution_interactive.gif")
    println()
    println("  Interactive (GraphMakie):")
    println("    - output/winner_interactive_screenshot.png")
    println("    - output/comparison_interactive_screenshot.png")
    println("    - output/winner_spring_layout.png")
    println("    - output/winner_stress_layout.png")
    println("    - output/winner_circular_layout.png")
    println()
    println("To explore interactively, run the individual sections in a REPL")
    println("or Jupyter notebook and use display() on the Figure objects.")

catch e
    if isa(e, ArgumentError) && (occursin("Package GLMakie not found", string(e)) ||
                                  occursin("Package GraphMakie not found", string(e)) ||
                                  occursin("Package Graphs not found", string(e)))
        println()
        println("⚠ Interactive visualization packages not installed")
        println()
        println("To enable interactive visualization, install:")
        println("  using Pkg")
        println("  Pkg.add(\"GLMakie\")")
        println("  Pkg.add(\"GraphMakie\")")
        println("  Pkg.add(\"Graphs\")")
        println()
        println("Static visualizations were created successfully in output/")
    else
        rethrow(e)
    end
end

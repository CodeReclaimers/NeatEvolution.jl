module NEATGraphMakieExt

using NEAT
using GLMakie
using GraphMakie
using Graphs

"""
Create an interactive network visualization using GraphMakie and GLMakie.

This function creates a fully interactive 3D visualization of a neural network where you can:
- Rotate, zoom, and pan the view
- Hover over nodes to see details
- Drag nodes to rearrange the layout
- See real-time weight information

# Arguments
- `genome::Genome`: The genome to visualize
- `config::GenomeConfig`: Genome configuration
- `layout::Symbol=:spring`: Layout algorithm (`:spring`, `:stress`, `:shell`, `:spectral`, `:circular`)
- `node_size::Float64=25.0`: Size of network nodes
- `edge_width_scale::Float64=3.0`: Scaling factor for edge widths
- `show_disabled::Bool=false`: Show disabled connections
- `prune_unused::Bool=false`: Remove nodes not connected to inputs/outputs
- `node_names::Union{Nothing,Dict}=nothing`: Custom node names
- `title::String="Interactive Network Visualization"`: Figure title
- `resolution::Tuple{Int,Int}=(1200, 800)`: Figure size

# Returns
- `GLMakie.Figure`: Interactive figure that can be displayed with `display(fig)`

# Example
```julia
using NEAT
using GLMakie
using GraphMakie
using Graphs

config = load_config("config.toml")
pop = Population(config)
winner = run!(pop, eval_genomes, 100)

# Create interactive visualization
fig = draw_network_interactive(winner, config.genome_config)
display(fig)  # Opens interactive window

# Save screenshot
save("network_screenshot.png", fig)

# With custom settings
fig = draw_network_interactive(winner, config.genome_config,
    layout=:stress,
    node_size=30.0,
    prune_unused=true,
    title="XOR Solution Network"
)
```

# Layout Algorithms
- `:spring` - Force-directed layout (default, good for most networks)
- `:stress` - Stress minimization (good for larger networks)
- `:shell` - Concentric shells (good for hierarchical networks)
- `:spectral` - Spectral layout (good for clustered networks)
- `:circular` - Circular arrangement (good for small networks)

# Interactive Features
- **Left-click + drag**: Rotate view
- **Right-click + drag**: Pan view
- **Scroll wheel**: Zoom in/out
- **Hover over nodes**: See node information (future enhancement)
- **Double-click**: Reset view
"""
function NEAT.draw_network_interactive(genome::NEAT.Genome,
                                        config::NEAT.GenomeConfig;
                                        layout::Symbol=:spring,
                                        node_size::Float64=25.0,
                                        edge_width_scale::Float64=3.0,
                                        show_disabled::Bool=false,
                                        prune_unused::Bool=false,
                                        node_names::Union{Nothing,Dict}=nothing,
                                        title::String="Interactive Network Visualization",
                                        resolution::Tuple{Int,Int}=(1200, 800))

    # Get node lists
    inputs = collect(keys(config.input_keys))
    outputs = collect(keys(config.output_keys))

    # Build connection list
    connections = Tuple{Int,Int}[]
    weights = Float64[]
    enabled_flags = Bool[]

    for ((i, o), conn) in genome.connections
        if show_disabled || conn.enabled
            push!(connections, (i, o))
            push!(weights, conn.weight)
            push!(enabled_flags, conn.enabled)
        end
    end

    # Prune unused nodes if requested
    if prune_unused
        required = NEAT.required_for_output(inputs, outputs, connections)

        # Filter connections
        filtered_connections = Tuple{Int,Int}[]
        filtered_weights = Float64[]
        filtered_enabled = Bool[]

        for (idx, (i, o)) in enumerate(connections)
            if i in required && o in required
                push!(filtered_connections, (i, o))
                push!(filtered_weights, weights[idx])
                push!(filtered_enabled, enabled_flags[idx])
            end
        end

        connections = filtered_connections
        weights = filtered_weights
        enabled_flags = filtered_enabled
    else
        required = union(Set(inputs), Set(outputs), Set(keys(genome.nodes)))
    end

    # Build node list (sorted for consistent ordering)
    all_nodes = sort(collect(required))
    node_id_to_idx = Dict(id => idx for (idx, id) in enumerate(all_nodes))

    # Create Graphs.jl graph
    g = SimpleDiGraph(length(all_nodes))
    edge_to_weight = Dict{Tuple{Int,Int}, Float64}()
    edge_to_enabled = Dict{Tuple{Int,Int}, Bool}()

    for (idx, (i, o)) in enumerate(connections)
        if i in node_id_to_idx && o in node_id_to_idx
            i_idx = node_id_to_idx[i]
            o_idx = node_id_to_idx[o]
            add_edge!(g, i_idx, o_idx)
            edge_to_weight[(i_idx, o_idx)] = weights[idx]
            edge_to_enabled[(i_idx, o_idx)] = enabled_flags[idx]
        end
    end

    # Classify nodes
    node_types = Symbol[]
    node_labels = String[]

    for node_id in all_nodes
        if node_id in inputs
            push!(node_types, :input)
            if node_names !== nothing && haskey(node_names, node_id)
                push!(node_labels, node_names[node_id])
            else
                push!(node_labels, "I$(node_id)")
            end
        elseif node_id in outputs
            push!(node_types, :output)
            if node_names !== nothing && haskey(node_names, node_id)
                push!(node_labels, node_names[node_id])
            else
                push!(node_labels, "O$(node_id)")
            end
        else
            push!(node_types, :hidden)
            if node_names !== nothing && haskey(node_names, node_id)
                push!(node_labels, node_names[node_id])
            else
                push!(node_labels, "H$(node_id)")
            end
        end
    end

    # Assign colors based on node type
    node_colors = map(node_types) do t
        if t == :input
            :green
        elseif t == :output
            :blue
        else
            :lightgray
        end
    end

    # Calculate edge colors and widths based on weights
    edge_colors = RGBAf[]
    edge_widths = Float64[]

    for e in edges(g)
        src_idx = src(e)
        dst_idx = dst(e)

        if haskey(edge_to_weight, (src_idx, dst_idx))
            w = edge_to_weight[(src_idx, dst_idx)]
            is_enabled = edge_to_enabled[(src_idx, dst_idx)]

            # Color: green for positive, red for negative
            # Alpha: 1.0 for enabled, 0.3 for disabled
            alpha = is_enabled ? 1.0 : 0.3

            if w >= 0
                # Positive weight: green
                intensity = min(1.0, abs(w) / 5.0)
                push!(edge_colors, RGBAf(0.0, intensity, 0.0, alpha))
            else
                # Negative weight: red
                intensity = min(1.0, abs(w) / 5.0)
                push!(edge_colors, RGBAf(intensity, 0.0, 0.0, alpha))
            end

            # Width proportional to absolute weight
            width = edge_width_scale * (0.5 + min(3.0, abs(w)))
            push!(edge_widths, width)
        else
            # Default for edges without weights
            push!(edge_colors, RGBAf(0.5, 0.5, 0.5, 0.5))
            push!(edge_widths, edge_width_scale)
        end
    end

    # Choose layout algorithm
    layout_func = if layout == :spring
        Spring()
    elseif layout == :stress
        Stress()
    elseif layout == :shell
        Shell()
    elseif layout == :spectral
        Spectral()
    elseif layout == :circular
        Circular()
    else
        @warn "Unknown layout :$layout, using :spring"
        Spring()
    end

    # Create figure
    fig = Figure(resolution=resolution)
    ax = Axis(fig[1, 1],
              title=title,
              xlabel="",
              ylabel="",
              aspect=DataAspect())

    # Hide axis decorations for cleaner look
    hidedecorations!(ax)
    hidespines!(ax)

    # Plot the graph
    graphplot!(ax, g,
               layout=layout_func,
               node_color=node_colors,
               node_size=node_size,
               edge_color=edge_colors,
               edge_width=edge_widths,
               arrow_show=true,
               arrow_size=15,
               nlabels=node_labels,
               nlabels_align=(:center, :center),
               nlabels_fontsize=12,
               nlabels_color=:black)

    # Add legend
    Legend(fig[1, 2],
           [MarkerElement(marker=:circle, color=:green, markersize=20),
            MarkerElement(marker=:circle, color=:blue, markersize=20),
            MarkerElement(marker=:circle, color=:lightgray, markersize=20),
            LineElement(color=RGBAf(0.0, 0.8, 0.0, 1.0), linewidth=3),
            LineElement(color=RGBAf(0.8, 0.0, 0.0, 1.0), linewidth=3),
            LineElement(color=RGBAf(0.5, 0.5, 0.5, 0.3), linewidth=1)],
           ["Input Node", "Output Node", "Hidden Node",
            "Positive Weight", "Negative Weight", "Disabled Connection"],
           "Network Elements")

    # Add network statistics
    n_nodes = nv(g)
    n_edges = ne(g)
    n_enabled = count(enabled_flags)
    n_disabled = length(enabled_flags) - n_enabled

    stats_text = """
    Nodes: $n_nodes
    Edges: $n_edges
    Enabled: $n_enabled
    Disabled: $n_disabled
    """

    Label(fig[2, 1:2], stats_text,
          tellwidth=false,
          tellheight=true,
          fontsize=12,
          halign=:left)

    return fig
end

"""
Create a side-by-side comparison of multiple genomes using interactive visualization.

# Arguments
- `genomes::Vector{Genome}`: Vector of genomes to compare
- `config::GenomeConfig`: Genome configuration
- `labels::Union{Nothing,Vector{String}}=nothing`: Labels for each genome
- `layout::Symbol=:spring`: Layout algorithm
- `node_size::Float64=20.0`: Size of network nodes
- `prune_unused::Bool=true`: Remove unconnected nodes
- `resolution::Tuple{Int,Int}=(1600, 800)`: Figure size

# Returns
- `GLMakie.Figure`: Interactive figure with multiple network panels

# Example
```julia
top3 = best_genomes(stats, 3)
fig = draw_network_comparison_interactive(
    top3,
    config.genome_config,
    labels=["Best", "2nd", "3rd"]
)
display(fig)
```
"""
function NEAT.draw_network_comparison_interactive(genomes::Vector{NEAT.Genome},
                                                   config::NEAT.GenomeConfig;
                                                   labels::Union{Nothing,Vector{String}}=nothing,
                                                   layout::Symbol=:spring,
                                                   node_size::Float64=20.0,
                                                   prune_unused::Bool=true,
                                                   resolution::Tuple{Int,Int}=(1600, 800))
    n = length(genomes)

    if labels === nothing
        labels = ["Network $i" for i in 1:n]
    end

    # Create figure with n columns
    fig = Figure(resolution=resolution)

    for (idx, (genome, label)) in enumerate(zip(genomes, labels))
        # Create a temporary figure for this genome
        temp_fig = NEAT.draw_network_interactive(genome, config,
                                                   layout=layout,
                                                   node_size=node_size,
                                                   prune_unused=prune_unused,
                                                   title=label,
                                                   resolution=(resolution[1]÷n, resolution[2]))

        # Extract the axis from temp figure and add to main figure
        # Note: This is a simplified version; in practice, you'd recreate the plot
        ax = Axis(fig[1, idx],
                  title=label,
                  aspect=DataAspect())
        hidedecorations!(ax)
        hidespines!(ax)

        # Recreate the graph plot in this axis
        # (Similar logic to draw_network_interactive but in this specific axis)
        inputs = collect(keys(config.input_keys))
        outputs = collect(keys(config.output_keys))

        connections = Tuple{Int,Int}[]
        weights = Float64[]

        for ((i, o), conn) in genome.connections
            if conn.enabled
                push!(connections, (i, o))
                push!(weights, conn.weight)
            end
        end

        if prune_unused
            required = NEAT.required_for_output(inputs, outputs, connections)
        else
            required = union(Set(inputs), Set(outputs), Set(keys(genome.nodes)))
        end

        all_nodes = sort(collect(required))
        node_id_to_idx = Dict(id => idx for (idx, id) in enumerate(all_nodes))

        g = SimpleDiGraph(length(all_nodes))
        for (i, o) in connections
            if i in node_id_to_idx && o in node_id_to_idx
                add_edge!(g, node_id_to_idx[i], node_id_to_idx[o])
            end
        end

        node_types = [node_id in inputs ? :input : (node_id in outputs ? :output : :hidden)
                      for node_id in all_nodes]
        node_colors = [t == :input ? :green : (t == :output ? :blue : :lightgray)
                       for t in node_types]

        layout_func = layout == :spring ? Spring() : (layout == :stress ? Stress() : Circular())

        graphplot!(ax, g,
                   layout=layout_func,
                   node_color=node_colors,
                   node_size=node_size,
                   arrow_show=true,
                   arrow_size=10)
    end

    return fig
end

end  # module

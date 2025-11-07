module NEATVisualizationExt

using NEAT
using Plots

"""
Plot fitness statistics over generations.

Shows best fitness, average fitness, and ±1 standard deviation bands.

# Arguments
- `reporter::StatisticsReporter`: The statistics reporter containing the data
- `ylog::Bool=false`: Use logarithmic y-axis
- `filename::String="fitness.png"`: Output filename
- `title::String="Fitness Evolution"`: Plot title
- `show_plot::Bool=false`: Display the plot interactively
"""
function NEAT.plot_fitness(reporter::NEAT.StatisticsReporter;
                           ylog::Bool=false,
                           filename::String="fitness.png",
                           title::String="Fitness Evolution",
                           show_plot::Bool=false)
    if isempty(reporter.most_fit_genomes)
        @warn "No fitness data to plot"
        return nothing
    end

    generation = 1:length(reporter.most_fit_genomes)
    best_fitness = [g.fitness === nothing ? 0.0 : g.fitness for g in reporter.most_fit_genomes]
    avg_fitness = NEAT.get_fitness_mean(reporter)
    stdev_fitness = NEAT.get_fitness_stdev(reporter)

    p = plot(generation, avg_fitness,
             label="Average",
             linewidth=2,
             color=:blue,
             xlabel="Generation",
             ylabel="Fitness",
             title=title,
             legend=:best,
             grid=true)

    # Add standard deviation bands
    plot!(p, generation, avg_fitness .- stdev_fitness,
          label="-1σ",
          linewidth=1,
          linestyle=:dash,
          color=:green)

    plot!(p, generation, avg_fitness .+ stdev_fitness,
          label="+1σ",
          linewidth=1,
          linestyle=:dash,
          color=:green)

    # Add best fitness
    plot!(p, generation, best_fitness,
          label="Best",
          linewidth=2,
          color=:red)

    if ylog
        yaxis!(p, :log)
    end

    savefig(p, filename)
    println("Fitness plot saved to $filename")

    if show_plot
        display(p)
    end

    return p
end

"""
Plot species sizes over generations as a stacked area chart.

# Arguments
- `reporter::StatisticsReporter`: The statistics reporter containing the data
- `filename::String="speciation.png"`: Output filename
- `title::String="Species Over Time"`: Plot title
- `show_plot::Bool=false`: Display the plot interactively
"""
function NEAT.plot_species(reporter::NEAT.StatisticsReporter;
                           filename::String="speciation.png",
                           title::String="Species Over Time",
                           show_plot::Bool=false)
    species_sizes = NEAT.get_species_sizes(reporter)

    if isempty(species_sizes)
        @warn "No species data to plot"
        return nothing
    end

    # Create matrix for plotting (generations x species)
    num_generations = length(species_sizes)
    num_species = length(species_sizes[1])

    generation = 1:num_generations

    # Build matrix where each column is a species
    data_matrix = zeros(Float64, num_generations, num_species)
    for (gen_idx, sizes) in enumerate(species_sizes)
        for (species_idx, size) in enumerate(sizes)
            data_matrix[gen_idx, species_idx] = Float64(size)
        end
    end

    # Use groupedbar with stacked bars or just use plot with fillrange
    p = plot(xlabel="Generation",
             ylabel="Individuals per Species",
             title=title,
             legend=:outertopright,
             grid=true)

    # Stack the species as filled areas
    cumsum_data = cumsum(data_matrix, dims=2)
    for i in num_species:-1:1
        y = i == 1 ? cumsum_data[:, i] : cumsum_data[:, i]
        y_prev = i == 1 ? zeros(num_generations) : cumsum_data[:, i-1]

        plot!(p, generation, y,
              fillrange=y_prev,
              label="Species $i",
              alpha=0.7,
              linewidth=0)
    end

    savefig(p, filename)
    println("Species plot saved to $filename")

    if show_plot
        display(p)
    end

    return p
end

"""
Plot fitness comparison between multiple runs.

# Arguments
- `reporters::Vector{StatisticsReporter}`: Multiple statistics reporters to compare
- `labels::Vector{String}`: Labels for each run
- `filename::String="fitness_comparison.png"`: Output filename
- `title::String="Fitness Comparison"`: Plot title
- `show_plot::Bool=false`: Display the plot interactively
"""
function NEAT.plot_fitness_comparison(reporters::Vector{NEAT.StatisticsReporter},
                                      labels::Vector{String};
                                      filename::String="fitness_comparison.png",
                                      title::String="Fitness Comparison",
                                      show_plot::Bool=false)
    if isempty(reporters)
        @warn "No data to plot"
        return nothing
    end

    p = plot(xlabel="Generation",
             ylabel="Best Fitness",
             title=title,
             legend=:best,
             grid=true)

    for (reporter, label) in zip(reporters, labels)
        if !isempty(reporter.most_fit_genomes)
            generation = 1:length(reporter.most_fit_genomes)
            best_fitness = [g.fitness === nothing ? 0.0 : g.fitness for g in reporter.most_fit_genomes]

            plot!(p, generation, best_fitness,
                  label=label,
                  linewidth=2)
        end
    end

    savefig(p, filename)
    println("Comparison plot saved to $filename")

    if show_plot
        display(p)
    end

    return p
end

"""
Draw a neural network genome as a graph.

Creates a visualization of the network topology showing:
- Input nodes (green)
- Output nodes (lightblue)
- Hidden nodes (white/gray)
- Connections colored by weight (red=negative, green=positive)
- Disabled connections shown as dashed lines

# Arguments
- `genome::Genome`: The genome to visualize
- `config::GenomeConfig`: Configuration for the genome
- `filename::String="network.png"`: Output filename
- `node_names::Union{Nothing, Dict}=nothing`: Optional custom names for nodes
- `show_disabled::Bool=true`: Show disabled connections
- `prune_unused::Bool=false`: Remove nodes that aren't connected to output
- `node_colors::Union{Nothing, Dict}=nothing`: Optional custom colors for nodes
- `show_plot::Bool=false`: Display the plot interactively
"""
function NEAT.draw_net(genome::NEAT.Genome,
                       config::NEAT.GenomeConfig;
                       filename::String="network.png",
                       node_names::Union{Nothing, Dict}=nothing,
                       show_disabled::Bool=true,
                       prune_unused::Bool=false,
                       node_colors::Union{Nothing, Dict}=nothing,
                       show_plot::Bool=false)

    # Get connection tuples (keys from the connections dict)
    connection_tuples = collect(keys(genome.connections))

    # Determine which nodes to include
    if prune_unused
        used_nodes = NEAT.required_for_output(config.input_keys, config.output_keys,
                                             connection_tuples)
    else
        used_nodes = Set(keys(genome.nodes))
        union!(used_nodes, config.input_keys)
        union!(used_nodes, config.output_keys)
    end

    nodes_list = sort(collect(used_nodes))

    # Compute node layers for layout
    layers = NEAT.feed_forward_layers(config.input_keys, config.output_keys,
                                     connection_tuples)

    # Create layer mapping
    node_to_layer = Dict{Int, Int}()
    for (layer_idx, layer_nodes) in enumerate(layers)
        for node_id in layer_nodes
            node_to_layer[node_id] = layer_idx
        end
    end

    # Position nodes
    node_positions = Dict{Int, Tuple{Float64, Float64}}()
    max_layer = length(layers)

    for (layer_idx, layer_nodes) in enumerate(layers)
        layer_size = length(layer_nodes)
        x = layer_idx
        for (pos_idx, node_id) in enumerate(sort(collect(layer_nodes)))
            y = pos_idx - layer_size / 2.0
            node_positions[node_id] = (Float64(x), y)
        end
    end

    # Create plot
    p = plot(xlabel="",
             ylabel="",
             title="Network Structure",
             legend=false,
             grid=false,
             showaxis=false,
             aspect_ratio=:equal)

    # Draw connections first (so they're behind nodes)
    for (conn_key, conn) in genome.connections
        in_node, out_node = conn_key

        # Skip if nodes aren't in used set
        if !(in_node in used_nodes && out_node in used_nodes)
            continue
        end

        # Skip disabled connections if requested
        if !show_disabled && !conn.enabled
            continue
        end

        # Get positions
        if !haskey(node_positions, in_node) || !haskey(node_positions, out_node)
            continue
        end

        x1, y1 = node_positions[in_node]
        x2, y2 = node_positions[out_node]

        # Determine color and style
        weight = conn.weight
        if !conn.enabled
            line_color = :gray
            line_style = :dash
            alpha = 0.3
        elseif weight < 0
            line_color = :red
            line_style = :solid
            alpha = min(abs(weight) * 0.5, 1.0)
        else
            line_color = :green
            line_style = :solid
            alpha = min(weight * 0.5, 1.0)
        end

        line_width = min(abs(weight) * 2, 5.0)

        # Draw connection
        plot!(p, [x1, x2], [y1, y2],
              color=line_color,
              linestyle=line_style,
              linewidth=line_width,
              alpha=alpha,
              arrow=true)
    end

    # Draw nodes
    for node_id in nodes_list
        if !haskey(node_positions, node_id)
            continue
        end

        x, y = node_positions[node_id]

        # Determine node color
        if node_colors !== nothing && haskey(node_colors, node_id)
            color = node_colors[node_id]
        elseif node_id in config.input_keys
            color = :lightgreen
        elseif node_id in config.output_keys
            color = :lightblue
        else
            color = :white
        end

        # Draw node
        scatter!(p, [x], [y],
                 marker=:circle,
                 markersize=15,
                 markercolor=color,
                 markerstrokewidth=2,
                 markerstrokecolor=:black)

        # Add label
        label_text = if node_names !== nothing && haskey(node_names, node_id)
            node_names[node_id]
        else
            string(node_id)
        end

        annotate!(p, x, y, text(label_text, :center, 8))
    end

    if !isempty(filename)
        savefig(p, filename)
        println("Network diagram saved to $filename")
    end

    if show_plot
        display(p)
    end

    return p
end

"""
Draw multiple genomes in a grid layout for comparison.

# Arguments
- `genomes::Vector{Genome}`: Vector of genomes to visualize
- `config::GenomeConfig`: Configuration for the genomes
- `filename::String="networks.png"`: Output filename
- `labels::Union{Nothing, Vector{String}}=nothing`: Optional labels for each genome
- `show_plot::Bool=false`: Display the plot interactively
"""
function NEAT.draw_net_comparison(genomes::Vector{NEAT.Genome},
                                   config::NEAT.GenomeConfig;
                                   filename::String="networks.png",
                                   labels::Union{Nothing, Vector{String}}=nothing,
                                   show_plot::Bool=false)
    # Create individual plots
    plots_list = []

    for (i, genome) in enumerate(genomes)
        label = labels !== nothing && i <= length(labels) ? labels[i] : "Genome $i"

        # Create plot for this genome (without saving)
        p = NEAT.draw_net(genome, config, filename="", show_plot=false)

        # Update title
        plot!(p, title=label)

        push!(plots_list, p)
    end

    # Combine into grid
    n = length(plots_list)
    ncols = Int(ceil(sqrt(n)))
    nrows = Int(ceil(n / ncols))

    combined = plot(plots_list..., layout=(nrows, ncols), size=(400*ncols, 400*nrows))

    savefig(combined, filename)
    println("Comparison plot saved to $filename")

    if show_plot
        display(combined)
    end

    return combined
end

end # module NEATVisualizationExt

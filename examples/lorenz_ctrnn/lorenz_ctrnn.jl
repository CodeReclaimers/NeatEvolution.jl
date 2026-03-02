"""
Lorenz Attractor CTRNN Prediction Example for NEAT.

Evolves a Continuous-Time Recurrent Neural Network to predict the next-step
state of the Lorenz attractor from its current state. The Lorenz system is a
classic chaotic dynamical system — small prediction errors compound rapidly,
making this a challenging benchmark for evolved networks.

The CTRNN receives normalized (x, y, z) as input and must produce the
normalized next-step (x, y, z) as output. Fitness is negative mean squared
error, so evolution drives fitness toward zero.

Usage:
    julia --project examples/lorenz_ctrnn/lorenz_ctrnn.jl
    julia --project examples/lorenz_ctrnn/lorenz_ctrnn.jl --products

The --products flag augments inputs with pairwise products (xy, xz, yz),
giving the network 6 inputs instead of 3. This tests whether z-prediction
failure is a representation problem: the z dynamics (dz/dt = xy - βz) require
a bilinear term that small networks with additive aggregation struggle to
discover on their own.
"""

using NeatEvolution
using TOML

# --- Lorenz system parameters ---
const LORENZ_SIGMA = 10.0
const LORENZ_RHO   = 28.0
const LORENZ_BETA  = 8.0 / 3.0

# --- Simulation parameters ---
const INTEGRATION_DT  = 0.01   # Lorenz integration timestep
const SUBSAMPLE       = 10     # take every Nth integration step as a data point
const DATA_DT         = INTEGRATION_DT * SUBSAMPLE  # effective dt between data points (0.1s)
const TOTAL_STEPS     = 11000  # integration steps: 1000 transient + 8000 train + 2000 test
const TRANSIENT_STEPS = 1000
const TRAIN_STEPS     = 8000
const TEST_STEPS      = 2000
const N_GENERATIONS   = 300
const PENALTY_FITNESS = -10.0  # assigned to NaN/Inf genomes

# --- Lorenz integration (RK4) ---

"""
    lorenz_rk4_step(x, y, z, dt) -> (x_new, y_new, z_new)

One step of 4th-order Runge-Kutta for the Lorenz system:
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz
"""
function lorenz_rk4_step(x, y, z, dt)
    # Lorenz derivatives
    f(x, y, z) = (LORENZ_SIGMA * (y - x),
                   x * (LORENZ_RHO - z) - y,
                   x * y - LORENZ_BETA * z)

    k1x, k1y, k1z = f(x, y, z)
    k2x, k2y, k2z = f(x + 0.5dt * k1x, y + 0.5dt * k1y, z + 0.5dt * k1z)
    k3x, k3y, k3z = f(x + 0.5dt * k2x, y + 0.5dt * k2y, z + 0.5dt * k2z)
    k4x, k4y, k4z = f(x + dt * k3x, y + dt * k3y, z + dt * k3z)

    x_new = x + (dt / 6.0) * (k1x + 2k2x + 2k3x + k4x)
    y_new = y + (dt / 6.0) * (k1y + 2k2y + 2k3y + k4y)
    z_new = z + (dt / 6.0) * (k1z + 2k2z + 2k3z + k4z)

    return (x_new, y_new, z_new)
end

"""
    generate_lorenz_trajectory(n_steps, dt) -> Matrix{Float64}

Integrate the Lorenz system from [1.0, 1.0, 1.0] for n_steps.
Returns a 3×n_steps matrix where rows are x, y, z.
"""
function generate_lorenz_trajectory(n_steps, dt)
    data = Matrix{Float64}(undef, 3, n_steps)
    x, y, z = 1.0, 1.0, 1.0
    for i in 1:n_steps
        x, y, z = lorenz_rk4_step(x, y, z, dt)
        data[1, i] = x
        data[2, i] = y
        data[3, i] = z
    end
    return data
end

# --- Normalization ---

struct NormParams
    min_vals::Vector{Float64}
    max_vals::Vector{Float64}
end

function compute_normalization(data::Matrix{Float64})
    min_vals = vec(minimum(data, dims=2))
    max_vals = vec(maximum(data, dims=2))
    return NormParams(min_vals, max_vals)
end

function normalize(data::Matrix{Float64}, p::NormParams)
    result = similar(data)
    for i in 1:3
        range = p.max_vals[i] - p.min_vals[i]
        if range == 0.0
            result[i, :] .= 0.0
        else
            result[i, :] .= 2.0 .* (data[i, :] .- p.min_vals[i]) ./ range .- 1.0
        end
    end
    return result
end

function denormalize(data::Matrix{Float64}, p::NormParams)
    result = similar(data)
    for i in 1:3
        range = p.max_vals[i] - p.min_vals[i]
        result[i, :] .= (data[i, :] .+ 1.0) ./ 2.0 .* range .+ p.min_vals[i]
    end
    return result
end

# --- Product-augmented inputs ---

"""
    augment_with_products(data::Matrix{Float64}) -> Matrix{Float64}

Append pairwise product rows (xy, xz, yz) to a 3×N matrix, returning 6×N.
Products of normalized [-1,1] values stay in [-1,1], so no extra normalization
is needed.
"""
function augment_with_products(data::Matrix{Float64})
    vcat(data,
         data[1:1, :] .* data[2:2, :],   # xy
         data[1:1, :] .* data[3:3, :],   # xz
         data[2:2, :] .* data[3:3, :])   # yz
end

# --- Data preparation ---

function prepare_data(; use_products::Bool=false)
    println("Generating Lorenz trajectory ($TOTAL_STEPS steps at dt=$INTEGRATION_DT, subsample=$(SUBSAMPLE)x)...")
    full_traj = generate_lorenz_trajectory(TOTAL_STEPS, INTEGRATION_DT)

    # Split: discard transient, then train, then test (subsample every Nth step)
    train_start = TRANSIENT_STEPS + 1
    train_end   = TRANSIENT_STEPS + TRAIN_STEPS
    test_start  = train_end + 1
    test_end    = train_end + TEST_STEPS

    train_raw = full_traj[:, train_start:SUBSAMPLE:train_end]
    test_raw  = full_traj[:, test_start:SUBSAMPLE:test_end]

    # Normalize based on training data only (always 3 variables for norm)
    norm_params = compute_normalization(train_raw)
    train_norm  = normalize(train_raw, norm_params)
    test_norm   = normalize(test_raw, norm_params)

    # Augment inputs with product terms if requested
    if use_products
        train_input = augment_with_products(train_norm)
        test_input  = augment_with_products(test_norm)
    else
        train_input = train_norm
        test_input  = test_norm
    end

    return train_input, test_input, norm_params, train_raw, test_raw
end

# --- Fitness evaluation ---

function make_eval_genomes(train_data::Matrix{Float64})
    n_steps = size(train_data, 2) - 1  # predict t+1 from t

    function eval_genomes(genomes, config)
        for (genome_id, genome) in genomes
            net = CTRNNNetwork(genome, config.genome_config)
            reset!(net)

            total_se = 0.0
            valid = true

            for t in 1:n_steps
                input  = train_data[:, t]

                output = advance!(net, input, DATA_DT, DATA_DT)

                if any(isnan, output) || any(isinf, output)
                    valid = false
                    break
                end

                # Target is always the x,y,z (first 3 rows) at next timestep
                for i in 1:3
                    total_se += (output[i] - train_data[i, t + 1])^2
                end
            end

            if valid
                genome.fitness = -total_se / (n_steps * 3)
            else
                genome.fitness = PENALTY_FITNESS
            end
        end
    end

    return eval_genomes
end

# --- Test evaluation ---

function evaluate_on_test(winner, config, test_data::Matrix{Float64})
    net = CTRNNNetwork(winner, config.genome_config)
    reset!(net)

    n_steps = size(test_data, 2) - 1
    predictions = Matrix{Float64}(undef, 3, n_steps)
    total_se = 0.0

    for t in 1:n_steps
        input  = test_data[:, t]

        output = advance!(net, input, DATA_DT, DATA_DT)
        predictions[:, t] .= output

        # Target is always the x,y,z (first 3 rows) at next timestep
        for i in 1:3
            total_se += (output[i] - test_data[i, t + 1])^2
        end
    end

    mse = total_se / (n_steps * 3)
    return mse, predictions
end

# --- Visualization (optional CairoMakie dependency) ---

function try_visualize(winner, config, test_norm, test_raw, norm_params)
    if Base.find_package("CairoMakie") === nothing
        println("\nVisualization skipped: CairoMakie not installed.")
        println("  To enable plots, run:")
        println("    julia --project -e 'using Pkg; Pkg.add(\"CairoMakie\")'")
        println("  Then re-run this script.")
        println("  (You can also use GLMakie for interactive plots.)")
        return
    end

    println("\nGenerating visualizations...")
    @eval using CairoMakie

    test_mse, pred_norm = evaluate_on_test(winner, config, test_norm)
    pred_raw = denormalize(pred_norm, norm_params)
    n_pred = size(pred_raw, 2)

    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)

    # Plot 1: 3D phase portrait
    Base.invokelatest() do
        fig = Figure(size=(1200, 900))
        ax = Axis3(fig[1, 1],
            title = "Lorenz Attractor: True vs CTRNN Predicted (test MSE=$(round(test_mse, digits=6)))",
            xlabel = "x", ylabel = "y", zlabel = "z")

        # True trajectory (use test_raw aligned with predictions: steps 2 through n_pred+1)
        true_data = test_raw[:, 2:(n_pred + 1)]
        lines!(ax, true_data[1, :], true_data[2, :], true_data[3, :],
            color=(:gray70, 0.4), linewidth=1, label="True")
        lines!(ax, pred_raw[1, :], pred_raw[2, :], pred_raw[3, :],
            color=:orange, linewidth=1.5, label="Predicted")
        axislegend(ax, position=:lt)

        path = joinpath(results_dir, "lorenz_phase_portrait.png")
        save(path, fig)
        println("  Saved: $path")
    end

    # Plot 2: Time series comparison
    Base.invokelatest() do
        fig = Figure(size=(1200, 600))
        time_axis = (1:n_pred) .* DATA_DT
        true_data = test_raw[:, 2:(n_pred + 1)]
        labels = ["x", "y", "z"]

        for (row, label) in enumerate(labels)
            ax = Axis(fig[row, 1],
                ylabel = label,
                xlabel = row == 3 ? "Time (s)" : "")
            lines!(ax, time_axis, true_data[row, :],
                color=:gray60, linewidth=1, label="True")
            lines!(ax, time_axis, pred_raw[row, :],
                color=:orange, linewidth=1.2, label="Predicted")
            if row == 1
                axislegend(ax, position=:rt)
            end
        end

        path = joinpath(results_dir, "lorenz_time_series.png")
        save(path, fig)
        println("  Saved: $path")
    end
end

# --- Config loading ---

"""
    load_lorenz_config(; num_inputs::Int=3) -> Config

Load config.toml, overriding num_inputs if product-augmented inputs are used.
"""
function load_lorenz_config(; num_inputs::Int=3)
    config_path = joinpath(@__DIR__, "config.toml")
    if num_inputs == 3
        return load_config(config_path)
    end
    # Override num_inputs in a temp copy (GenomeConfig is immutable)
    toml = TOML.parsefile(config_path)
    toml["DefaultGenome"]["num_inputs"] = num_inputs
    tmp = tempname() * ".toml"
    open(tmp, "w") do io
        TOML.print(io, toml)
    end
    config = load_config(tmp)
    rm(tmp)
    return config
end

# --- Main ---

function main()
    use_products = "--products" in ARGS
    start_time = time()

    num_inputs = use_products ? 6 : 3
    mode_str = use_products ? "augmented (x,y,z,xy,xz,yz)" : "standard (x,y,z)"

    println("=" ^ 60)
    println("Lorenz Attractor CTRNN Prediction")
    println("=" ^ 60)
    println("Mode: $mode_str — $num_inputs inputs, 3 outputs")
    println("Lorenz parameters: σ=$LORENZ_SIGMA, ρ=$LORENZ_RHO, β=$(round(LORENZ_BETA, digits=4))")
    println("Integration: $TOTAL_STEPS steps at dt=$INTEGRATION_DT, subsampled $(SUBSAMPLE)x (effective dt=$DATA_DT)")
    println("Data split: $(TRANSIENT_STEPS) transient + $(TRAIN_STEPS) train + $(TEST_STEPS) integration steps")
    println("Evolution: $N_GENERATIONS generations, pop_size=150")
    println()

    # Prepare data
    train_data, test_data, norm_params, train_raw, test_raw = prepare_data(use_products=use_products)
    println("Data shape: $(size(train_data, 1)) inputs × $(size(train_data, 2)) train points")
    println("Normalization ranges (from training data):")
    for (i, label) in enumerate(["x", "y", "z"])
        println("  $label: [$(round(norm_params.min_vals[i], digits=2)), $(round(norm_params.max_vals[i], digits=2))]")
    end
    println()

    # Load config and create population
    config = load_lorenz_config(num_inputs=num_inputs)
    pop = Population(config)
    add_reporter!(pop, StdOutReporter(true))

    # Evolve
    eval_fn = make_eval_genomes(train_data)
    winner = run!(pop, eval_fn, N_GENERATIONS)

    elapsed = time() - start_time
    println()
    println("=" ^ 60)
    println("Evolution complete")
    println("=" ^ 60)
    println("  Best fitness (train): $(round(winner.fitness, digits=6))")
    println("  Train MSE:            $(round(-winner.fitness, digits=6))")
    println("  Nodes:                $(length(winner.nodes))")
    println("  Connections:          $(length(winner.connections))")
    println("  Wall-clock time:      $(round(elapsed, digits=1))s")

    # Evaluate on test set
    test_mse, _ = evaluate_on_test(winner, config, test_data)
    println("  Test MSE:             $(round(test_mse, digits=6))")
    println()

    # Visualization (optional)
    try_visualize(winner, config, test_data, test_raw, norm_params)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

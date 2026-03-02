# Lorenz Attractor CTRNN Prediction

Evolves a Continuous-Time Recurrent Neural Network (CTRNN) using NEAT to predict the next-step state of the Lorenz attractor. The Lorenz system is a chaotic dynamical system where small errors grow exponentially, making it a challenging benchmark for evolved networks.

## Running

```bash
# Standard mode: 3 inputs (x, y, z)
julia --project examples/lorenz_ctrnn/lorenz_ctrnn.jl

# Augmented mode: 6 inputs (x, y, z, xy, xz, yz)
julia --project examples/lorenz_ctrnn/lorenz_ctrnn.jl --products
```

Runs 300 generations with a population of 150. Expect ~15-30 seconds depending on hardware.

## Product-augmented inputs

The `--products` flag adds pairwise product terms (xy, xz, yz) as additional network inputs. This tests whether the network's difficulty with certain variables is a representation problem: the Lorenz z equation (dz/dt = xy - βz) requires a bilinear term that small networks with additive aggregation struggle to discover on their own.

Results across multiple trials:

| Variable | Standard (3 inputs) | Augmented (6 inputs) |
|----------|---------------------|----------------------|
| x        | corr 0.90-0.96      | corr 0.89-0.96       |
| y        | corr 0.39-0.82      | corr 0.71-0.93       |
| z        | corr ~0.00          | corr ~0.07-0.13      |

The y variable improves consistently with products (the y equation dy/dt = x(ρ-z) - y contains the xz product). The z variable remains difficult in both modes — the product inputs are available but evolution doesn't reliably find the right wiring in 300 generations, likely because the fitness signal for z improvement is diluted across all three output variables.

## Visualization (optional)

If CairoMakie is installed, the script generates two PNG plots in `results/`:

- **lorenz_phase_portrait.png** — 3D phase portrait comparing true and predicted trajectories
- **lorenz_time_series.png** — Per-variable time series (x, y, z) on the test set

To install:

```bash
julia --project -e 'using Pkg; Pkg.add("CairoMakie")'
```

For interactive 3D plots, substitute GLMakie for CairoMakie in the script.

## How it works

The Lorenz system is defined by three coupled ODEs:

    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz

with standard parameters σ=10, ρ=28, β=8/3. The system is integrated with a hand-written 4th-order Runge-Kutta method (no external ODE solver dependency).

A trajectory of 11,000 steps (dt=0.01) is generated and subsampled every 10th step, giving an effective data timestep of 0.1s. The first 1,000 integration steps are discarded as transient. The next 8,000 form the training set (800 data points after subsampling), and the final 2,000 are held out for testing (200 data points).

Each CTRNN receives the normalized current state as input and predicts the normalized next-step state. Fitness is negative mean squared error, so evolution drives it toward zero. The CTRNN's continuous-time dynamics (per-node time constants) make it naturally suited to modeling temporal systems.

## Expected performance

After 300 generations with standard inputs, typical results are:

- **Overall MSE**: ~0.09-0.12 (in normalized [-1,1] space)
- **x variable**: correlation ~0.90-0.96 — well tracked
- **y variable**: correlation ~0.39-0.82 — partially learned
- **z variable**: correlation ~0.00 — typically not learned

The evolved CTRNNs (typically 3-10 nodes) learn the approximately linear x dynamics well but cannot discover a multiplication-like circuit for z. This is an honest result that demonstrates both the capability and limitations of small evolved CTRNNs on chaotic dynamical systems.

## Configuration highlights

- **3 inputs, 3 outputs** (or 6 inputs with `--products`)
- **No initial hidden nodes** — NEAT discovers the topology
- **feed_forward = false** — required for CTRNN recurrence
- **time_constant range [0.01, 5.0]** — covers fast reactions and slow integration
- **bias/weight range [-5, 5]** — narrower than default since the activation is tanh(2.5z)
- **max_stagnation = 30** — patient stagnation threshold for a harder task
- **10x subsampling** — ensures per-step state change is large enough that networks must learn dynamics, not just copy input

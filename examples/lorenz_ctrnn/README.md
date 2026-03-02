# Lorenz Attractor CTRNN Prediction

Evolves a Continuous-Time Recurrent Neural Network (CTRNN) using NEAT to predict the next-step state of the Lorenz attractor. The Lorenz system is a chaotic dynamical system where small errors grow exponentially, making it a challenging benchmark for evolved networks.

## Running

```bash
# Default: 3 inputs (x,y,z), 3 outputs (x,y,z)
julia --project examples/lorenz_ctrnn/lorenz_ctrnn.jl

# Add product inputs: 6 inputs (x,y,z,xy,xz,yz), 3 outputs
julia --project examples/lorenz_ctrnn/lorenz_ctrnn.jl --products

# Predict only z: 3 inputs, 1 output
julia --project examples/lorenz_ctrnn/lorenz_ctrnn.jl --z-only

# Both: 6 inputs, 1 output (best z prediction)
julia --project examples/lorenz_ctrnn/lorenz_ctrnn.jl --products --z-only
```

Runs 300 generations with a population of 150. Expect ~15-30 seconds depending on hardware.

## Experimental results

The four flag combinations reveal two independent factors limiting z prediction in the default mode:

### 3-output mode (default)

| Variable | Standard (3 inputs) | Augmented (6 inputs) |
|----------|---------------------|----------------------|
| x        | corr 0.90-0.96      | corr 0.89-0.96       |
| y        | corr 0.39-0.82      | corr 0.71-0.93       |
| z        | corr ~0.00          | corr ~0.07-0.13      |

In the default 3-output mode, **z is never learned**. Adding product inputs improves y (the y equation dy/dt = x(ρ-z) - y contains xz) but does not rescue z.

### z-only mode (`--z-only`)

| Inputs | z correlation | z MSE |
|--------|---------------|-------|
| Standard (3) | 0.54-0.85 | 0.06-0.15 |
| Augmented (6) | 0.94-0.96 | 0.015-0.023 |

With all fitness focused on z alone, the picture changes dramatically:

1. **Standard inputs, z-only**: correlation jumps from ~0.00 to 0.54-0.85. This proves that **fitness dilution** was the primary bottleneck — when z competed with the easier x and y outputs for fitness signal, evolution never bothered improving it.

2. **Augmented inputs, z-only**: correlation reaches 0.94-0.96 with tiny networks (often 1 hidden node, 4 connections). When xy is provided directly and evolution focuses exclusively on z, the task becomes easy.

### Summary of limiting factors

The z-prediction failure in the default mode has **two independent causes**:

- **Fitness dilution** (larger effect): With 3 outputs, easy gains on x and y dominate the fitness landscape. Evolution has no incentive to improve z when the same mutation effort yields larger MSE reductions on the other variables. Removing x and y from the fitness function (via `--z-only`) fixes this.

- **Representation difficulty** (smaller but real effect): The z equation dz/dt = xy - βz requires a bilinear term. Small networks with additive aggregation struggle to represent multiplication. Providing product inputs (via `--products`) fixes this. The effect is visible in both z-only mode (corr 0.54-0.85 → 0.94-0.96) and 3-output mode (y improves consistently).

## Visualization (optional)

If CairoMakie is installed, the script generates PNG plots in `results/`:

- **lorenz_phase_portrait.png** — 3D phase portrait (3-output mode only)
- **lorenz_time_series.png** — Per-variable time series on the test set

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

Each CTRNN receives the normalized current state as input and predicts the normalized next-step state (or just z). Fitness is negative mean squared error, so evolution drives it toward zero. The CTRNN's continuous-time dynamics (per-node time constants) make it naturally suited to modeling temporal systems.

## Configuration highlights

- **3 or 6 inputs** (with `--products`), **3 or 1 outputs** (with `--z-only`)
- **No initial hidden nodes** — NEAT discovers the topology
- **feed_forward = false** — required for CTRNN recurrence
- **time_constant range [0.01, 5.0]** — covers fast reactions and slow integration
- **bias/weight range [-5, 5]** — narrower than default since the activation is tanh(2.5z)
- **max_stagnation = 30** — patient stagnation threshold for a harder task
- **10x subsampling** — ensures per-step state change is large enough that networks must learn dynamics, not just copy input

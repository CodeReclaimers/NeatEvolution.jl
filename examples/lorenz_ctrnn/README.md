# Lorenz Attractor CTRNN Prediction

Evolves a Continuous-Time Recurrent Neural Network (CTRNN) using NEAT to predict the next-step state of the Lorenz attractor. The Lorenz system is a chaotic dynamical system where small errors grow exponentially, making it a challenging benchmark for evolved networks.

## Running

```bash
# Default: 3 inputs (x,y,z), 3 outputs (x,y,z), sum aggregation
julia --project examples/lorenz_ctrnn/lorenz_ctrnn.jl

# Pre-computed product inputs: 6 inputs (x,y,z,xy,xz,yz)
julia --project examples/lorenz_ctrnn/lorenz_ctrnn.jl --mode products

# Product aggregation: 3 inputs, but nodes can use * instead of +
julia --project examples/lorenz_ctrnn/lorenz_ctrnn.jl --mode product-agg

# Any mode can be combined with --z-only to predict only z
julia --project examples/lorenz_ctrnn/lorenz_ctrnn.jl --mode product-agg --z-only
```

Runs 300 generations with a population of 150. Wall-clock times on a single core range from ~9s (z-only) to ~20s (3-output) depending on mode and network complexity.

## Input modes

| Mode | Inputs | Description |
|------|--------|-------------|
| `base` (default) | x, y, z | Standard 3-variable state |
| `products` | x, y, z, xy, xz, yz | Pre-computed pairwise products |
| `product-agg` | x, y, z | Same inputs as base, but `product` added as an aggregation function alongside `sum`, so evolution can discover multiplicative nodes |

The three modes test different hypotheses about why z prediction fails: is the problem that the network lacks access to product terms (`products` mode), or that it lacks the ability to represent multiplication internally (`product-agg` mode)?

## Experimental results

All results are from 300 generations, population 150, 10x subsampled Lorenz trajectory (800 train / 200 test points, effective dt=0.1s). Wall-clock times are on a single core (Intel i7).

### 3-output mode (predicting x, y, z)

| Mode | x corr | y corr | z corr | MSE | Nodes | Time |
|------|--------|--------|--------|-----|-------|------|
| base | 0.84-0.96 | -0.09-0.82 | 0.00-0.17 | 0.10-0.12 | 4-10 | 12-18s |
| products | 0.93-0.97 | -0.17-0.66 | 0.00-0.12 | 0.11-0.12 | 6-17 | 13-20s |
| product-agg | 0.92-0.96 | 0.09-0.65 | 0.00-0.15 | 0.10-0.12 | 4-9 | 14-15s |

In 3-output mode, z is never meaningfully learned regardless of input representation. The fitness signal is dominated by easy x/y improvements.

### z-only mode (`--z-only`)

| Mode | z corr | z MSE | Nodes | Time |
|------|--------|-------|-------|------|
| base | 0.83-0.86 | 0.055-0.069 | 2-3 | 9-10s |
| products | 0.94 | 0.023 | 1-3 | 10-12s |
| product-agg | 0.94 | 0.023-0.024 | 1-6 | 9-10s |

With all fitness focused on z alone:

1. **base**: z correlation jumps from ~0.00 to 0.83-0.86. This proves **fitness dilution** is the primary bottleneck — not representation.

2. **products**: z correlation reaches 0.94 with tiny networks (often 1 node, 4 connections). Pre-computing xy lets the network trivially learn the z dynamics.

3. **product-agg**: z correlation also reaches 0.94 — matching `products` exactly. Evolution discovers multiplicative nodes when the aggregation function is available, without needing pre-computed product inputs. Networks are similarly tiny (1-6 nodes).

### Summary

The z-prediction failure in the default configuration has **two independent causes**:

- **Fitness dilution (primary):** With 3 outputs, easy gains on x and y dominate. Switching to z-only raises z correlation from ~0.00 to 0.83-0.86 even with standard inputs.

- **Representation difficulty (secondary):** The z equation dz/dt = xy - βz requires a bilinear term. Either pre-computing products (`products` mode) or enabling multiplicative aggregation (`product-agg` mode) raises z correlation from 0.83-0.86 to 0.94 in z-only mode.

The `product-agg` mode is the most interesting result: it demonstrates that NEAT can discover the right computational primitive (multiplication) when it's available in the search space, achieving the same performance as hand-engineered product inputs.

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

- **3 or 6 inputs** (with `--mode products`), **3 or 1 outputs** (with `--z-only`)
- **No initial hidden nodes** — NEAT discovers the topology
- **feed_forward = false** — required for CTRNN recurrence
- **time_constant range [0.01, 5.0]** — covers fast reactions and slow integration
- **bias/weight range [-5, 5]** — narrower than default since the activation is tanh(2.5z)
- **max_stagnation = 30** — patient stagnation threshold for a harder task
- **10x subsampling** — ensures per-step state change is large enough that networks must learn dynamics, not just copy input
- **product-agg mode** — adds `product` to aggregation options with 10% mutation rate

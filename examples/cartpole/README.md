# CartPole Example

Evolve a neural network to balance an inverted pendulum on a cart using NEAT.

This example uses a **pure-Julia CartPole simulation** — no Python, Gymnasium, or external dependencies required. The physics match OpenAI Gymnasium's CartPole-v1 environment for reproducibility.

## The Problem

A pole is attached by a hinge to a cart that moves along a track. The controller applies a force of +10 or -10 Newtons to the cart at each time step. The goal is to keep the pole balanced (within 12 degrees of vertical) and the cart on the track (within 2.4 meters of center) for 500 time steps.

**State** (4 inputs): cart position, cart velocity, pole angle, pole angular velocity

**Action** (1 output): push left (output < 0.5) or push right (output >= 0.5)

**Fitness**: average reward over 5 episodes (max 500 per episode)

## Running

From the repository root:

```bash
julia --project examples/cartpole/evolve.jl
```

A solution is typically found within 10-30 generations.

## Files

- `cartpole.jl` — CartPole physics simulation
- `evolve.jl` — NEAT evolution script
- `config.toml` — NEAT configuration (4 inputs, 1 output, feed-forward)

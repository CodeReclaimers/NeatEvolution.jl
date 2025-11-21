# Inverted Pendulum (CartPole) Example

This example demonstrates using NEAT to evolve a neural network controller for the classic CartPole-v1 control problem from Gymnasium (formerly OpenAI Gym).

## Problem Description

The **CartPole** or **Inverted Pendulum** problem involves balancing a pole attached to a cart that moves along a frictionless track. The controller must learn to keep the pole upright by applying forces to move the cart left or right.

### Environment Details

- **Observation Space**: 4 continuous values
  - Cart Position: -4.8 to 4.8
  - Cart Velocity: -∞ to ∞
  - Pole Angle: -0.418 to 0.418 radians (~24°)
  - Pole Angular Velocity: -∞ to ∞

- **Action Space**: 2 discrete actions
  - 0: Push cart to the left
  - 1: Push cart to the right

- **Rewards**: +1 for each timestep the pole remains upright

- **Episode Termination**:
  - Pole angle exceeds ±12 degrees
  - Cart position exceeds ±2.4 units from center
  - Episode length reaches 500 timesteps

- **Success Criteria**: Average reward of 475+ over 100 consecutive episodes

## Setup

### 1. Install Julia Dependencies

```bash
# Install PyCall for Python interop
julia -e 'using Pkg; Pkg.add("PyCall")'
```

### 2. Install Python Dependencies

You need Python with the Gymnasium library:

```bash
# Using pip
pip install gymnasium

# Or using conda
conda install -c conda-forge gymnasium
```

### 3. (Optional) Install Visualization Dependencies

To visualize the pole balancing:

```bash
# Install pygame for rendering
pip install gymnasium[classic-control]
```

## Running the Example

```bash
# From the repository root
julia --project examples/inverted_pendulum/evolve.jl
```

The evolution will run for up to 100 generations or until a solution is found (fitness ≥ 500).

## Configuration

The `config.toml` file contains all NEAT parameters:

- **Population size**: 150 genomes
- **Network structure**: 4 inputs → hidden nodes (evolved) → 1 output
- **Activation functions**: tanh (default), with mutation to sigmoid, relu, or identity
- **Fitness criterion**: maximize average episode reward
- **Success threshold**: 500.0 (perfect score)

### Key Parameters

- `initial_connection = "full"`: Start with fully connected networks
- `feed_forward = true`: Networks are feed-forward (no recurrence)
- `node_add_prob = 0.3`: Probability of adding hidden nodes
- `conn_add_prob = 0.5`: Probability of adding connections

## How It Works

1. **Initialization**: Population of 150 random neural networks is created
2. **Evaluation**: Each network controls the cart for 3 episodes, fitness = average reward
3. **Selection**: Best-performing networks are selected for reproduction
4. **Speciation**: Networks are grouped by structural similarity
5. **Reproduction**: Selected networks produce offspring through mutation and crossover
6. **Repeat**: Process continues until solution is found or generation limit is reached

### Network Architecture

The evolved networks map observations directly to actions:

```
[cart_pos, cart_vel, pole_angle, pole_vel]
    → (evolved hidden layers)
    → [output]
    → action = output > 0.0 ? push_right : push_left
```

## Expected Results

- **Early generations**: Networks typically achieve 20-50 reward (pole falls quickly)
- **Mid evolution**: Networks learn to balance for 100-200 timesteps
- **Solution**: Networks can achieve 500 reward (perfect balance for maximum episode length)

Typical evolution time: 10-50 generations (varies with random seed)

## Troubleshooting

### PyCall Issues

If PyCall can't find Gymnasium:

```julia
# Configure PyCall to use specific Python
ENV["PYTHON"] = "/path/to/python"
using Pkg
Pkg.build("PyCall")
```

### Visualization Not Working

If visualization fails:
- Ensure you have a display available (won't work over SSH without X11 forwarding)
- Install pygame: `pip install pygame`
- Try running with `DISPLAY=:0` on Linux

### Slow Evolution

If evolution is too slow:
- Reduce population size in config.toml (try 50-100)
- Reduce number of evaluation episodes in `eval_genomes` (try 1-2)
- The trade-off is less stable fitness estimates

## Extending This Example

### Try Different Environments

Replace `"CartPole-v1"` with other Gymnasium environments:
- `"Acrobot-v1"`: Swing up a two-link robot arm
- `"MountainCar-v0"`: Drive a car up a hill (continuous control)
- `"LunarLander-v2"`: Land a spacecraft

Note: You'll need to adjust `num_inputs` and `num_outputs` in config.toml for different environments.

### Add Recurrent Connections

Set `feed_forward = false` in config.toml to allow recurrent networks that can maintain state across timesteps.

### Improve Evaluation

- Increase episodes in `eval_genomes` for more stable fitness
- Add noise to observations to test robustness
- Vary initial conditions to ensure general solutions

## References

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [CartPole-v1 Environment](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
- [NEAT Paper](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)

"""
Pure-Julia CartPole environment.

Physics and parameters match OpenAI Gymnasium's CartPole-v1 for reproducibility.

State: [x, x_dot, theta, theta_dot]
  x     - cart position (m)
  x_dot - cart velocity (m/s)
  theta  - pole angle from vertical (rad)
  theta_dot - pole angular velocity (rad/s)

Action: 0 (push left) or 1 (push right)
"""

using Random

const GRAVITY = 9.8
const CART_MASS = 1.0
const POLE_MASS = 0.1
const TOTAL_MASS = CART_MASS + POLE_MASS
const POLE_HALF_LENGTH = 0.5
const FORCE_MAG = 10.0
const TAU = 0.02           # integration time step (seconds)
const MAX_STEPS = 500
const X_THRESHOLD = 2.4
const THETA_THRESHOLD = 12.0 * pi / 180.0  # 12 degrees in radians

mutable struct CartPoleEnv
    x::Float64
    x_dot::Float64
    theta::Float64
    theta_dot::Float64
    steps::Int
    done::Bool
end

CartPoleEnv() = CartPoleEnv(0.0, 0.0, 0.0, 0.0, 0, false)

"""
    reset!(env::CartPoleEnv; rng=Random.GLOBAL_RNG)

Reset the environment to a random initial state in [-0.05, 0.05].
Returns the observation vector.
"""
function reset!(env::CartPoleEnv; rng::AbstractRNG=Random.GLOBAL_RNG)
    env.x         = rand(rng) * 0.1 - 0.05
    env.x_dot     = rand(rng) * 0.1 - 0.05
    env.theta     = rand(rng) * 0.1 - 0.05
    env.theta_dot = rand(rng) * 0.1 - 0.05
    env.steps = 0
    env.done = false
    return observe(env)
end

"""Return the current observation as a 4-element vector."""
function observe(env::CartPoleEnv)
    return [env.x, env.x_dot, env.theta, env.theta_dot]
end

"""
    step!(env::CartPoleEnv, action::Int) -> (observation, reward, done)

Advance the simulation by one time step. `action` is 0 (left) or 1 (right).
Uses semi-implicit Euler integration matching Gymnasium.
"""
function step!(env::CartPoleEnv, action::Int)
    if env.done
        error("Episode is done. Call reset! before stepping.")
    end

    force = action == 1 ? FORCE_MAG : -FORCE_MAG

    costheta = cos(env.theta)
    sintheta = sin(env.theta)

    # Equations of motion (from Gymnasium cart_pole.py)
    temp = (force + POLE_MASS * POLE_HALF_LENGTH * env.theta_dot^2 * sintheta) / TOTAL_MASS
    theta_acc = (GRAVITY * sintheta - costheta * temp) /
                (POLE_HALF_LENGTH * (4.0/3.0 - POLE_MASS * costheta^2 / TOTAL_MASS))
    x_acc = temp - POLE_MASS * POLE_HALF_LENGTH * theta_acc * costheta / TOTAL_MASS

    # Semi-implicit Euler integration
    env.x_dot     += TAU * x_acc
    env.x         += TAU * env.x_dot
    env.theta_dot += TAU * theta_acc
    env.theta     += TAU * env.theta_dot

    env.steps += 1

    # Termination conditions
    env.done = abs(env.x) > X_THRESHOLD ||
               abs(env.theta) > THETA_THRESHOLD ||
               env.steps >= MAX_STEPS

    reward = env.done && env.steps < MAX_STEPS ? 0.0 : 1.0
    return (observe(env), reward, env.done)
end

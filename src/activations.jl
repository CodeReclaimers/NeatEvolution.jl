"""
Activation functions for NEAT nodes.
"""

# Activation functions
function sigmoid_activation(z::Float64)
    z = clamp(5.0 * z, -60.0, 60.0)
    return 1.0 / (1.0 + exp(-z))
end

function tanh_activation(z::Float64)
    z = clamp(2.5 * z, -60.0, 60.0)
    return tanh(z)
end

function sin_activation(z::Float64)
    z = clamp(5.0 * z, -60.0, 60.0)
    return sin(z)
end

function gauss_activation(z::Float64)
    z = clamp(z, -3.4, 3.4)
    return exp(-5.0 * z^2)
end

function relu_activation(z::Float64)
    return max(0.0, z)
end

function elu_activation(z::Float64)
    return z > 0.0 ? z : exp(z) - 1.0
end

function lelu_activation(z::Float64)
    leaky = 0.005
    return z > 0.0 ? z : leaky * z
end

function selu_activation(z::Float64)
    lam = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return z > 0.0 ? lam * z : lam * alpha * (exp(z) - 1.0)
end

function softplus_activation(z::Float64)
    z = clamp(5.0 * z, -60.0, 60.0)
    return 0.2 * log(1.0 + exp(z))
end

function identity_activation(z::Float64)
    return z
end

function clamped_activation(z::Float64)
    return clamp(z, -1.0, 1.0)
end

function inv_activation(z::Float64)
    return z == 0.0 ? 0.0 : 1.0 / z
end

function log_activation(z::Float64)
    z = max(1e-7, z)
    return log(z)
end

function exp_activation(z::Float64)
    z = clamp(z, -60.0, 60.0)
    return exp(z)
end

function abs_activation(z::Float64)
    return abs(z)
end

function hat_activation(z::Float64)
    return max(0.0, 1.0 - abs(z))
end

function square_activation(z::Float64)
    return z^2
end

function cube_activation(z::Float64)
    return z^3
end

# Activation function registry
const ACTIVATION_FUNCTIONS = Dict{Symbol, Function}(
    :sigmoid => sigmoid_activation,
    :tanh => tanh_activation,
    :sin => sin_activation,
    :gauss => gauss_activation,
    :relu => relu_activation,
    :elu => elu_activation,
    :lelu => lelu_activation,
    :selu => selu_activation,
    :softplus => softplus_activation,
    :identity => identity_activation,
    :clamped => clamped_activation,
    :inv => inv_activation,
    :log => log_activation,
    :exp => exp_activation,
    :abs => abs_activation,
    :hat => hat_activation,
    :square => square_activation,
    :cube => cube_activation
)

function get_activation_function(name::Symbol)
    if haskey(ACTIVATION_FUNCTIONS, name)
        return ACTIVATION_FUNCTIONS[name]
    else
        error("Unknown activation function: $name")
    end
end

function add_activation_function!(name::Symbol, func::Function)
    ACTIVATION_FUNCTIONS[name] = func
end

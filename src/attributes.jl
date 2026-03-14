"""
Attribute system for gene parameters.

Handles initialization, mutation, and validation of gene attributes
like weights, biases, activation functions, etc.
"""

using Random

# Abstract base type for attributes
abstract type BaseAttribute end

"""
FloatAttribute handles floating-point numeric attributes like weights and biases.
"""
struct FloatAttribute <: BaseAttribute
    name::Symbol
    init_mean::Float64
    init_stdev::Float64
    init_type::Symbol  # :gaussian or :uniform
    replace_rate::Float64
    mutate_rate::Float64
    mutate_power::Float64
    min_value::Float64
    max_value::Float64
end

function FloatAttribute(name::Symbol, config::Dict)
    prefix = string(name)
    FloatAttribute(
        name,
        get(config, Symbol(prefix * "_init_mean"), 0.0),
        get(config, Symbol(prefix * "_init_stdev"), 1.0),
        Symbol(lowercase(get(config, Symbol(prefix * "_init_type"), "gaussian"))),
        get(config, Symbol(prefix * "_replace_rate"), 0.0),
        get(config, Symbol(prefix * "_mutate_rate"), 0.0),
        get(config, Symbol(prefix * "_mutate_power"), 0.0),
        get(config, Symbol(prefix * "_min_value"), -30.0),
        get(config, Symbol(prefix * "_max_value"), 30.0)
    )
end

function clamp_value(attr::FloatAttribute, value::Float64)
    return clamp(value, attr.min_value, attr.max_value)
end

function init_value(attr::FloatAttribute, rng::AbstractRNG=Random.GLOBAL_RNG)
    if attr.init_type == :gaussian || attr.init_type == :normal
        val = randn(rng) * attr.init_stdev + attr.init_mean
        return clamp_value(attr, val)
    elseif attr.init_type == :uniform
        min_val = max(attr.min_value, attr.init_mean - 2 * attr.init_stdev)
        max_val = min(attr.max_value, attr.init_mean + 2 * attr.init_stdev)
        return rand(rng) * (max_val - min_val) + min_val
    else
        error("Unknown init_type: $(attr.init_type)")
    end
end

function mutate_value(attr::FloatAttribute, value::Float64, rng::AbstractRNG=Random.GLOBAL_RNG)
    r = rand(rng)

    if r < attr.mutate_rate
        # Mutate by adding Gaussian noise
        new_val = value + randn(rng) * attr.mutate_power
        return clamp_value(attr, new_val)
    elseif r < attr.mutate_rate + attr.replace_rate
        # Replace with new random value
        return init_value(attr, rng)
    end

    return value
end

function validate(attr::FloatAttribute)
    if attr.max_value < attr.min_value
        error("Invalid min/max configuration for $(attr.name)")
    end
end

"""
BoolAttribute handles boolean attributes like whether a connection is enabled.
"""
struct BoolAttribute <: BaseAttribute
    name::Symbol
    default::Symbol  # Symbol("true"), Symbol("false"), or Symbol("random")
    mutate_rate::Float64
    rate_to_true_add::Float64
    rate_to_false_add::Float64
end

function BoolAttribute(name::Symbol, config::Dict)
    prefix = string(name)
    default_val = get(config, Symbol(prefix * "_default"), "true")

    # Handle Bool values from TOML by explicitly converting first
    if default_val isa Bool
        default_val = default_val ? "true" : "false"
    end

    # Convert to string and parse to Symbol
    default_str = lowercase(string(default_val))

    # Note: :true and :false are Bool literals in Julia, not Symbols.
    # We must use Symbol("true") etc. to get actual Symbol values.
    if default_str == "true" || default_str == "1" || default_str == "on" || default_str == "yes"
        default_sym = Symbol("true")
    elseif default_str == "false" || default_str == "0" || default_str == "off" || default_str == "no"
        default_sym = Symbol("false")
    elseif default_str == "random" || default_str == "none"
        default_sym = Symbol("random")
    else
        error("Unknown default value '$default_str' for attribute $name")
    end

    # Construct and return
    return BoolAttribute(
        name,
        default_sym,
        Float64(get(config, Symbol(prefix * "_mutate_rate"), 0.0)),
        Float64(get(config, Symbol(prefix * "_rate_to_true_add"), 0.0)),
        Float64(get(config, Symbol(prefix * "_rate_to_false_add"), 0.0))
    )
end

function init_value(attr::BoolAttribute, rng::AbstractRNG=Random.GLOBAL_RNG)
    if attr.default == Symbol("true")
        return true
    elseif attr.default == Symbol("false")
        return false
    else  # Symbol("random")
        return rand(rng, Bool)
    end
end

function mutate_value(attr::BoolAttribute, value::Bool, rng::AbstractRNG=Random.GLOBAL_RNG)
    rate = attr.mutate_rate
    if value
        rate += attr.rate_to_false_add
    else
        rate += attr.rate_to_true_add
    end

    if rate > 0 && rand(rng) < rate
        return !value
    end

    return value
end

"""
StringAttribute handles string attributes like activation and aggregation functions.
"""
struct StringAttribute <: BaseAttribute
    name::Symbol
    default::Union{Symbol, Nothing}
    options::Vector{Symbol}
    mutate_rate::Float64
end

function StringAttribute(name::Symbol, config::Dict)
    prefix = string(name)
    default_str = lowercase(string(get(config, Symbol(prefix * "_default"), "random")))
    default_sym = if default_str in ["none", "random"]
        nothing
    else
        Symbol(default_str)
    end

    options_val = get(config, Symbol(prefix * "_options"), Symbol[])
    if options_val isa String
        options = Symbol[Symbol(strip(s)) for s in split(options_val)]
    elseif options_val isa Vector
        options = Symbol[Symbol(o) for o in options_val]
    else
        options = Symbol[]
    end

    StringAttribute(
        name,
        default_sym,
        options,
        get(config, Symbol(prefix * "_mutate_rate"), 0.0)
    )
end

function init_value(attr::StringAttribute, rng::AbstractRNG=Random.GLOBAL_RNG)
    if attr.default === nothing
        return rand(rng, attr.options)
    else
        return attr.default
    end
end

function mutate_value(attr::StringAttribute, value::Symbol, rng::AbstractRNG=Random.GLOBAL_RNG)
    if attr.mutate_rate > 0 && rand(rng) < attr.mutate_rate
        return rand(rng, attr.options)
    end
    return value
end

function validate(attr::StringAttribute)
    if attr.default !== nothing && !(attr.default in attr.options)
        error("Invalid initial value $(attr.default) for $(attr.name)")
    end
end

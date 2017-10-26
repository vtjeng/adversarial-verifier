if !(pwd() in LOAD_PATH)
    push!(LOAD_PATH, pwd())
end

module NNOps

using Base.Cartesian
using JuMP
using ConditionalJuMP
using Gurobi

using NNParameters

JuMPReal = Union{Real, JuMP.AbstractJuMPScalar}

"""
For a convolution of `filter` on `input`, determines the size of the output.

# Throws
* ArgumentError if input and filter are not compatible.
"""
function getconv2doutputsize{T}(
    input::Array{T, 4},
    params::Conv2DParameters)::NTuple{4, Int}
    (batch, in_height, in_width, input_in_channels) = size(input)
    (filter_height, filter_width, filter_in_channels, out_channels) = size(params.filter)
    if input_in_channels != filter_in_channels
        throw(ArgumentError("Number of channels in input, $input_in_channels, does not match number of channels, $filter_in_channels, that filters operate on."))
    end
    return (batch, in_height, in_width, out_channels)
end

# TODO: Really test this conv2d logic!!!
"""
Computes a 2D-convolution given 4-D `input` and `filter` tensors.

Mirrors `tf.nn.conv2d` from `tensorflow` package, with `strides` = [1, 1, 1, 1],
 `padding` = 'SAME'.
"""
function conv2d{T<:Union{Real, JuMP.AffExpr, JuMP.Variable}, U<:Real, V<:Real}(
    input::Array{T, 4},
    params::Conv2DParameters{U, V})
    # TEST PLAN:
    #  (1) Incorrectly sized input,
    #  (2) Incorrectly sized filter,
    #  (3) Non-matching elements of array
    #  (4) Non-matching input_in_channels and filter_in_channels
    filter = params.filter

    (batch, in_height, in_width, input_in_channels) = size(input)
    (filter_height, filter_width, filter_in_channels, filter_out_channels) = size(filter)

    output_size = getconv2doutputsize(input, params)

    # Considered using offset arrays here, but looks like it currently is not
    # really supported

    # Calculating appropriate offsets so that center of kernel is matched with
    # cell at which correlation is being calculated. Note that tensorflow
    # chooses a specific convention for a dimension with even size which we
    # replicate here.
    filter_height_offset = round(Int, filter_height/2, RoundUp)
    filter_width_offset = round(Int, filter_width/2, RoundUp)
    W = Base.promote_op(+, V, Base.promote_op(*, T, U))
    output = Array{W}(output_size)

    @nloops 4 i output begin
        s::W = 0
        @nloops 4 j filter begin
            if i_4 == j_4
                x = i_2 + j_1 - filter_height_offset
                y = i_3 + j_2 - filter_width_offset
                if x > 0 && y > 0 && x<=in_height && y<=in_width
                    # Doing bounds check to make sure that we stay within bounds
                    # for input. This effectively zero-pads the input.
                    # TODO: Use default checkbounds function here instead?
                    # TODO: Addition here is a bottleneck; figure out whether
                    # you could use append without making this incompatible
                    # with normal numbers
                    s = increment!(s, input[i_1, x, y, j_3], filter[j_1, j_2, j_3, j_4])
                end
            end
        end
        s += params.bias[i_4]
        (@nref 4 output i) = s
    end

    return output
end

function increment!{S<:Real, T<:Real, U<:Real}(s::S, input_val::T, filter_val::U)
    return s + input_val*filter_val
end

function increment!{T<:JuMP.AffExpr, U<:Real}(s::JuMP.AffExpr, input_val::T, filter_val::U)
    append!(s, input_val, filter_val)
    return s
end

function increment!{T<:JuMP.Variable, U<:Real}(s::JuMP.AffExpr, input_val::T, filter_val::U)
    push!(s, Float64(filter_val), input_val)
end

"""
For pooling operations on an array where a given element in the output array
corresponds to equal-sized blocks in the input array, returns (for a given
dimension) the index range in the input array corresponding to a particular
index `output_index` in the output array.

Returns an empty array if the `output_index` does not correspond to any input
indices.

# Arguments
* `stride::Integer`: the size of the operating blocks along the active
     dimension.

"""
function getsliceindex(input_array_size::Int, stride::Int, output_index::Int)::Array{Int, 1}
    parent_start_index = (output_index-1)*stride+1
    parent_end_index = min((output_index)*stride, input_array_size)
    if parent_start_index > parent_end_index
        return []
    else
        return parent_start_index:parent_end_index
    end
end

"""
For pooling operations on an array, returns a view of the parent array
corresponding to the `output_index` in the output array.
"""
function getpoolview{T, N}(input_array::Array{T, N}, strides::NTuple{N, Int}, output_index::NTuple{N, Int})::SubArray{T, N}
    it = zip(size(input_array), strides, output_index)
    input_index_range = map((x)-> getsliceindex(x...), it)
    return view(input_array, input_index_range...)
end

"""
For pooling operations on an array, returns the expected size of the output
array.
"""
function getoutputsize{T, N}(input_array::Array{T, N}, strides::NTuple{N, Int})::NTuple{N, Int}
    output_size = ((x, y) -> round(Int, x/y, RoundUp)).(size(input_array), strides)
    return output_size
end

"""
Returns output from applying `f` to subarrays of `input_array`, with the windows
determined by the `strides`.
"""
function poolmap{T, N}(f::Function, input_array::Array{T, N}, strides::NTuple{N, Int})
    output_size = getoutputsize(input_array, strides)
    output_indices = collect(CartesianRange(output_size))
    return ((I) -> f(getpoolview(input_array, strides, I.I))).(output_indices)
end

function avgpool{T<:Real, N}(
    input::Array{T, N},
    params::PoolParameters{N})::Array{T, N}
    return poolmap(mean, input, params.strides)
end

"""
Computes the result of a max-pooling operation on `input_array` with specified
`strides`.
"""
function maxpool{T<:Real, N}(
    input::Array{T, N},
    params::PoolParameters{N})::Array{T, N}
    # NB: Tried to use pooling function from Knet.relu but it had way too many
    # incompatibilities
    return poolmap(Base.maximum, input, params.strides)
end

"""
Imposes a max-pooling constraint between `x` and `x_pooled` using the big-M
formulation.

`x` is divided into cells of size `strides`, and each entry of `x_pooled`
is equal to the maximum value in the corresponding cell.

If the height (viz. width) of the input array `x` is not an integer multiple
of the stride along the height (viz. width) as specified in strides, the bottom
(viz. rightmost) cell's height (viz. width) is truncated, and we select the
maximum value from the truncated cell.

Note that `x` and `x_pooled` must have sizes that match according to `strides`.

TODO: finish up documentation
"""
function maxpool{T<:JuMP.AbstractJuMPScalar}(
    xs::Array{T, 4},
    params::PoolParameters{4})
    return poolmap(
        (x) -> NNOps.maximum(x[:]),
        xs,
        params.strides
    )
end

function maximum_with_relu{T<:JuMP.AbstractJuMPScalar}(xs::Array{T, 1})::JuMP.GenericAffExpr
    if length(xs) == 1
        return xs[1]
    else
        a = xs[1]
        b = length(xs) == 2 ? xs[2] : NNOps.maximum(xs[2:end])
        return relu(a-b) + b
    end
end

function maximum{T<:JuMP.AbstractJuMPScalar}(xs::Array{T, 1})::JuMP.Variable
    @assert length(xs) >= 1
    model = ConditionalJuMP.getmodel(xs[1])
    x_max = @variable(model,
        lowerbound = Base.maximum(map(lowerbound, xs)),
        upperbound = Base.maximum(map(upperbound, xs)))
    indicators = []
    for x in xs
        a = @variable(model, category =:Bin)
        @implies(model, a, x_max == x)
        @implies(model, 1 - a, x_max >= x)
        push!(indicators, a)
    end
    @constraint(model, sum(indicators) == 1)
    return x_max
end

"""
Computes the rectification of `x`
"""
function relu(x::Real)::Real
    return max(0, x)
end

function relu(x::JuMP.AbstractJuMPScalar)::JuMP.Variable
    model = ConditionalJuMP.getmodel(x)
    x_rect = @variable(model)
    u = upperbound(x)
    l = lowerbound(x)

    if u < 0
        # rectified value is always 0
        @constraint(model, x_rect == 0)
        setlowerbound(x_rect, 0)
        setupperbound(x_rect, 0)
    elseif l > 0
        # rectified value is always equal to x itself.
        @constraint(model, x_rect == x)
        setlowerbound(x_rect, l)
        setupperbound(x_rect, u)
    else
        a = @variable(model, category = :Bin)

        # refined big-M formulation that takes advantage of the knowledge
        # that lower and upper bounds  are different.
        @constraint(model, x_rect <= x + (-l)*(1-a))
        @constraint(model, x_rect >= x)
        @constraint(model, x_rect <= u*a)
        @constraint(model, x_rect >= 0)

        # model.ext[:objective] = get(model.ext, :objective, 0) + x_rect - x
        model.ext[:objective] = get(model.ext, :objective, 0) + x_rect - x*u/(u-l)

        # Manually set the bounds for x_rect so they can be used by downstream operations.
        setlowerbound(x_rect, 0)
        setupperbound(x_rect, u)
    end

    return x_rect
end

function convlayer{T<:JuMPReal}(
    x::Array{T, 4},
    params::ConvolutionLayerParameters)
    x_conv = params.conv2dparams(x)
    x_maxpool = maxpool(x_conv, params.maxpoolparams)
    x_relu = relu.(x_maxpool)
    return x_relu
end

function matmul{T<:JuMPReal}(
    x::Array{T, 1}, 
    params::MatrixMultiplicationParameters)
    return params.matrix*x .+ params.bias
end

function fullyconnectedlayer{T<:JuMPReal}(
    x::Array{T, 1}, 
    params::FullyConnectedLayerParameters)
    # TODO: Check with Robin whether I can force type inference here.
    return relu.(x |> params.mmparams)
end

# TODO: Handle interaction between setting max index and setting unmax index.
# TODO: rename to set_max_output_index

function set_max_index{T<:JuMP.AbstractJuMPScalar}(
    x::Array{T, 1},
    target_index::Integer,
    tol::Real = 0)
    """
    Sets the target index to be the maximum.

    Tolerance is the amount of gap between x[target_index] and the other elements.
    """
    @assert length(x) >= 1
    @assert (target_index >= 1) && (target_index <= length(x))
    model = ConditionalJuMP.getmodel(x[1])

    other_vars = [x[1:target_index-1]; x[target_index+1:end]]
    @constraint(model, other_vars - x[target_index] .<= -tol)
    
end

function set_unmax_index{T<:JuMP.AbstractJuMPScalar}(
    x::Array{T, 1},
    target_index::Integer,
    tol::Real = 0)
    """
    Sets the target index to NOT be the maximum.
    """
    @assert length(x) >= 1
    @assert (target_index >= 1) && (target_index <= length(x))
    model = ConditionalJuMP.getmodel(x[1])
    x_max = NNOps.maximum(x)
    @constraint(model, x_max - x[target_index] >= tol)
end

function get_max_index{T<:Real}(
    x::Array{T, 1})::Integer
    return findmax(x)[2]
end

function tight_upperbound(x::JuMP.AbstractJuMPScalar)
    m = ConditionalJuMP.getmodel(x)
    @objective(m, Max, x)
    solve(m)
    return min(getobjectivevalue(m), upperbound(x))
end

function tight_lowerbound(x::JuMP.AbstractJuMPScalar)
    m = ConditionalJuMP.getmodel(x)
    @objective(m, Min, x)
    solve(m)
    return max(getobjectivevalue(m), lowerbound(x))
end

function set_input_constraint{T<:Real}(v_input::Array{JuMP.Variable}, input::Array{T})
    @assert length(v_input) > 0
    m = ConditionalJuMP.getmodel(v_input[1])
    @constraint(m, v_input .== input)
end

(p::MatrixMultiplicationParameters){T<:JuMPReal}(x::Array{T, 1}) = matmul(x, p)
(p::Conv2DParameters){T<:JuMPReal}(x::Array{T, 4}) = conv2d(x, p)

(p::ConvolutionLayerParameters){T<:JuMPReal}(x::Array{T, 4}) = convlayer(x, p)
(p::FullyConnectedLayerParameters){T<:JuMPReal}(x::Array{T, 1}) = fullyconnectedlayer(x, p)
(p::SoftmaxParameters){T<:JuMPReal}(x::Array{T, 1}) = p.mmparams(x)

(ps::Array{U, 1}){T<:JuMPReal, U<:Union{ConvolutionLayerParameters, FullyConnectedLayerParameters}}(x::Array{T}) = (
    length(ps) == 0 ? x : ps[2:end](ps[1](x))
)

(p::StandardNeuralNetParameters){T<:JuMPReal}(x::Array{T, 4}) = (
    x |> p.convlayer_params |> NNOps.flatten |> p.fclayer_params |> p.softmax_params
)

"""
Permute dimensions of array because Python flattens arrays in the opposite order.
"""
function flatten{T, N}(x::Array{T, N})
    # return x[:]
    return permutedims(x, N:-1:1)[:]
end

function abs_ge(x::JuMP.AbstractJuMPScalar)::JuMP.Variable
    model = ConditionalJuMP.getmodel(x)
    x_abs = @variable(model)
    u = upperbound(x)
    l = lowerbound(x)
    if u < 0
        @constraint(model, x_abs == -x)
        setlowerbound(x_abs, -u)
        setupperbound(x_abs, -l)
    elseif l > 0
        @constraint(model, x_abs == x)
        setlowerbound(x_abs, l)
        setupperbound(x_abs, u)
    else
        @constraint(model, x_abs >= x)
        @constraint(model, x_abs >= -x)
        setlowerbound(x_abs, 0)
        setupperbound(x_abs, max(-l, u))
    end
    return x_abs
end

function abs_strict(x::JuMP.AbstractJuMPScalar)::JuMP.Variable
    model = ConditionalJuMP.getmodel(x)
    x_abs = @variable(model)
    u = upperbound(x)
    l = lowerbound(x)
    if u < 0
        @constraint(model, x_abs == -x)
        setlowerbound(x_abs, -u)
        setupperbound(x_abs, -l)
    elseif l > 0
        @constraint(model, x_abs == x)
        setlowerbound(x_abs, l)
        setupperbound(x_abs, u)
    else
        a = @variable(model, category = :Bin)
        @constraint(model, x_abs <= x + 2(-l)*(1-a))
        @constraint(model, x_abs >= x)
        @constraint(model, x_abs <= -x + 2*u*a)
        @constraint(model, x_abs >= -x)
        setlowerbound(x_abs, 0)
        setupperbound(x_abs, max(-l, u))
    end
    return x_abs
end

function layer{T<:JuMPReal}(
    input::Array{T, 4},
    params::ConvolutionLayerParameters)
    return convlayer(input, params)
end

function layer{T<:JuMPReal}(
    input::Array{T, 1},
    params::FullyConnectedLayerParameters)
    return fullyconnectedlayer(input, params)
end

function prop{T<:Real, U<:Real, V<:Real}(
    input::Array{T},
    input_lowerbounds::Array{U},
    input_upperbounds::Array{V},
    params::NNParameters.LayerParameters
    )

    @assert size(input) == size(input_lowerbounds) "Size of input does not match size of lowerbounds."
    @assert size(input) == size(input_upperbounds) "Size of input does not match size of upperbounds."

    m = Model(solver=GurobiSolver(MIPFocus = 3))
    ve = map(_ -> @variable(m), input)
    ve_abs = abs_ge.(m, ve)

    vx0 = reshape(
        map(t -> @variable(m, lowerbound = t[1], upperbound = t[2]), zip(input_lowerbounds, input_upperbounds)),
        size(input)
    )

    @constraint(m, vx0 .== input + ve)

    conv_input = layer(input, params)
    vx1 = layer(vx0, params)
    dvx1_abs = abs_strict.(m, vx1-conv_input)

    input_perturbation_norm = sum(ve_abs)
    output_perturbation_norm = sum(dvx1_abs)

    return (m, input_perturbation_norm, output_perturbation_norm)

end

function forwardprop{T<:Real, U<:Real, V<:Real}(
    input::Array{T},
    input_lowerbounds::Array{U},
    input_upperbounds::Array{V},
    params::NNParameters.LayerParameters,
    k_in::Real
    )::Real

    (m, input_perturbation_norm, output_perturbation_norm) =
        prop(input, input_lowerbounds, input_upperbounds, params)

    @constraint(m, input_perturbation_norm <= k_in)
    @objective(m, Max, output_perturbation_norm)

    status = solve(m)

    return getobjectivevalue(m)
end

function backprop{T<:Real, U<:Real, V<:Real}(
    input::Array{T},
    input_lowerbounds::Array{U},
    input_upperbounds::Array{V},
    params::NNParameters.LayerParameters,
    k_out::Real
    )::Real

    (m, input_perturbation_norm, output_perturbation_norm) =
        prop(input, input_lowerbounds, input_upperbounds, params)

    @objective(m, Min, input_perturbation_norm)
    @constraint(m, output_perturbation_norm >= k_out)

    status = solve(m)

    return getobjectivevalue(m)
end

end

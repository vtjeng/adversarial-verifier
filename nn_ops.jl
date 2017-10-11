module NNOps

using Base.Cartesian
using JuMP
using ConditionalJuMP
using Gurobi

"""
For a convolution of `filter` on `input`, determines the size of the output.

# Throws
* ArgumentError if input and filter are not compatible.
"""
function getconv2doutputsize{T, U}(input::AbstractArray{T, 4}, filter::AbstractArray{U, 4})::NTuple{4, Int}
    (batch, in_height, in_width, input_in_channels) = size(input)
    (filter_height, filter_width, filter_in_channels, out_channels) = size(filter)
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
    input::AbstractArray{T, 4},
    filter::AbstractArray{U, 4},
    bias::AbstractArray{V, 1})
    # TEST PLAN:
    #  (1) Incorrectly sized input,
    #  (2) Incorrectly sized filter,
    #  (3) Non-matching elements of array
    #  (4) Non-matching input_in_channels and filter_in_channels
    (batch, in_height, in_width, input_in_channels) = size(input)
    (filter_height, filter_width, filter_in_channels, filter_out_channels) = size(filter)
    bias_out_channels = length(bias)
    if filter_out_channels != bias_out_channels
        throw(ArgumentError("Number of output channels in filter, $filter_out_channels, does not match number of output channels in bias, $bias_out_channels."))
    end

    output_size = getconv2doutputsize(input, filter)

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
        s += bias[i_4]
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
Computes the rectification of `x`
"""
function relu{T<:Real, N}(input::AbstractArray{T, N})::AbstractArray{T, N}
    return max.(0, input)
end

"""
Computes the result of a max-pooling operation on `input_array` with specified
`strides`.
"""
function maxpool{T<:Real, N}(input::AbstractArray{T, N}, strides::NTuple{N, Int})::AbstractArray{T, N}
    # NB: Tried to use pooling function from Knet.relu but it had way too many
    # incompatibilities
    return poolmap(Base.maximum, input, strides)
end

function avgpool{T<:Real, N}(input::AbstractArray{T, N}, strides::NTuple{N, Int})::AbstractArray{T, N}
    return poolmap(mean, input, strides)
end

function convlayer{T<:Real, U<:Real, V<:Real}(input::AbstractArray{T, 4}, filter::AbstractArray{U, 4}, bias::AbstractArray{V, 1}, strides::NTuple{4, Int})
    return maxpool(relu(conv2d(input, filter, bias)), strides)
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
function getsliceindex(input_array_size::Int, stride::Int, output_index::Int)::AbstractArray{Int, 1}
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
function getpoolview{T<:Any, N}(input_array::AbstractArray{T, N}, strides::NTuple{N, Int}, output_index::NTuple{N, Int})::SubArray{T, N}
    it = zip(size(input_array), strides, output_index)
    input_index_range = map((x)-> getsliceindex(x...), it)
    return view(input_array, input_index_range...)
end

"""
For pooling operations on an array, returns the expected size of the output
array.
"""
function getoutputsize{T<:Any, N}(input_array::AbstractArray{T, N}, strides::NTuple{N, Int})::NTuple{N, Int}
    output_size = ((x, y) -> round(Int, x/y, RoundUp)).(size(input_array), strides)
    return output_size
end

"""
Returns output from applying `f` to subarrays of `input_array`, with the windows
determined by the `strides`.
"""
function poolmap{T<:Any, N}(f::Function, input_array::AbstractArray{T, N}, strides::NTuple{N, Int})
    output_size = getoutputsize(input_array, strides)
    output_indices = collect(CartesianRange(output_size))
    return ((I) -> f(getpoolview(input_array, strides, I.I))).(output_indices)
end

"""
Imposes a 2d convolution constraint between `x` and `x_conv` as determined by
the filters.
"""
function conv2dconstraint{T<:JuMP.AbstractJuMPScalar, U<:Real, V<:Real}(model::JuMP.Model, x::AbstractArray{T, 4}, filter::AbstractArray{U, 4}, bias::AbstractArray{V, 1})
    return conv2d(x, filter, bias)
end

"""
Imposes a rectified linearity constraint between `x` and `x_rect` using
the big-M formulation.

For `|x| < M`, `x_rect` = `x` if `x` > 0 and 0 otherwise.

Note that `x` and `x_rect` must be arrays of the same size.

"""
function reluconstraint{T<:JuMP.AbstractJuMPScalar, N}(model::JuMP.Model, xs::AbstractArray{T, N})::Array{JuMP.Variable, N}
    return map(x -> reluconstraint(model, x), xs)
end

function reluconstraint_cj{T<:JuMP.AbstractJuMPScalar}(model::JuMP.Model, x::T)::JuMP.Variable
    x_rect = @switch(
        (x <= 0) => 0,
        (x >= 0) => x
    )
    setlowerbound(x_rect, 0) # we're smart, so we strengthen the lowerbound.
    # we expect the upperbound to be taken care of.
    return x_rect
end

function reluconstraint{T<:JuMP.AbstractJuMPScalar}(model::JuMP.Model, x::T)::JuMP.Variable
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

        # Manually set the bounds for x_rect so they can be used by downstream operations.
        setlowerbound(x_rect, 0)
        setupperbound(x_rect, u)
    end

    return x_rect
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
function maxpoolconstraint{T<:JuMP.AbstractJuMPScalar}(model::JuMP.Model, xs::AbstractArray{T, 4}, strides::NTuple{4, Integer})::Array{JuMP.Variable, 4}
    # TODO: Fixme change strides to a 4-tuple for consistency.
    return poolmap(
        (x) -> NNOps.maximum(model, x[:]),
        xs,
        strides
    )
end

function maximum{T<:JuMP.AbstractJuMPScalar}(model::JuMP.Model, xs::AbstractArray{T, 1})
    if (Base.maximum(map(upperbound, xs)) == Inf)
        println(xs)
        println(map(upperbound, xs))
        for (a, b) in linearterms(xs[1])
            println(a)
            println(upperbound(b))
            println(" ")
        end
    end
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

function convlayerconstraint{T<:JuMP.AbstractJuMPScalar, U<:Real, V<:Real}(model::JuMP.Model, x::AbstractArray{T, 4}, filter::AbstractArray{U, 4}, bias::AbstractArray{V, 1}, strides::NTuple{4, Integer})::Array{JuMP.Variable, 4}
    x_conv = conv2dconstraint(model, x, filter, bias)
    x_maxpool = maxpoolconstraint(model, x_conv, strides)
    x_relu = reluconstraint(model, x_maxpool)
    return x_relu
end

function matmulconstraint{T<:JuMP.AbstractJuMPScalar, U<:Real, V<:Real}(model::JuMP.Model, x::AbstractArray{T, 1}, weights::AbstractArray{U, 2}, bias::AbstractArray{V, 1})
    return weights*x .+ bias
end

function fullyconnectedlayerconstraint{T<:JuMP.AbstractJuMPScalar, U<:Real, V<:Real}(model::JuMP.Model, x::AbstractArray{T, 1}, weights::AbstractArray{U, 2}, bias::AbstractArray{V, 1})::Array{JuMP.Variable, 1}
    return reluconstraint(model, matmulconstraint(model, x, weights, bias))
end

function fullyconnectedlayer{T<:Real, U<:Real, V<:Real}(x::AbstractArray{T, 1}, weights::AbstractArray{U, 2}, bias::AbstractArray{V, 1})
    return NNOps.relu(weights*x + bias)
end

# TODO: refactor fully connected layer to make non-linearity constraint optional
# TODO: refactor softmax to apply on single variable

function softmaxconstraint{T<:JuMP.AbstractJuMPScalar, U<:Real, V<:Real}(model::JuMP.Model, x::AbstractArray{T, 1}, weights::AbstractArray{U, 2}, bias::AbstractArray{V, 1}, target_index::Integer)
    # TODO: error checking on target index
    x_matmul = matmulconstraint(model, x, weights, bias)
    @constraint(model, x_matmul - x_matmul[target_index].<= 0)
end

function softmaxconstraint{T<:JuMP.AbstractJuMPScalar, U<:Real, V<:Real}(model::JuMP.Model, x::AbstractArray{T, 1}, weights::AbstractArray{U, 2}, bias::AbstractArray{V, 1}, target_index::Integer, tol::Float64)
    # TODO: error checking on target index
    x_matmul = matmulconstraint(model, x, weights, bias)
    for i in 1:size(x_matmul)[1]
        if (i != target_index)
            @constraint(model, x_matmul[i] - x_matmul[target_index]<= tol)
        end
    end
end

function softmaxindex{T<:Real, U<:Real, V<:Real}(x::AbstractArray{T, 1}, weights::AbstractArray{U, 2}, bias::AbstractArray{V, 1})
    return findmax(weights*x + bias)[2]
end

"""
Permute dimensions of array because Python flattens arrays in the opposite order.
"""
function flatten{T, N}(x::AbstractArray{T, N})
    return permutedims(x, N:-1:1)[:]
end

function abs_ge{T<:JuMP.AbstractJuMPScalar}(model::JuMP.Model, x::T)::JuMP.Variable
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

function abs_le{T<:JuMP.AbstractJuMPScalar}(model::JuMP.Model, x::T)::JuMP.Variable
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

function convlayer_forwardprop{T<:Real, U<:Real, V<:Real, W<:Real}(
    input::AbstractArray{T, 4},
    filter::AbstractArray{U, 4},
    bias::AbstractArray{V, 1},
    strides::NTuple{4, Int},
    k_in::W
    )

    m = Model(solver=GurobiSolver(MIPFocus = 3))
    ve = map(_ -> @variable(m), input)
    ve_abs = NNOps.abs_ge.(m, ve)
    @constraint(m, sum(ve_abs) <= k_in)

    vx0 = map(_ -> @variable(m, lowerbound = 0, upperbound = 1), input)
    @constraint(m, vx0 .== input + ve)

    conv_input = NNOps.convlayer(input, filter, bias, strides)
    vx1 = NNOps.convlayerconstraint(m, vx0, filter, bias, strides)
    dvx1_abs = NNOps.abs_le.(m, vx1-conv_input)

    @objective(m, Max, sum(dvx1_abs))

    status = solve(m)

    return getobjectivevalue(m)
end

function convlayer_backprop{T<:Real, U<:Real, V<:Real, W<:Real}(
    input::AbstractArray{T, 4},
    filter::AbstractArray{U, 4},
    bias::AbstractArray{V, 1},
    strides::NTuple{4, Int},
    k_out::W
    )

    m = Model(solver=GurobiSolver(MIPFocus = 3))
    ve = map(_ -> @variable(m), input)
    ve_abs = NNOps.abs_ge.(m, ve)
    # @constraint(m, sum(ve_abs) <= k_in)
    @objective(m, Min, sum(ve_abs))

    vx0 = map(_ -> @variable(m, lowerbound = 0, upperbound = 1), input)
    @constraint(m, vx0 .== input + ve)

    conv_input = NNOps.convlayer(input, filter, bias, strides)
    vx1 = NNOps.convlayerconstraint(m, vx0, filter, bias, strides)
    dvx1_abs = NNOps.abs_le.(m, vx1-conv_input)

    # @objective(m, Max, sum(dvx1_abs))
    @constraint(m, sum(dvx1_abs) >= k_out)

    status = solve(m)

    return getobjectivevalue(m)
end

function fclayer_forwardprop{T<:Real, U<:Real, V<:Real, W<:Real}(
    input::AbstractArray{T, 1},
    mat::AbstractArray{U, 2},
    bias::AbstractArray{V, 1},
    k_in::W
    )

    m = Model(solver=GurobiSolver(MIPFocus = 3))
    ve = map(_ -> @variable(m), input)
    ve_abs = NNOps.abs_ge.(m, ve)
    @constraint(m, sum(ve_abs) <= k_in)

    vx0 = map(_ -> @variable(m, lowerbound = 0, upperbound = 1), input)
    @constraint(m, vx0 .== input + ve)

    fc_input = NNOps.fullyconnectedlayer(input, mat, bias)
    vx1 = NNOps.fullyconnectedlayerconstraint(m, vx0, mat, bias)
    dvx1_abs = NNOps.abs_le.(m, vx1-fc_input)

    @objective(m, Max, sum(dvx1_abs))

    status = solve(m)

    return getobjectivevalue(m)
end

function fclayer_backprop{T<:Real, U<:Real, V<:Real, W<:Real}(
    input::AbstractArray{T, 1},
    mat::AbstractArray{U, 2},
    bias::AbstractArray{V, 1},
    k_out::W
    )

    m = Model(solver=GurobiSolver(MIPFocus = 3))
    ve = map(_ -> @variable(m), input)
    ve_abs = NNOps.abs_ge.(m, ve)
    @objective(m, Min, sum(ve_abs))

    vx0 = map(_ -> @variable(m, lowerbound = 0, upperbound = 1), input)
    @constraint(m, vx0 .== input + ve)

    fc_input = NNOps.fullyconnectedlayer(input, mat, bias)
    vx1 = NNOps.fullyconnectedlayerconstraint(m, vx0, mat, bias)
    dvx1_abs = NNOps.abs_le.(m, vx1-fc_input)

    @constraint(m, sum(dvx1_abs) >= k_out)

    status = solve(m)

    return getobjectivevalue(m)
end

end

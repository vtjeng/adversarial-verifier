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
    return poolmap(maximum, input, strides)
end

function avgpool{T<:Real, N}(input::AbstractArray{T, N}, strides::NTuple{N, Int})::AbstractArray{T, N}
    return poolmap(mean, input, strides)
end

function convlayer{T, U<:Real, V<:Real}(input::AbstractArray{T, 4}, filter::AbstractArray{U, 4}, bias::AbstractArray{V, 1}, strides::NTuple{4, Int})
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
    output_size = getconv2doutputsize(x, filter)
    # TODO: @robin How to take care of this pattern of reshaping a variable
    # while passing kwargs?
    x_conv = reshape(@variable(model, [1:prod(output_size)]), output_size)
    aff_ex = conv2d(x, filter, bias)
    # TODO: change to explicit equality function
    for (x, a) in zip(x_conv, aff_ex)
        print(typeof(a))
        setlowerbound(x, lowerbound(a))
        setupperbound(x, upperbound(a))
    end
    @constraint(model, aff_ex .== x_conv)
    return x_conv

    # return conv2d(x, filter, bias)
end

"""
Imposes a rectified linearity constraint between `x` and `x_rect` using
the big-M formulation.

For `|x| < M`, `x_rect` = `x` if `x` > 0 and 0 otherwise.

Note that `x` and `x_rect` must be arrays of the same size.

"""
function reluconstraint{T<:JuMP.AbstractJuMPScalar, N}(model::JuMP.Model, x::AbstractArray{T, N}, M::Real)::Array{JuMP.Variable, N}
    # x_rect_raw = []
    #
    # @nloops 4 i x begin
    #     push!(x_rect_raw, reluconstraint_return(model, (@nref 4 x i)))
    # end
    #
    # x_rect = reshape(x_rect_raw, size(x))

    x_rect = reshape(@variable(model, [1:length(x)]), size(x))

    a = reshape(@variable(model, [1:length(x)], category = :Bin), size(x))

    @constraint(model, x_rect .<= x + M*(1-a))
    @constraint(model, x_rect .>= x)
    @constraint(model, x_rect .<= M*a)
    @constraint(model, x_rect .>= 0)
    return x_rect
end

function reluconstraint_return{T<:JuMP.AbstractJuMPScalar}(model::JuMP.Model, x::T)
    x_rect = @switch(
        (x <= 0) => 0,
        (x >= 0) => x
    )
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
function maxpoolconstraint{T<:JuMP.AbstractJuMPScalar}(model::JuMP.Model, x::AbstractArray{T, 4}, strides::Tuple{Integer, Integer}, M::Real)::Array{JuMP.Variable, 4}
    (pool_height, pool_width) = strides
    full_strides = (1, pool_height, pool_width, 1)

    x_pooled_size = getoutputsize(x, full_strides)
    x_pooled = reshape(@variable(model, [1:prod(x_pooled_size)]), x_pooled_size)

    a = reshape(@variable(model, [1:length(x)], category = :Bin), size(x))

    @nloops 4 r x_pooled begin
        a_sum = 0
        x_pooled_cur = (@nref 4 x_pooled r)
        getcurpoolview = (input_array) -> getpoolview(input_array, full_strides, @ntuple 4 r)

        for e in zip(getcurpoolview(a), getcurpoolview(x))
            (a_cur, x_cur) = e
            @constraint(model, x_pooled_cur <= x_cur + M*(1-a_cur))
            @constraint(model, x_pooled_cur >= x_cur)
        end

        @constraint(model, sum(getcurpoolview(a)) == 1)
    end
    return x_pooled
end

# function maxpoolconstraint{T<:JuMP.AbstractJuMPScalar}(model::JuMP.Model, x::AbstractArray{T, 4}, strides::Tuple{Integer, Integer}, M::Real)::Array{JuMP.Variable, 4}
#     (pool_height, pool_width) = strides
#     full_strides = (1, pool_height, pool_width, 1)
#
#     x_pooled_size = getoutputsize(x, full_strides)
#     dummy = reshape(1:prod(x_pooled_size), x_pooled_size)
#     x_pooled_raw = []
#
#     @nloops 4 r dummy begin
#         getcurpoolview = (input_array) -> getpoolview(input_array, full_strides, @ntuple 4 r)
#         print(size(getcurpoolview(x)))
#         push!(x_pooled_raw, fat_maximum(model, getcurpoolview(x)))
#     end
#     return reshape(x_pooled_raw, x_pooled_size)
# end
#
# function fat_maximum{T<:JuMP.AbstractJuMPScalar, N}(model::JuMP.Model, x::AbstractArray{T, N})
#     return maximum(model, x[:])
# end
#
#
# function maximum{T<:JuMP.AbstractJuMPScalar}(model::JuMP.Model, x::AbstractArray{T, 1})
#     x_max = @variable(model, lowerbound = Base.maximum(map(getlowerbound, x)), upperbound = Base.maximum(map(getupperbound, x)))
#     println(getupperbound(x_max))
#     println(getlowerbound(x_max))
#     indicators = []
#     for e in x
#         a = @variable(model, category =:Bin)
#         @implies(model, (a == 1) => (x_max == x), (a==0) => (x_max >= x))
#         push!(indicators, a)
#     end
#     @constraint(model, sum(indicators) == 1)
#     return x_max
# end

function convlayerconstraint{T<:JuMP.AbstractJuMPScalar, U<:Real, V<:Real}(model::JuMP.Model, x::AbstractArray{T, 4}, filter::AbstractArray{U, 4}, bias::AbstractArray{V, 1}, strides::Tuple{Integer, Integer}, M::Real)::Array{JuMP.Variable, 4}
    x_conv = conv2dconstraint(model, x, filter, bias)
    x_maxpool = maxpoolconstraint(model, x_conv, strides, M)
    x_relu = reluconstraint(model, x_maxpool, M)
    return x_relu
end

function convlayer{T<:Real, U<:Real, V<:Real}(x::AbstractArray{T, 4}, filter::AbstractArray{U, 4}, bias::AbstractArray{V, 1}, strides::Tuple{Integer, Integer})::Array{Real, 4}
    x_conv = conv2d(x, filter, bias)
    x_maxpool = maxpool(x_conv, (1, strides[1], strides[2], 1))
    x_relu = relu(x_maxpool)
    return x_relu
end


function matmulconstraint{T<:JuMP.AbstractJuMPScalar, U<:Real, V<:Real}(model::JuMP.Model, x::AbstractArray{T, 1}, weights::AbstractArray{U, 2}, bias::AbstractArray{V, 1})::Array{JuMP.Variable, 1}
    # TODO: error checking
    (in_height, in_width) = size(weights)
    x_matmul = @variable(model, [1:in_height])
    @constraint(model, x_matmul .== weights*x + bias)
    return x_matmul
end

function fullyconnectedlayerconstraint{T<:JuMP.AbstractJuMPScalar, U<:Real, V<:Real}(model::JuMP.Model, x::AbstractArray{T, 1}, weights::AbstractArray{U, 2}, bias::AbstractArray{V, 1}, M::Real)::Array{JuMP.Variable, 1}
    return reluconstraint(model, matmulconstraint(model, x, weights, bias), M)
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

function flatten{T}(x::AbstractArray{T, 4})
    # need to permute dimensions because Python flattens arrays in the opposite
    # order
    return permutedims(x, [4, 3, 2, 1])[:]
end

end

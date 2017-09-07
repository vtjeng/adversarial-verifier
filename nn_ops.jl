module NNOps

using Base.Cartesian
using JuMP
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
        throw(ArgumentError(@printf "Number of channels in input %d does not match number of channels %d filters operate on." input_in_channels filter_in_channels))
    end
    return (batch, in_height, in_width, out_channels)
end

"""
Computes a 2D-convolution given 4-D `input` and `filter` tensors.

Mirrors `tf.nn.conv2d` from `tensorflow` package, with `strides` = [1, 1, 1, 1],
 `padding` = 'SAME'.
"""
function conv2d{T<:Real, U<:Real}(input::AbstractArray{T, 4}, filter::AbstractArray{U, 4})
    # TEST PLAN:
    #  (1) Incorrectly sized input,
    #  (2) Incorrectly sized filter,
    #  (3) Non-matching elements of array
    #  (4) Non-matching input_in_channels and filter_in_channels
    (batch, in_height, in_width, input_in_channels) = size(input)
    (filter_height, filter_width, filter_in_channels, out_channels) = size(filter)
    output_size = getconv2doutputsize(input, filter)

    # Considered using offset arrays here, but looks like it currently is not
    # really supported

    # Calculating appropriate offsets so that center of kernel is matched with
    # cell at which correlation is being calculated. Note that tensorflow
    # chooses a specific convention for a dimension with even size which we
    # replicate here.
    filter_height_offset = round(Int, filter_height/2, RoundUp)
    filter_width_offset = round(Int, filter_width/2, RoundUp)
    output = Array{T}(output_size)

    @nloops 4 i output begin
        s = 0
        @nloops 4 j filter begin
            x = i_2 + j_1 - filter_height_offset
            y = i_3 + j_2 - filter_width_offset
            if x > 0 && y > 0 && x<=in_height && y<=in_width
                # Doing bounds check to make sure that we stay within bounds
                # for input. This effectively zero-pads the input.
                # TODO: Use default checkbounds function here instead?
                # TODO: Addition here is a bottleneck; figure out whether
                # you could use append without making this incompatible
                # with normal numbers
                s += input[i_1, x, y, j_3] * filter[j_1, j_2, j_3, j_4]
            end
        end
        (@nref 4 output i) = s
    end

    return output

end

"""
Same as `conv2d` above, but optimized for adding scalars together.
"""
function conv2d{T<:JuMP.GenericAffExpr, U<:Real}(input::AbstractArray{T, 4}, filter::AbstractArray{U, 4})
    (batch, in_height, in_width, input_in_channels) = size(input)
    (filter_height, filter_width, filter_in_channels, out_channels) = size(filter)
    output_size = getconv2doutputsize(input, filter)

    # Considered using offset arrays here, but looks like it currently is not
    # really supported

    # Calculating appropriate offsets so that center of kernel is matched with
    # cell at which correlation is being calculated. Note that tensorflow
    # chooses a specific convention for a dimension with even size which we
    # replicate here.
    filter_height_offset = round(Int, filter_height/2, RoundUp)
    filter_width_offset = round(Int, filter_width/2, RoundUp)
    output = Array{T}(output_size)

    @nloops 4 i output begin
        s = AffExpr([], [], 0)
        @nloops 4 j filter begin
            x = i_2 + j_1 - filter_height_offset
            y = i_3 + j_2 - filter_width_offset
            if x > 0 && y > 0 && x<=in_height && y<=in_width
                append!(s, filter[j_1, j_2, j_3, j_4] * input[i_1, x, y, j_3])
            end
        end
        (@nref 4 output i) = s
    end

    return output

end

"""
Computes the rectification of `x`
"""
function relu{T<:Real, N}(input::AbstractArray{T, N})::AbstractArray{T, N}
    return max(0, input)
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

function convlayer{T, U}(input::AbstractArray{T, 4}, filter::AbstractArray{U, 4}, strides::NTuple{4, Int})
    return maxpool(relu(conv2d(input, filter)), strides)
end

# TODO: make type annotations more specific for all added code in this commit

function fullyconnectedlayer{T, U}(input::AbstractArray{T, 1}, weights::AbstractArray{U, 2})
    # return weights*input
    return relu(weights*input)
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
function conv2dconstraint{T<:JuMP.AbstractJuMPScalar, U<:Real}(model::JuMP.Model, x::AbstractArray{T, 4}, filter::AbstractArray{U, 4})::Array{JuMP.Variable, 4}
    output_size = getconv2doutputsize(x, filter)
    # TODO: @robin take care of this pattern of reshaping a variable while passing kwargs?
    x_conv = reshape(@variable(model, [1:prod(output_size)]), output_size)
    @constraint(model, conv2d(x, filter) .== x_conv)
    return x_conv
end

"""
Imposes a rectified linearity constraint between `x` and `x_rect` using
the big-M formulation. Intended to be more efficient than reluconstraint
by imposing fewer restrictions

For `|x| < M`, `x_rect` = `x` if `x` > 0 and 0 otherwise.

Note that `x` and `x_rect` must be arrays of the same size.

"""
function reluconstraint{T<:JuMP.AbstractJuMPScalar, N}(model::JuMP.Model, x::AbstractArray{T, N}, M::Real)::Array{JuMP.Variable, N}
    x_rect = reshape(@variable(model, [1:length(x)]), size(x))
    a = reshape(@variable(model, [1:length(x)], category = :Bin), size(x))

    @constraint(model, x_rect .<= x + M*(1-a))
    @constraint(model, x_rect .>= x)
    @constraint(model, x_rect .<= M*a)
    @constraint(model, x_rect .>= 0)
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

function convlayerconstraint{T<:JuMP.AbstractJuMPScalar, U<:Real}(model::JuMP.Model, x::AbstractArray{T, 4}, filter::AbstractArray{U, 4}, strides::Tuple{Integer, Integer}, M::Real)::Array{JuMP.Variable, 4}
    x_conv = conv2dconstraint(model, x, filter)
    x_relu = reluconstraint(model, x_conv, M)
    x_maxpool = maxpoolconstraint(model, x_relu, strides, M)
    return x_maxpool
end

function matmulconstraint{T<:JuMP.AbstractJuMPScalar, U<:Real}(model::JuMP.Model, x::AbstractArray{T, 1}, weights::AbstractArray{U, 2})::Array{JuMP.Variable, 1}
    # TODO: error checking
    (in_height, in_width) = size(weights)
    x_matmul = @variable(model, [1:in_height])
    @constraint(model, x_matmul .== weights*x)
    return x_matmul
end

function fullyconnectedlayerconstraint{T<:JuMP.AbstractJuMPScalar, U<:Real}(model::JuMP.Model, x::AbstractArray{T, 1}, weights::AbstractArray{U, 2}, M::Real)::Array{JuMP.Variable, 1}
    return reluconstraint(model, matmulconstraint(model, x, weights), M)
end

# TODO: refactor fully connected layer to make non-linearity constraint optional
# TODO: refactor softmax to apply on signle variable

function softmaxconstraint{T<:JuMP.AbstractJuMPScalar, U<:Real}(model::JuMP.Model, x::AbstractArray{T, 1}, weights::AbstractArray{U, 2}, target_index::Integer)
    # TODO: error checking on target index
    x_matmul = matmulconstraint(model, x, weights)
    @constraint(model, x_matmul - x_matmul[target_index].<= 0)
end

end

using Base.Cartesian
using JuMP
using Gurobi
"""
Computes a 2D-convolution given 4-D `input` and `filter` tensors.

Mirrors `tf.nn.conv2d` from `tensorflow` package, with `strides` = [1, 1, 1, 1],
 `padding` = 'SAME'.
"""
function conv2d{T, U}(input::AbstractArray{T, 4}, filter::AbstractArray{U, 4})

    # TEST PLAN:
    #  (1) Incorrectly sized input,
    #  (2) Incorrectly sized filter,
    #  (3) Non-matching elements of array
    #  (4) Non-matching input_in_channels and filter_in_channels
    (batch, in_height, in_width, input_in_channels) = size(input)
    (filter_height, filter_width, filter_in_channels, out_channels) = size(filter)
    if input_in_channels != filter_in_channels
        throw(ArgumentError())
    else
        in_channels = input_in_channels
    end

    # Considered using offset arrays here, but looks like it currently is not
    # really supported

    # Calculating appropriate offsets so that center of kernel is matched with
    # cell at which correlation is being calculated. Note that tensorflow
    # chooses a specific convention for a dimension with even size which we
    # replicate here.
    filter_height_offset = round(Int, filter_height/2, RoundUp)
    filter_width_offset = round(Int, filter_width/2, RoundUp)

    output = Array(eltype(input), batch, in_height, in_width, out_channels)

    @nloops 4 i output begin
        s = 0
        @nloops 4 j filter begin
            x = i_2 + j_1 - filter_height_offset
            y = i_3 + j_2 - filter_width_offset
            if x > 0 && y > 0 && x<=in_height && y<=in_width
                # Doing bounds check to make sure that we stay within bounds
                # for input. This effectively zero-pads the input.
                # TODO: Use default checkbounds function here instead?
                s += input[i_1, x, y, j_3] * filter[j_1, j_2, j_3, j_4]
            end
        end
        (@nref 4 output i) = s
    end

    return output

end


"""
Imposes a rectified linearity constraint between `x` and `x_rect` using
the big-M formulation.

For `|x| < M`, `x_rect` = `x` if `x` > 0 and 0 otherwise.

Note that `x` and `x_rect` must be arrays of the same size.

"""
function reluconstraint{T<:JuMP.AbstractJuMPScalar, U<:JuMP.AbstractJuMPScalar, N}(model::JuMP.Model, x::Array{T, N}, x_rect::Array{U, N}, M::Number)
    # TODO: Check with Robin whether you can change this into a macro.
    # TODO: Support single-variable recitifed linearities
    if size(x) != size(x_rect)
        throw(ArgumentError())
    end
    @variable(model, a[1:length(x)], category = :Bin)
    a = reshape(a, size(x))

    @constraint(model, x_rect .<= M*a)
    @constraint(model, x_rect .>= -M*a)
    @constraint(model, x_rect .<= x + M*(1-a))
    @constraint(model, x_rect .>= x - M*(1-a))
    @constraint(model, x .>= M*(a-1))
    @constraint(model, x .<= M*a)

end

"""
Imposes a rectified linearity constraint between `x` and `x_rect` using
the big-M formulation.

For `|x| < M`, `x_rect` = `x` if `x` > 0 and 0 otherwise.

Note that `x` and `x_rect` must be arrays of the same size.

"""
function reluconstraint2{T<:JuMP.AbstractJuMPScalar, U<:JuMP.AbstractJuMPScalar, N}(model::JuMP.Model, x::Array{T, N}, x_rect::Array{U, N}, M::Number)
    # TODO: figure out whether this is truly equivalent to the above rectified
    # linearity
    if size(x) != size(x_rect)
        throw(ArgumentError())
    end
    @variable(model, a[1:length(x)], category = :Bin)
    a = reshape(a, size(x))

    @constraint(model, x_rect .<= x + M*(1-a))
    @constraint(model, x_rect .>= x)
    @constraint(model, x_rect .<= M*a)
    @constraint(model, x_rect .>= 0)

end

"""
Imposes a max-pooling constraint between `x` and `x_pool` using the big-M
formulation.

TODO: finish up documentation
"""
function maxpoolconstraint{T<:JuMP.AbstractJuMPScalar, U<:JuMP.AbstractJuMPScalar}(model::JuMP.Model, x::Array{T, 4}, x_pool::Array{U, 4}, strides::Tuple{Integer, Integer}, M::Number)
    # TODO: check whether we can avoid having the user explicitly construct the pooled array.

    (pool_height, pool_width) = strides

    (in_batch, in_height, in_width, in_channels) = size(x)
    (out_batch, out_height, out_width, out_channels) = size(x_pool)

    batch_match = (out_batch==in_batch)
    height_match = (out_height==round(Int, in_height/pool_height, RoundUp))
    width_match = (out_width==round(Int, in_width/pool_width, RoundUp))
    channel_match = (out_channels==in_channels)
    if !(batch_match && height_match && width_match && channel_match)
        throw(ArgumentError())
    end
    # TODO: Wrap matched variable size creation in helper function? Need to
    # figure out whether anonymous syntax allows for it though
    @variable(model, a[1:length(x)], category = :Bin)
    a = reshape(a, size(x))

    # TODO: Re-write by slicing the appropriate parts of a, x and x_pool and applying a more general operation to those slices.
    # TODO: Ask robin whether ∃ more concise syntax for looping over a subset of variables
    @nloops 4 r x_pool begin
        a_sum = 0
        for i in 1:pool_height
            for j in 1:pool_width
                if (r_2-1)*pool_height+i<=in_height && (r_3-1)*pool_width+j<=in_width
                    a_cur = a[r_1, (r_2-1)*pool_height+i, (r_3-1)*pool_width+j, r_4]
                    x_cur = x[r_1, (r_2-1)*pool_height+i, (r_3-1)*pool_width+j, r_4]
                    x_pool_cur = (@nref 4 x_pool r)
                    @constraint(m4, x_pool_cur <= x_cur + M*(1-a_cur))
                    @constraint(m4, x_pool_cur >= x_cur)
                    a_sum += a_cur # TODO: check with Robin, this summation here is probably quite inefficient
                end
            end
        end
    #     @constraint(m4, sum(a[r_1, (r_2-1)*pool_height+i, (r_3-1)*pool_width+j, r_4] for i=1:pool_height, j=1:pool_width) == 1)
        @constraint(m4, a_sum == 1)
    end

end

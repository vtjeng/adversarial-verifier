module NNParameters

export Conv2DParameters, PoolParameters, ConvolutionLayerParameters, MatrixMultiplicationParameters

abstract type LayerParameters end

struct Conv2DParameters{T<:Real, U<:Real} <: LayerParameters
    filter::Array{T, 4}
    bias::Array{U, 1}

    function Conv2DParameters{T, U}(filter::Array{T, 4}, bias::Array{U, 1}) where {T<:Real, U<:Real}
        (filter_height, filter_width, filter_in_channels, filter_out_channels) = size(filter)
        bias_out_channels = length(bias)
        @assert filter_out_channels == bias_out_channels "For the convolution layer, number of output channels in filter, $filter_out_channels, does not match number of output channels in bias, $bias_out_channels."
        return new(filter, bias)
    end

end

function Conv2DParameters(filter::Array{T, 4}, bias::Array{U, 1}) where {T<:Real, U<:Real}
    Conv2DParameters{T, U}(filter, bias)
end

struct PoolParameters{N} <: LayerParameters
    strides::NTuple{N, Int}
end

struct ConvolutionLayerParameters{T<:Real, U<:Real} <: LayerParameters
    conv2dparams::Conv2DParameters{T, U}
    maxpoolparams::PoolParameters{4}

end

function ConvolutionLayerParameters{T<:Real, U<:Real}(filter::Array{T, 4}, bias::Array{U, 1}, strides::NTuple{4, Int})
    ConvolutionLayerParameters(Conv2DParameters(filter, bias), PoolParameters(strides))
end

struct MatrixMultiplicationParameters{T<:Real, U<:Real} <: LayerParameters
    matrix::Array{T, 2}
    bias::Array{U, 1}

    function MatrixMultiplicationParameters{T, U}(matrix::Array{T, 2}, bias::Array{U, 1}) where {T<:Real, U<:Real}
        (matrix_height, matrix_width) = size(matrix)
        bias_height = length(bias)
        @assert matrix_height == bias_height "Number of output channels in matrix, $matrix_height, does not match number of output channels in bias, $bias_height."
        return new(matrix, bias)
    end

end

function MatrixMultiplicationParameters(matrix::Array{T, 2}, bias::Array{U, 1}) where {T<:Real, U<:Real}
    MatrixMultiplicationParameters{T, U}(matrix, bias)
end

end

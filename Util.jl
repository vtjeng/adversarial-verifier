if !(pwd() in LOAD_PATH)
    push!(LOAD_PATH, pwd())
end

module Util

using NNParameters

export check_size, get_label, get_input, get_matrix_params, get_conv_params

function check_size{N}(input::AbstractArray, expected_size::NTuple{N, Int})::Void
    input_size = size(input)
    @assert input_size == expected_size "Input size $input_size did not match expected size $expected_size."
end

function check_size(params::ConvolutionLayerParameters, sizes::NTuple{4, Int})::Void
    check_size(params.conv2dparams, sizes)
end

function check_size(params::Conv2DParameters, sizes::NTuple{4, Int})::Void
    check_size(params.filter, sizes)
    check_size(params.bias, (sizes[4], ))
end

function check_size(params::MatrixMultiplicationParameters, sizes::NTuple{2, Int})::Void
    check_size(params.matrix, sizes)
    check_size(params.bias, (sizes[1], ))
end

function get_label{T<:Real}(y::Array{T, 2}, test_index::Int)::Int
    return findmax(y[test_index, :])[2]
end

function get_input{T<:Real}(x::Array{T, 4}, test_index::Int)::Array{T, 4}
    return x[test_index:test_index, :, :, :]
end

# Maybe merge functionality?
function get_matrix_params(
    param_dict::Dict{String},
    layer_name::String,
    expected_size::NTuple{2, Int};
    matrix_name::String = "weight",
    bias_name::String = "bias")

    params = MatrixMultiplicationParameters(
        transpose(param_dict["$layer_name/$matrix_name"]),
        squeeze(param_dict["$layer_name/$bias_name"], 1)
    )

    check_size(params, expected_size)

    return params
end

function get_conv_params(
    param_dict::Dict{String},
    layer_name::String,
    expected_size::NTuple{4, Int};
    matrix_name::String = "weight",
    bias_name::String = "bias")

    params = Conv2DParameters(
        param_dict["$layer_name/$matrix_name"],
        squeeze(param_dict["$layer_name/$bias_name"], 1)
    )

    check_size(params, expected_size)

    return params
end

end

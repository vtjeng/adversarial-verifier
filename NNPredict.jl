if !(pwd() in LOAD_PATH)
    push!(LOAD_PATH, pwd())
end

module NNPredict

using NNOps
using NNParameters

export predict_label

function predict_label{T<:Real}(
    input::Array{T, 4},
    conv1_params::ConvolutionLayerParameters,
    softmax_params::SoftmaxParameters
    )::Int
    return input |> conv1_params |> NNOps.flatten |> softmax_params
end

function predict_label{T<:Real}(
    input::Array{T, 4},
    conv1_params::ConvolutionLayerParameters,
    fc1_params::FullyConnectedLayerParameters,
    softmax_params::SoftmaxParameters,
    )::Int
    return input |> conv1_params |> NNOps.flatten |> fc1_params |> softmax_params
end

function predict_label{T<:Real}(
    input::Array{T, 4},
    conv1_params::ConvolutionLayerParameters,
    conv2_params::ConvolutionLayerParameters,
    fc1_params::FullyConnectedLayerParameters,
    softmax_params::SoftmaxParameters
    )::Int
    return input |> conv1_params |> conv2_params |> NNOps.flatten |> fc1_params |> softmax_params
end

function predict_label{T<:Real}(
    input::Array{T, 4},
    fc1_params::FullyConnectedLayerParameters,
    fc2_params::FullyConnectedLayerParameters,
    softmax_params::SoftmaxParameters,    
    )::Int
    return input |> NNOps.flatten |> fc1_params |> fc2_params |> softmax_params
end

end
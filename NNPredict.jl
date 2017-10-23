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
    softmax_params::MatrixMultiplicationParameters
    )::Int
    x1 = NNOps.convlayer(input, conv1_params)
    x2 = NNOps.flatten(x1)
    predicted_label = NNOps.softmaxindex(x2, softmax_params)
    return predicted_label
end

function predict_label{T<:Real}(
    input::Array{T, 4},
    conv1_params::ConvolutionLayerParameters,
    fc1_params::MatrixMultiplicationParameters,
    softmax_params::MatrixMultiplicationParameters,
    )::Int
    x1 = NNOps.convlayer(input, conv1_params)
    x2 = NNOps.flatten(x1)
    x3 = NNOps.fullyconnectedlayer(x2, fc1_params)
    predicted_label = NNOps.softmaxindex(x3, softmax_params)
    return predicted_label
end

function predict_label{T<:Real}(
    input::Array{T, 4},
    conv1_params::ConvolutionLayerParameters,
    conv2_params::ConvolutionLayerParameters,
    fc1_params::MatrixMultiplicationParameters,
    softmax_params::MatrixMultiplicationParameters
    )::Int
    x1 = NNOps.convlayer(input, conv1_params)
    x2 = NNOps.convlayer(x1, conv2_params)
    x3 = NNOps.flatten(x2)
    x4 = NNOps.fullyconnectedlayer(x3, fc1_params)
    predicted_label = NNOps.softmaxindex(x4, softmax_params)
    return predicted_label
end

function predict_label{T<:Real}(
    input::Array{T, 4},
    fc1_params::MatrixMultiplicationParameters,
    fc2_params::MatrixMultiplicationParameters,
    softmax_params::MatrixMultiplicationParameters,    
    )::Int
    x1 = NNOps.flatten(input)
    x2 = NNOps.fullyconnectedlayer(x1, fc1_params)
    x3 = NNOps.fullyconnectedlayer(x2, fc2_params)
    predicted_label = NNOps.softmaxindex(x3, softmax_params)
    return predicted_label
end

end
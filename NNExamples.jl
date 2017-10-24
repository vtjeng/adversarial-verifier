if !(pwd() in LOAD_PATH)
    push!(LOAD_PATH, pwd())
end

module NNExamples

using JuMP
using Gurobi

using NNOps
using NNParameters
using NNPredict

function initialize_common{T<:Real}(
    input::Array{T, 4},
    perturbation_warm_start::Union{Void, Array} = nothing
    )::Tuple{JuMP.Model, Array{JuMP.Variable, 4}, Array{JuMP.Variable, 4}}
    m = Model(solver=GurobiSolver(MIPFocus = 3))
    vx0 = map(_ -> @variable(m, lowerbound = 0, upperbound = 1), input)
    ve = map(_ -> @variable(m), input)
    @constraint(m, vx0 .== input + ve)
    if perturbation_warm_start != nothing
        setvalue(ve, perturbation_warm_start)
    end
    return (m, vx0, ve)
end

function initialize{T<:Real, U<:Real}(
    input::Array{T, 4},
    conv1_params::ConvolutionLayerParameters,
    target_output::Array{U, 4},
    perturbation_warm_start::Union{Void, Array} = nothing
    )::Tuple{JuMP.Model, Array{JuMP.Variable}}
    
    (m, vx0, ve) = initialize_common(input, perturbation_warm_start)

    output = vx0 |> conv1_params
    @constraint(m, output .== target_output)

    return (m, ve)
end

function initialize{T<:Real}(
    input::Array{T, 4},
    conv1_params::ConvolutionLayerParameters,
    softmax_params::SoftmaxParameters,
    target_label::Int,
    margin::Real,
    perturbation_warm_start::Union{Void, Array} = nothing
    )::Tuple{JuMP.Model, Array{JuMP.Variable}}

    predicted_label = predict_label(input, conv1_params, softmax_params)
    println("Attempting to find adversarial example. Neural net predicted label is $predicted_label, target label is $target_label")

    (m, vx0, ve) = initialize_common(input, perturbation_warm_start)
    vx0 |> conv1_params |> NNOps.flatten |> (x) -> softmax_params(x, target_label, margin)

    return (m, ve)
end

function initialize{T<:Real}(
    input::Array{T, 4},
    conv1_params::ConvolutionLayerParameters,
    fc1_params::FullyConnectedLayerParameters,
    softmax_params::SoftmaxParameters,
    target_label::Int,
    margin::Real,
    perturbation_warm_start::Union{Void, Array} = nothing
    )::Tuple{JuMP.Model, Array{JuMP.Variable}}

    predicted_label = predict_label(input, conv1_params, fc1_params, softmax_params)
    println("Attempting to find adversarial example. Neural net predicted label is $predicted_label, target label is $target_label")

    (m, vx0, ve) = initialize_common(input, perturbation_warm_start)
    vx0 |> conv1_params |> NNOps.flatten |> fc1_params |> (x) -> softmax_params(x, target_label, margin)

    return(m, ve)
end

function initialize{T<:Real}(
    input::Array{T, 4},
    conv1_params::ConvolutionLayerParameters,
    conv2_params::ConvolutionLayerParameters,
    fc1_params::FullyConnectedLayerParameters,
    softmax_params::SoftmaxParameters,
    target_label::Int,
    margin::Real,
    perturbation_warm_start::Union{Void, Array} = nothing
    )::Tuple{JuMP.Model, Array{JuMP.Variable}}

    predicted_label = predict_label(input, conv1_params, conv2_params, fc1_params, softmax_params)
    println("Attempting to find adversarial example. Neural net predicted label is $predicted_label, target label is $target_label")

    (m, vx0, ve) = initialize_common(input, perturbation_warm_start)
    vx0 |> conv1_params |> conv2_params |> NNOps.flatten |> fc1_params |> (x) -> softmax_params(x, target_label, margin)

    return(m, ve)
end

function initialize{T<:Real}(
    input::Array{T, 4},
    fc1_params::FullyConnectedLayerParameters,
    fc2_params::FullyConnectedLayerParameters,
    softmax_params::SoftmaxParameters,
    target_label::Int,
    margin::Real,
    perturbation_warm_start::Union{Void, Array} = nothing
    )::Tuple{JuMP.Model, Array{JuMP.Variable}}

    predicted_label = predict_label(input, fc1_params, fc2_params, softmax_params)

    println("Attempting to find adversarial example. Neural net predicted label is $predicted_label, target label is $target_label")

    (m, vx0, ve) = initialize_common(input, perturbation_warm_start)
    vx0 |> NNOps.flatten |> fc1_params |> fc2_params |> (x) -> softmax_params(x, target_label, margin)

    return(m, ve)
end

end

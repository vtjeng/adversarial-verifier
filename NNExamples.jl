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

    vx1 = NNOps.convlayer(vx0, conv1_params)
    @constraint(m, vx1 .== target_output)

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
    vx1 = NNOps.convlayer(vx0, conv1_params)
    vx2 = NNOps.flatten(vx1)
    NNOps.softmaxconstraint(vx2, softmax_params, target_label, margin)

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
    vx1 = NNOps.convlayer(vx0, conv1_params)
    vx2 = NNOps.flatten(vx1)
    vx3 = NNOps.fullyconnectedlayer(vx2, fc1_params)
    NNOps.softmaxconstraint(vx3, softmax_params, target_label, margin)

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
    vx1 = NNOps.convlayer(vx0, conv1_params)
    vx2 = NNOps.convlayer(vx1, conv2_params)
    vx3 = NNOps.flatten(vx2)
    vx4 = NNOps.fullyconnectedlayer(vx3, fc1_params)
    NNOps.softmaxconstraint(vx4, softmax_params, target_label, margin)

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

    vx1 = NNOps.flatten(vx0)
    vx2 = NNOps.fullyconnectedlayer(vx1, fc1_params)
    vx3 = NNOps.fullyconnectedlayer(vx2, fc2_params)

    NNOps.softmaxconstraint(vx3, softmax_params, target_label, margin)

    return(m, ve)
end

end

if !(pwd() in LOAD_PATH)
    push!(LOAD_PATH, pwd())
end

module NNExamples

using JuMP
using Gurobi

using NNOps
using NNParameters

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
    nn_params::StandardNeuralNetParameters,
    target_label::Int,
    margin::Real,
    perturbation_warm_start::Union{Void, Array} = nothing
    )::Tuple{JuMP.Model, Array{JuMP.Variable}}

    predicted_label = input |> nn_params
    println("Attempting to find adversarial example. Neural net predicted label is $predicted_label, target label is $target_label")

    (m, vx0, ve) = initialize_common(input, perturbation_warm_start)
    vx0 |> (x) -> nn_params(x, target_label, margin)

    return(m, ve)
end

end

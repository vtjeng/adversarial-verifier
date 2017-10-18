if !(pwd() in LOAD_PATH)
    push!(LOAD_PATH, pwd())
end

module NNExamples

using JuMP
using Gurobi

using NNOps
using NNParameters

function solve_conv{T<:Real, U<:Real, V<:Real}(
    input::Array{T, 4},
    conv1_params::NNParameters.ConvolutionLayerParameters,
    target_output::AbstractArray{U, 4},
    perturbation_warm_start::AbstractArray{V, 4}
    )
    # TODO: Make warm start optional.
    m = Model(solver=GurobiSolver(MIPFocus = 3))

    vx0 = map(_ -> @variable(m, lowerbound = 0, upperbound = 1), input)
    vx1 = NNOps.convlayerconstraint(m, vx0, conv1_params)

    ve = map(_ -> @variable(m), input)
    @objective(m, Min, sum(ve.^2))
    setvalue(ve, perturbation_warm_start)
    @constraint(m, vx0 .== input + ve)
    @constraint(m, vx1 .== target_output)

    status = solve(m)

    println("Objective value: ", getobjectivevalue(m))
    println("e = ", getvalue(ve))
end

function solve_conv_softmax{T<:Real, U<:Real, V<:Real}(
    input::AbstractArray{T, 4},
    conv1_params::ConvolutionLayerParameters,
    softmax_params::MatrixMultiplicationParameters,
    target_label::Int,
    margin::U,
    perturbation_warm_start::AbstractArray{V, 4}
    )

    x1 = NNOps.convlayer(input, conv1_params)
    x2 = NNOps.flatten(x1)
    predicted_label = NNOps.softmaxindex(x2, softmax_params)

    println("Attempting to find adversarial example. Neural net predicted label is $predicted_label, target label is $target_label")

    m = Model(solver=GurobiSolver(MIPFocus = 3))

    vx0 = map(_ -> @variable(m, lowerbound = 0, upperbound = 1), input)
    vx1 = NNOps.convlayerconstraint(m, vx0, conv1_params)
    vx2 = NNOps.flatten(vx1)

    ve = map(_ -> @variable(m), input)
    @objective(m, Min, sum(ve.^2))
    setvalue(ve, perturbation_warm_start)
    @constraint(m, vx0 .== input + ve)
    NNOps.softmaxconstraint(m, vx2, softmax_params, target_label, margin)

    status = solve(m)

    println("Objective value: ", getobjectivevalue(m))
    println("e = ", getvalue(ve))
end

function solve_conv_fc_softmax{T<:Real, U<:Real, V<:Real}(
    input::AbstractArray{T, 4},
    conv1_params::ConvolutionLayerParameters,
    fc1_params::MatrixMultiplicationParameters,
    softmax_params::MatrixMultiplicationParameters,
    target_label::Int,
    margin::U,
    perturbation_warm_start::AbstractArray{V, 4}
    )

    x1 = NNOps.convlayer(input, conv1_params)
    x2 = NNOps.flatten(x1)
    x3 = NNOps.fullyconnectedlayer(x2, fc1_params)
    predicted_label = NNOps.softmaxindex(x3, softmax_params)

    println("Attempting to find adversarial example. Neural net predicted label is $predicted_label, target label is $target_label")

    m = Model(solver=GurobiSolver(MIPFocus = 3))

    vx0 = map(_ -> @variable(m, lowerbound = 0, upperbound = 1), input)
    vx1 = NNOps.convlayerconstraint(m, vx0, conv1_params)
    vx2 = NNOps.flatten(vx1)
    vx3 = NNOps.fullyconnectedlayerconstraint(m, vx2, fc1_params)

    ve = map(_ -> @variable(m), input)
    @objective(m, Min, sum(ve.^2))
    setvalue(ve, perturbation_warm_start)
    @constraint(m, vx0 .== input + ve)
    NNOps.softmaxconstraint(m, vx3, softmax_params, target_label, margin)

    status = solve(m)

    println("Objective value: ", getobjectivevalue(m))
    println("e = ", getvalue(ve))
end

function solve_conv_conv_fc_softmax{T<:Real, U<:Real, V<:Real}(
    input::AbstractArray{T, 4},
    conv1_params::ConvolutionLayerParameters,
    conv2_params::ConvolutionLayerParameters,
    fc1_params::MatrixMultiplicationParameters,
    softmax_params::MatrixMultiplicationParameters,
    target_label::Int,
    margin::U,
    perturbation_warm_start::AbstractArray{V, 4}
    )

    x1 = NNOps.convlayer(input, conv1_params)
    x2 = NNOps.convlayer(x1, conv2_params)
    x3 = NNOps.flatten(x2)
    x4 = NNOps.fullyconnectedlayer(x3, fc1_params)
    predicted_label = NNOps.softmaxindex(x4, softmax_params)

    println("Attempting to find adversarial example. Neural net predicted label is $predicted_label, target label is $target_label")

    m = Model(solver=GurobiSolver(MIPFocus = 3))

    vx0 = map(_ -> @variable(m, lowerbound = 0, upperbound = 1), input)
    vx1 = NNOps.convlayerconstraint(m, vx0, conv1_params)
    vx2 = NNOps.convlayerconstraint(m, vx1, conv2_params)
    vx3 = NNOps.flatten(vx2)
    vx4 = NNOps.fullyconnectedlayerconstraint(m, vx3, fc1_params)

    ve = map(_ -> @variable(m), input)
    @objective(m, Min, sum(ve.^2))
    setvalue(ve, perturbation_warm_start)
    @constraint(m, vx0 .== input + ve)
    NNOps.softmaxconstraint(m, vx4, softmax_params, target_label, margin)

    status = solve(m)

    println("Objective value: ", getobjectivevalue(m))
    println("e = ", getvalue(ve))
end

function solve_fc_fc_softmax{T<:Real}(
    input::AbstractArray{T, 4},
    fc1_params::MatrixMultiplicationParameters,
    fc2_params::MatrixMultiplicationParameters,
    softmax_params::MatrixMultiplicationParameters,
    target_label::Int,
    margin::Float64,
    perturbation_warm_start::AbstractArray{Float64, 4}
    )

    x1 = NNOps.flatten(input)
    x2 = NNOps.fullyconnectedlayer(x1, fc1_params)
    x3 = NNOps.fullyconnectedlayer(x2, fc2_params)
    predicted_label = NNOps.softmaxindex(x3, softmax_params)

    println("Attempting to find adversarial example. Neural net predicted label is $predicted_label, target label is $target_label")

    m = Model(solver=GurobiSolver(MIPFocus = 3))

    vx0 = map(_ -> @variable(m, lowerbound = 0, upperbound = 1), input)
    vx1 = NNOps.flatten(vx0)
    vx2 = NNOps.fullyconnectedlayerconstraint(m, vx1, fc1_params)
    vx3 = NNOps.fullyconnectedlayerconstraint(m, vx2, fc2_params)

    ve = map(_ -> @variable(m), input)
    @objective(m, Min, sum(ve.^2))
    setvalue(ve, perturbation_warm_start)
    @constraint(m, vx0 .== input + ve)
    NNOps.softmaxconstraint(m, vx3, softmax_params, target_label, margin)

    status = solve(m)

    println("Objective value: ", getobjectivevalue(m))
    println("e = ", getvalue(ve))
    return getobjectivevalue(m)
end

end

module NNExamples

include("nn_ops.jl")
using JuMP
using Gurobi

function solve_conv{T<:Real, U<:Real, V<:Real, W<:Real, X<:Real}(
    input::AbstractArray{T, 4},
    filter_c1::AbstractArray{U, 4},
    bias_c1::AbstractArray{V, 1},
    strides_c1::NTuple{4, Int},
    target_output::AbstractArray{W, 4},
    perturbation_warm_start::AbstractArray{X, 4}
    )
    # TODO: Make warm start optional.
    m = Model(solver=GurobiSolver(MIPFocus = 3))

    vx0 = map(_ -> @variable(m, lowerbound = 0, upperbound = 1), input)
    vx1 = NNOps.convlayerconstraint(m, vx0, filter_c1, bias_c1, strides_c1)

    ve = map(_ -> @variable(m), input)
    @objective(m, Min, sum(ve.^2))
    setvalue(ve, perturbation_warm_start)
    @constraint(m, vx0 .== input + ve)
    @constraint(m, vx1 .== target_output)

    status = solve(m)

    println("Objective value: ", getobjectivevalue(m))
    println("e = ", getvalue(ve))
end

function solve_conv_softmax(
    input::AbstractArray{Float64, 4},
    filter_c1::AbstractArray{Float64, 4},
    bias_c1::AbstractArray{Float64, 1},
    strides_c1::NTuple{4, Int},
    mat_s1::AbstractArray{Float64, 2},
    bias_s1::AbstractArray{Float64, 1},
    target_label::Int,
    margin::Float64,
    perturbation_warm_start::AbstractArray{Float64, 4}
    )

    x1 = NNOps.convlayer(input, filter_c1, bias_c1, strides_c1)
    x2 = NNOps.flatten(x1)
    predicted_label = NNOps.softmaxindex(x2, mat_s1, bias_s1)

    println("Attempting to find adversarial example. Neural net predicted label is $predicted_label, target label is $target_label")

    m = Model(solver=GurobiSolver(MIPFocus = 3))

    vx0 = map(_ -> @variable(m, lowerbound = 0, upperbound = 1), input)
    vx1 = NNOps.convlayerconstraint(m, vx0, filter_c1, bias_c1, strides_c1)
    vx2 = NNOps.flatten(vx1)

    ve = map(_ -> @variable(m), input)
    @objective(m, Min, sum(ve.^2))
    setvalue(ve, perturbation_warm_start)
    @constraint(m, vx0 .== input + ve)
    NNOps.softmaxconstraint(m, vx2, mat_s1, bias_s1, target_label, margin)

    status = solve(m)

    println("Objective value: ", getobjectivevalue(m))
    println("e = ", getvalue(ve))
end

function solve_conv_fc_softmax(
    input::AbstractArray{Float64, 4},
    filter_c1::AbstractArray{Float64, 4},
    bias_c1::AbstractArray{Float64, 1},
    strides_c1::NTuple{4, Int},
    mat_fc1::AbstractArray{Float64, 2},
    bias_fc1::AbstractArray{Float64, 1},
    mat_s1::AbstractArray{Float64, 2},
    bias_s1::AbstractArray{Float64, 1},
    target_label::Int,
    margin::Float64,
    perturbation_warm_start::AbstractArray{Float64, 4}
    )

    x1 = NNOps.convlayer(input, filter_c1, bias_c1, strides_c1)
    x2 = NNOps.flatten(x1)
    x3 = NNOps.fullyconnectedlayer(x2, mat_fc1, bias_fc1)
    predicted_label = NNOps.softmaxindex(x3, mat_s1, bias_s1)

    println("Attempting to find adversarial example. Neural net predicted label is $predicted_label, target label is $target_label")

    m = Model(solver=GurobiSolver(MIPFocus = 3))

    vx0 = map(_ -> @variable(m, lowerbound = 0, upperbound = 1), input)
    vx1 = NNOps.convlayerconstraint(m, vx0, filter_c1, bias_c1, strides_c1)
    vx2 = NNOps.flatten(vx1)
    vx3 = NNOps.fullyconnectedlayerconstraint(m, vx2, mat_fc1, bias_fc1)

    ve = map(_ -> @variable(m), input)
    @objective(m, Min, sum(ve.^2))
    setvalue(ve, perturbation_warm_start)
    @constraint(m, vx0 .== input + ve)
    NNOps.softmaxconstraint(m, vx3, mat_s1, bias_s1, target_label, margin)

    status = solve(m)

    println("Objective value: ", getobjectivevalue(m))
    println("e = ", getvalue(ve))
end

function solve_conv_conv_fc_softmax{T<:Real}(
    input::AbstractArray{T, 4},
    filter_c1::AbstractArray{T, 4},
    bias_c1::AbstractArray{T, 1},
    strides_c1::NTuple{4, Int},
    filter_c2::AbstractArray{T, 4},
    bias_c2::AbstractArray{T, 1},
    strides_c2::NTuple{4, Int},
    mat_fc1::AbstractArray{T, 2},
    bias_fc1::AbstractArray{T, 1},
    mat_s1::AbstractArray{T, 2},
    bias_s1::AbstractArray{T, 1},
    target_label::Int,
    margin::Float64,
    perturbation_warm_start::AbstractArray{T, 4}
    )

    x1 = NNOps.convlayer(input, filter_c1, bias_c1, strides_c1)
    x2 = NNOps.convlayer(x1, filter_c2, bias_c2, strides_c2)
    x3 = NNOps.flatten(x2)
    x4 = NNOps.fullyconnectedlayer(x3, mat_fc1, bias_fc1)
    predicted_label = NNOps.softmaxindex(x4, mat_s1, bias_s1)

    println("Attempting to find adversarial example. Neural net predicted label is $predicted_label, target label is $target_label")

    m = Model(solver=GurobiSolver(MIPFocus = 3))

    vx0 = map(_ -> @variable(m, lowerbound = 0, upperbound = 1), input)
    vx1 = NNOps.convlayerconstraint(m, vx0, filter_c1, bias_c1, strides_c1)
    vx2 = NNOps.convlayerconstraint(m, vx1, filter_c2, bias_c2, strides_c2)
    vx3 = NNOps.flatten(vx2)
    vx4 = NNOps.fullyconnectedlayerconstraint(m, vx3, mat_fc1, bias_fc1)

    ve = map(_ -> @variable(m), input)
    @objective(m, Min, sum(ve.^2))
    setvalue(ve, perturbation_warm_start)
    @constraint(m, vx0 .== input + ve)
    NNOps.softmaxconstraint(m, vx4, mat_s1, bias_s1, target_label, margin)

    status = solve(m)

    println("Objective value: ", getobjectivevalue(m))
    println("e = ", getvalue(ve))
end

end

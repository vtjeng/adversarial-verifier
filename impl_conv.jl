include("nn_ops.jl")

"""
Basic example where we express the constraints for a neural net consisting
of a single convolution layer.
"""

# We specify the parameters for the size of the problem that we are solving.
batch = 1
in_height = 10
in_width = 10
stride_height = 2
stride_width = 2
in_channels = 1
filter_height = 5
filter_width = 5
out_channels = 32
bigM = 10000

pooled_height = round(Int, in_height/stride_height, RoundUp)
pooled_width = round(Int, in_width/stride_width, RoundUp)

# Fix random seed so that we get consistent results
srand(5)

# We select a random filter for the convolution.
filter = rand(-10:10, filter_height, filter_width, in_channels, out_channels)

# We select some random input, and determine the activations at each layer
# for that input.
# `x_conv_relu_maxpool_actual` is the target output that we will seek to
# achieve by perturbing `x_current`.
x_actual = rand(-10:10, batch, in_height, in_width, in_channels)
x_conv_actual = NNOps.conv2d(x_actual, filter)
x_conv_relu_actual = NNOps.relu(x_conv_actual)
x_conv_relu_maxpool_actual = NNOps.maxpool(x_conv_relu_actual, (1, 2, 2, 1))

x_current = rand(-10:10, batch, in_height, in_width, in_channels)

using JuMP
using Gurobi

m = Model(solver=GurobiSolver())

# `ve` is the tensor of perturbations, while `vx` is the tensor representing the
# original input which we are perturbing.
# Our objective in this optimization is to minimize the l-2 norm of the
# perturbations.
@variable(m, ve[1:batch, 1:in_height, 1:in_width, 1:in_channels])
@objective(m, Min, sum(ve.^2))
@variable(m, vx[1:batch, 1:in_height, 1:in_width, 1:in_channels])
@constraint(m, vx .== x_current)

# 1a. Apply convolution constraint.
@variable(m, vx_conv[1:batch, 1:in_height, 1:in_width, 1:out_channels])
vx_conv = NNOps.conv2dconstraint(m, vx+ve, filter)

# 1b. Apply relu constraint.
vx_conv_relu = NNOps.reluconstraint(m, vx_conv, bigM)

# 1c. Apply maxpool constraint.
vx_conv_relu_maxpool = NNOps.maxpoolconstraint(m, vx_conv_relu, (2, 2), bigM)
@constraint(m, vx_conv_relu_maxpool .== x_conv_relu_maxpool_actual)

status = solve(m)

println("Objective value: ", getobjectivevalue(m))
println("e = ", getvalue(ve))

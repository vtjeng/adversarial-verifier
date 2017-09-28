include("nn_ops.jl")

"""
Basic example where we express the constraints for a neural net consisting
of a single convolution layer.

We ensure that the output constraint is achievable by passing in some input
through the neural net.

Conventions:
  + x0: Array{Real} input corresponding to the output we are attempting to achieve (a.k.a "target input")
  + x_input: Array{Real} input that we are perturbing
  + xk: Array{Real} activation at kth layer of neural net to target input
  + vx0: Array{AbstractJuMPScalar} perturbed input to neural net (a.k.a. "perturbed input")
  + vxk: Array{AbstractJuMPScalar} activation at kth layer to perturbed input
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
out_channels = 4
bigM = 100

pooled_height = round(Int, in_height/stride_height, RoundUp)
pooled_width = round(Int, in_width/stride_width, RoundUp)

# Fix random seed so that we get consistent results
srand(5)
mt = MersenneTwister(5)

# We select a random filter for the convolution.
filter = rand(filter_height, filter_width, in_channels, out_channels)*2-1
bias = (rand(out_channels)*2-1)*0.25

# We select some random input, and determine the activations at each layer
# for that input.
# `x_conv_relu_maxpool_actual` is the target output that we will seek to
# achieve by perturbing `x_input`.
x0 = rand(batch, in_height, in_width, in_channels)
x1 = NNOps.convlayer(x0, filter, bias, (2, 2))

x_input = rand(batch, in_height, in_width, in_channels)

using JuMP
using Gurobi

m = Model(solver=GurobiSolver(MIPFocus = 3))

# `ve` is the tensor of perturbations, while `vx` is the tensor representing the
# original input which we are perturbing.
# Our objective in this optimization is to minimize the l-2 norm of the
# perturbations.
@variable(m, ve[1:batch, 1:in_height, 1:in_width, 1:in_channels])
@objective(m, Min, sum(ve.^2))
@variable(m, vx0[1:batch, 1:in_height, 1:in_width, 1:in_channels])
@constraint(m, vx0 .== x_input + ve)
setvalue(ve, x0 - x_input)

vx1 = NNOps.convlayerconstraint(m, vx0, filter, bias, (2, 2), bigM)
@constraint(m, vx1 .== x1);

status = solve(m)

println("Objective value: ", getobjectivevalue(m))
println("e = ", getvalue(ve))
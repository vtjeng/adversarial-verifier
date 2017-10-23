if !(pwd() in LOAD_PATH)
    push!(LOAD_PATH, pwd())
end

using JuMP

using Revise

using NNExamples
using NNOps
using NNParameters

"""
Basic example where we express the constraints for a neural net consisting
of a single convolution layer.

We ensure that the output constraint is achievable by passing in some input
through the neural net.

Conventions:
  + x0: Array{Real} input corresponding to the output we are attempting to
    achieve (a.k.a "target input")
  + x_input: Array{Real} input that we are perturbing
  + xk: Array{Real} activation at kth layer of neural net to target input
  + vx0: Array{AbstractJuMPScalar} perturbed input to neural net
    (a.k.a. "perturbed input")
  + vxk: Array{AbstractJuMPScalar} activation at kth layer to perturbed input
"""

# We specify the parameters for the size of the problem that we are solving.
batch = 1
in_height = 10
in_width = 10
in_channels = 1
filter_height = 5
filter_width = 5
out_channels = 4

# Fix random seed so that we get consistent results
srand(5)

conv1params = NNParameters.ConvolutionLayerParameters(
    rand(filter_height, filter_width, in_channels, out_channels)*2-1,
    (rand(out_channels)*2-1)*0.25,
    (1, 2, 2, 1)
)

x0 = rand(batch, in_height, in_width, in_channels)
x1 = NNOps.convlayer(x0, conv1params)

input = rand(batch, in_height, in_width, in_channels)

(m, ve) = NNExamples.initialize(input, conv1params, x1, x0-input)

abs_ve = NNOps.abs_ge.(m, ve)
e_norm = sum(abs_ve)
# @constraint(m, e_norm <= 18)

@objective(m, Min, e_norm)
# @objective(m, Min, m.ext[:objective])

status = solve(m)
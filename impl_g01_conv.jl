if !(pwd() in LOAD_PATH)
    push!(LOAD_PATH, pwd())
end

using JuMP
using Gurobi

using NNExamples
using NNOps

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
stride_height = 2
stride_width = 2
strides = (1, stride_height, stride_width, 1)
in_channels = 1
filter_height = 5
filter_width = 5
out_channels = 4

# Fix random seed so that we get consistent results
srand(5)

# We select a random filter for the convolution.
filter = rand(filter_height, filter_width, in_channels, out_channels)*2-1
bias = (rand(out_channels)*2-1)*0.25

x0 = rand(batch, in_height, in_width, in_channels)
x1 = NNOps.convlayer(x0, filter, bias, strides)

input = rand(batch, in_height, in_width, in_channels)

params = NNOps.ConvolutionLayerParameters(NNOps.Conv2DParameters(filter, bias), NNOps.MaxPoolParameters(strides))
NNExamples.solve_conv(input, params, x1, x0 - input)

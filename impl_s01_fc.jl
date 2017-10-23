# TODO: fix, the helper function for solving forward propagation is too strict
# with its inputs

if !(pwd() in LOAD_PATH)
    push!(LOAD_PATH, pwd())
end

using MAT

using NNOps
using Util

UUID = "2017-09-28_181157"
nn_params = matread("data/$UUID-ch-params.mat")

conv1params = NNParameters.ConvolutionLayerParameters(
    nn_params["conv1/weight"],
    squeeze(nn_params["conv1/bias"], 1),
    (1, 2, 2, 1)
)

conv2params = NNParameters.ConvolutionLayerParameters(
    nn_params["conv2/weight"],
    squeeze(nn_params["conv2/bias"], 1),
    (1, 1, 1, 1)
)

fc1params = NNParameters.MatrixMultiplicationParameters(
    transpose(nn_params["fc1/weight"]),
    squeeze(nn_params["fc1/bias"], 1)
)

k_out3 = 0.0001

mnist_test_data = matread("data/mnist_test_data.mat")
x_resize = mnist_test_data["x14_"]
test_index = 2
x0 = get_input(x_resize, test_index)
x1 = NNOps.convlayer(x0, conv1params)
x2 = NNOps.convlayer(x1, conv2params)
input = NNOps.flatten(x2)

input_lowerbounds = map(_ -> -Inf, input)
input_upperbounds = map(_ -> Inf, input)

k_in3 = NNOps.forwardprop(input, input_lowerbounds, input_upperbounds, fc1params, k_out3)
println(k_in3)

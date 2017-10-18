if !(pwd() in LOAD_PATH)
    push!(LOAD_PATH, pwd())
end

using MAT

using NNOps
using NNParameters
using Util

UUID = "2017-09-28_181157"
nn_params = matread("data/$UUID-ch-params.mat")
filter1 = nn_params["conv1/weight"]
bias1 = squeeze(nn_params["conv1/bias"], 1)
k_in1 = 1

convparams = NNParameters.ConvolutionLayerParameters(
    nn_params["conv1/weight"],
    squeeze(nn_params["conv1/bias"], 1),
    (1, 2, 2, 1)
)

# get input
mnist_test_data_resized = matread("data/mnist_test_data_resized.mat")
x_resize = mnist_test_data_resized["x_resize"]
test_index = 2
input = get_input(x_resize, test_index)

input_lowerbounds = map(_ -> 0, input)
input_upperbounds = map(_ -> 1, input)

k_out1 = NNOps.forwardprop(input, input_lowerbounds, input_upperbounds, convparams, k_in1)
println(k_out1)

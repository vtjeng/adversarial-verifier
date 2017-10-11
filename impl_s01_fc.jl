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
filter1 = nn_params["conv1/weight"]
bias1 = squeeze(nn_params["conv1/bias"], 1)
strides1 = (1, 2, 2, 1)
filter2 = nn_params["conv2/weight"]
bias2 = squeeze(nn_params["conv2/bias"], 1)
strides2 = (1, 1, 1, 1) # TODO: special case for pool size of 1
A = transpose(nn_params["fc1/weight"])
biasA = squeeze(nn_params["fc1/bias"], 1)
k_in3 = 10

mnist_test_data_resized = matread("data/mnist_test_data_resized.mat")
x_resize = mnist_test_data_resized["x_resize"]
test_index = 2
input = get_input(x_resize, test_index)
x1 = NNOps.convlayer(input, filter1, bias1, strides1)
x2 = NNOps.convlayer(x1, filter2, bias2, strides2)
k_out3 = NNOps.fclayer_forwardprop(NNOps.flatten(x2), A, biasA, k_in3)
println(k_out3)

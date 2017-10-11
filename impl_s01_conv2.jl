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
k_in2 = 2

mnist_test_data_resized = matread("data/mnist_test_data_resized.mat")
x_resize = mnist_test_data_resized["x_resize"]
test_index = 2
input = get_input(x_resize, test_index)
x1 = NNOps.convlayer(input, filter1, bias1, strides1)
println(minimum(x1))
println(maximum(x1))
k_out2 = NNOps.convlayer_forwardprop(x1, filter2, bias2, strides2, k_in2)
println(k_out2)

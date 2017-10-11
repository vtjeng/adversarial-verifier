using MAT
include("nn_ops.jl")
include("util.jl")

UUID = "2017-09-28_181157"
nn_params = matread("data/$UUID-ch-params.mat")
filter1 = nn_params["conv1/weight"]
bias1 = squeeze(nn_params["conv1/bias"], 1)
strides1 = (1, 2, 2, 1)
k_in1 = 1

mnist_test_data_resized = matread("data/mnist_test_data_resized.mat")
x_resize = mnist_test_data_resized["x_resize"]
test_index = 2
input = get_input(x_resize, test_index)
k_out1 = NNOps.convlayer_forwardprop(input, filter1, bias1, strides1, k_in1)
println(k_out1)

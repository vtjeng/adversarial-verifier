if !(pwd() in LOAD_PATH)
    push!(LOAD_PATH, pwd())
end

using MAT
using JuMP
using Gurobi

using NNExamples
using NNParameters
using NNOps
using Util

### Parameters for neural net
batch = 1
in1_height = 28
in1_width = 28

A_height = 40
A_width = in1_height*in1_width

B_height = 20
B_width = A_height

C_height = 10
C_width = B_height

UUID = "2017-10-09_201838"
param_dict = matread("data/$UUID-ch-params.mat")
x_ = matread("data/$UUID-adversarial-examples.mat")["x"]
y_ = matread("data/mnist_test_data_resized.mat")["y_"]
test_index = 2 # which test sample we're choosing

x0 = get_input(x_, test_index)
actual_label = get_label(y_, test_index)

fc1params = get_matrix_params(param_dict, "fc1", (A_height, A_width))
fc2params = get_matrix_params(param_dict, "fc2", (B_height, B_width))
softmaxparams = get_matrix_params(param_dict, "logits", (C_height, C_width))

for target_label in 1:1
    NNExamples.solve_fc_fc_softmax(
        x0,
        fc1params, fc2params, softmaxparams,
        target_label, 0.0, map(_ -> 0.0, x0))
end

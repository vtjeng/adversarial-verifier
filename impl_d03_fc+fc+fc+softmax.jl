if !(pwd() in LOAD_PATH)
    push!(LOAD_PATH, pwd())
end

using MAT
using JuMP

using NNExamples
using NNParameters
using NNOps
using Util
using NNSolve

### Parameters for neural net
batch = 1
in1_height = 28
in1_width = 28

A_height = 24
A_width = in1_height*in1_width

B_height = 24
B_width = A_height

C_height = 24
C_width = B_height

D_height = 10
D_width = C_height

UUID = "2017-10-23_170857"
param_dict = matread("data/$UUID-ch-params.mat")

fc1params = get_matrix_params(param_dict, "fc1", (A_height, A_width)) |> FullyConnectedLayerParameters
fc2params = get_matrix_params(param_dict, "fc2", (B_height, B_width)) |> FullyConnectedLayerParameters
fc3params = get_matrix_params(param_dict, "fc3", (C_height, C_width)) |> FullyConnectedLayerParameters
softmaxparams = get_matrix_params(param_dict, "logits", (D_height, D_width)) |> SoftmaxParameters

nnparams = StandardNeuralNetParameters(
    ConvolutionLayerParameters[], 
    [fc1params, fc2params, fc3params], 
    softmaxparams,
    UUID
)

num_samples = 100
num_correct = NNSolve.test_performance(nnparams, num_samples)
println("Total correct: $num_correct/$num_samples")

NNSolve.find_adversarial_examples(nnparams, 1, 100)
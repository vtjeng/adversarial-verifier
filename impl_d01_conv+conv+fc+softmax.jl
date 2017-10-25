if !(pwd() in LOAD_PATH)
    push!(LOAD_PATH, pwd())
end

using MAT
using JuMP

using NNExamples
using NNParameters
using NNOps
using Util

# TODO: fix, this guy is out of date and not using the right infrastructure.

### Parameters for neural net
batch = 1
in1_height = 14
in1_width = 14
stride1_height = 2
stride1_width = 2
strides1 = (1, stride1_height, stride1_width, 1)
pooled1_height = round(Int, in1_height/stride1_height, RoundUp)
pooled1_width = round(Int, in1_width/stride1_width, RoundUp)
in1_channels = 1
filter1_height = 3
filter1_width = 3
out1_channels = 4

in2_height = pooled1_height
in2_width = pooled1_width
stride2_height = 1
stride2_width = 1
strides2 = (1, stride2_height, stride2_width, 1)
pooled2_height = round(Int, in2_height/stride2_height, RoundUp)
pooled2_width = round(Int, in2_width/stride2_width, RoundUp)
in2_channels = out1_channels
filter2_height = 3
filter2_width = 3
out2_channels = 8

A_height = 64
A_width = pooled2_height*pooled2_width*out2_channels

B_height = 10
B_width = A_height

UUID = "2017-09-28_181157"
param_dict = matread("data/$UUID-ch-params.mat")
x_adv = matread("data/$UUID-adversarial-examples.mat")["adv_x"]
mnist_test_data = matread("data/mnist_test_data.mat")
test_index = 2 # which test sample we're choosing
y_ = mnist_test_data["y_"]
x_resize = mnist_test_data["x14_"]
actual_label = get_label(y_, test_index)
x0 = get_input(x_resize, test_index)

conv1params = ConvolutionLayerParameters(
    get_conv_params(param_dict, "conv1", (filter1_height, filter1_width, in1_channels, out1_channels)),
    PoolParameters(strides1)
    )

conv2params = ConvolutionLayerParameters(
    get_conv_params(param_dict, "conv2", (filter2_height, filter2_width, in2_channels, out2_channels)),
    PoolParameters(strides2)
    )

fc1params = get_matrix_params(param_dict, "fc1", (A_height, A_width)) |> FullyConnectedLayerParameters
softmaxparams = get_matrix_params(param_dict, "logits", (B_height, B_width)) |> SoftmaxParameters

check_size(x0, (batch, in1_height, in1_width, in1_channels))

nnparams = StandardNeuralNetParameters(
    [conv1params, conv2params], 
    [fc1params], 
    softmaxparams,
    UUID
)

num_samples = 100
num_correct = 0

target_label = -1 # TODO: fix
min_dist = Inf
target_sample_index = -1 # TODO: fix

test_predicted_label = x0 |> nnparams
adversarial_image = NNOps.avgpool(get_input(x_adv, test_index), PoolParameters((1, 2, 2, 1)))
adversarial_predicted_label = adversarial_image |> nnparams
println("FGSM adversarial image predicted label by NN is $adversarial_predicted_label, original image predicted label by NN is $test_predicted_label.")
for i = 1:num_samples
    sample_image = get_input(x_resize, i)
    sample_predicted_label = sample_image |> nnparams
    sample_actual_label = get_label(y_, i)
    # println("Running test case $i. Predicted is $pred, actual is $actual.")
    if sample_predicted_label == sample_actual_label
        num_correct += 1
    end
    sample_dist = sum((sample_image-x0).^2)
    if (sample_predicted_label != test_predicted_label) && (sample_dist < min_dist)
        target_label = sample_predicted_label
        min_dist = sample_dist
        target_sample_index = i
        println("New minimum distance, $min_dist at target sample index $target_sample_index.")
    end
end
candidate_adversarial_example = get_input(x_resize, target_sample_index)

adversarial_dist = sum((adversarial_image-x0).^2)
if (adversarial_predicted_label!=test_predicted_label) && (adversarial_dist < min_dist)
    target_label = adversarial_predicted_label
    min_dist = adversarial_dist
    candidate_adversarial_example = adversarial_image
    println("Using adversarial example at new minimum distance, $adversarial_dist.")
end
println("Number correct on regular samples is $num_correct out of $num_samples.")

(m, ve) = NNExamples.initialize(
    x0,
    nnparams,
    target_label, 0.0, candidate_adversarial_example - x0)

abs_ve = NNOps.abs_ge.(ve)
e_norm = sum(abs_ve)
          
@objective(m, Min, e_norm)
        
status = solve(m)

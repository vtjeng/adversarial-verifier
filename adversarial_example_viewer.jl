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

"""
Loads the parameters for a neural net and the adversarial examples
generated for that net, verifying that the labels are different
and providing information on the distance between the
"""

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

nnparams = StandardNeuralNetParameters(
    [conv1params, conv2params], 
    [fc1params], 
    softmaxparams,
    UUID
)

mnist_test_data = matread("data/mnist_test_data.mat")
y_ = mnist_test_data["y_"]
x_resize = mnist_test_data["x14_"]
x_adv = matread("data/$UUID-adversarial-examples.mat")["adv_x"]

num_adv = 0
for test_index = 1:size(x_adv)[1]
    actual_label = get_label(y_, test_index)
    x0 = get_input(x_resize, test_index)

    test_predicted_label = x0 |> nnparams |> NNOps.get_max_index
    adversarial_image = NNOps.avgpool(get_input(x_adv, test_index), PoolParameters((1, 2, 2, 1)))
    adversarial_predicted_label = adversarial_image |> nnparams |> NNOps.get_max_index

    if (test_predicted_label != adversarial_predicted_label)
        num_adv += 1
        println("For index $test_index, FGSM adversarial image predicted label by NN is $adversarial_predicted_label, original image predicted label by NN is $test_predicted_label.")
        adversarial_dist = sum((adversarial_image-x0).^2)
        println("Adversarial example is at distance of $adversarial_dist.")
    end

end

println("\nTotal number of examples that are truly adversarial is $num_adv.")

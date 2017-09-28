using MAT
using JuMP
using Gurobi
include("nn_ops.jl")
include("util.jl")

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
pooled2_height = round(Int, in2_height/stride2_height, RoundUp)
pooled2_width = round(Int, in2_width/stride2_width, RoundUp)
in2_channels = out1_channels
filter2_height = 3
filter2_width = 3
out2_channels = 8

bigM = 10000

A_height = 64
A_width = pooled2_height*pooled2_width*out2_channels

B_height = 10
B_width = A_height

UUID = "2017-09-28_181157"
nn_params = matread("data/$UUID-ch-params.mat")
x_adv = matread("data/$UUID-adversarial-examples.mat")["adv_x"]
mnist_test_data_resized = matread("data/mnist_test_data_resized.mat")
y_ = mnist_test_data_resized["y_"]
x_resize = mnist_test_data_resized["x_resize"]

filter1 = nn_params["conv1/weight"]
bias1 = squeeze(nn_params["conv1/bias"], 1)
filter2 = nn_params["conv2/weight"]
bias2 = squeeze(nn_params["conv2/bias"], 1)
A = transpose(nn_params["fc1/weight"])
biasA = squeeze(nn_params["fc1/bias"], 1)
B = transpose(nn_params["logits/weight"])
biasB = squeeze(nn_params["logits/bias"], 1)

function calculate_predicted_label{T<:Real}(
    x0_::AbstractArray{T, 4})::Int
    x1_ = NNOps.convlayer(x0_, filter1, bias1, (stride1_height, stride1_width))
    x2_ = NNOps.convlayer(x1_, filter2, bias2, (stride2_height, stride2_width))
    x3_ = NNOps.fullyconnectedlayer(permutedims(x2_, [4, 3, 2, 1])[:], A, biasA)
    predicted_label = NNOps.softmaxindex(x3_, B, biasB)
    return predicted_label
end

num_adv = 0
for test_index = 1:size(x_adv)[1]
    actual_label = get_label(y_, test_index)
    x0 = get_input(x_resize, test_index) # NB: weird indexing preserves singleton first dimension

    test_predicted_label = calculate_predicted_label(x0)
    adversarial_image = NNOps.avgpool(get_input(x_adv, test_index), (1, 2, 2, 1))
    adversarial_predicted_label = calculate_predicted_label(adversarial_image)

    if (test_predicted_label != adversarial_predicted_label)
        num_adv += 1
        println("For index $test_index, FGSM adversarial image predicted label by NN is $adversarial_predicted_label, original image predicted label by NN is $test_predicted_label.")
        adversarial_dist = sum((adversarial_image-x0).^2)
        println("Adversarial example is at distance of $adversarial_dist.")
    end

end

println("\nTotal number of examples that are truly adversarial is $num_adv")
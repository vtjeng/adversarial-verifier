module NNSolve

using MAT
using Util
using NNOps
using NNExamples
using NNParameters
using JuMP

function test_performance(nnparams::NeuralNetParameters, num_samples::Int)::Int
    """
    Returns the number of correctly classified items our neural net obtains
    of the first `num_samples` in the MNIST dataset.
    """

    x_ = matread("data/mnist_test_data.mat")["x_"]
    y_ = matread("data/mnist_test_data.mat")["y_"]

    num_correct = 0
    for sample_index in 1:num_samples
        x0 = get_input(x_, sample_index)
        actual_label = get_label(y_, sample_index)
        predicted_label = x0 |> nnparams |> NNOps.get_max_index
        if actual_label == predicted_label
            num_correct += 1
        end
    end
    return num_correct
end

function find_adversarial_examples(nnparams::NeuralNetParameters, norm_type::Int, num_samples::Int;
    redo_solve::Bool = false, one_sample_per_label = true, correct_classifications_only = true, tolerance::Real=0.0)
    x_ = matread("data/mnist_test_data.mat")["x_"]
    y_ = matread("data/mnist_test_data.mat")["y_"]

    covered_labels = Set{Int}()
    for sample_index in 1:num_samples
        x0 = get_input(x_, sample_index)
        actual_label = get_label(y_, sample_index)
        predicted_label = x0 |> nnparams |> NNOps.get_max_index
        if (actual_label == predicted_label || !correct_classifications_only) && (!in(actual_label, covered_labels) || !one_sample_per_label)
            push!(covered_labels, actual_label)
            println("\nWorking on test sample $sample_index, with ground-truth label $actual_label.")
            find_adversarial_example(x0, nnparams, sample_index, actual_label, norm_type, redo_solve = redo_solve, tolerance = tolerance)
        else
            println("Only working on one sample per label; skipping sample $sample_index.")
        end
        println("--------------------------------------------------------")
    end
end

function find_adversarial_example{T<:Real}(input::Array{T, 4}, nnparams::NeuralNetParameters,
    sample_index::Int, actual_label::Int, norm_type::Int; redo_solve::Bool = false, tolerance::Real=0.0)
    catalog_name = "solve_summaries/$(nnparams.UUID).tol=$tolerance.v1a.mat"
    for target_label in 1:10
        println("------------------------------")
        println("Target label is $target_label")
        do_this_target_label::Bool = !Util.check_solve(catalog_name, sample_index, target_label, norm_type) || redo_solve
        if do_this_target_label
            (m, v_input, v_e, v_output) = NNExamples.initialize(nnparams, size(input))
            
            # Set perturbation constraint
            abs_v_e = NNOps.abs_ge.(v_e)
            e_norm = sum(abs_v_e)
            @objective(m, Min, e_norm)
            
            # Set input constraint
            @constraint(m, v_input .== input)
            
            NNOps.set_max_index(v_output, target_label, tolerance)
            println("Attempting to find adversarial example. Neural net predicted label is $(input |> nnparams |> NNOps.get_max_index), target label is $target_label")
            status = solve(m)

            file_name = "solve_summaries/$(now()).mat"
            Util.save_solve(
                catalog_name, file_name, nnparams.UUID,
                sample_index, actual_label, target_label, 
                norm_type, getvalue(v_e), input, getsolvetime(m), tolerance)
        else
            println("Result found in catalog file; skipping this target label.")
        end
    end
end

end
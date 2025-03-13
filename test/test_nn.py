from nn import nn, preprocess
import numpy as np
import pytest

def test_single_forward():
    nn_test = nn.NeuralNetwork([{'input_dim': 2, 'output_dim': 3, 'activation': "relu"}, 
                                {'input_dim': 3, 'output_dim': 1, 'activation': 'relu'}], 
                                lr=0.001, seed=42, batch_size=2, epochs=3, 
                                loss_function="mean_square_error")
    
    nn_test._param_dict["W1"] = np.array([[1, 1], [1, -1], [-1, -1]])
    nn_test._param_dict["b1"] = np.array([[1], [2], [3]])
    nn_test._param_dict["W2"] = np.array([[1, -1, 1]])
    nn_test._param_dict["b2"] = np.array([[0]])
    
    # Test on a single sample
    A_test = np.array([[3], [2]])
    layer1_forward = nn_test._single_forward(nn_test._param_dict["W1"], 
                                             nn_test._param_dict["b1"], 
                                             A_test, 
                                             nn_test.arch[0]["activation"])
    assert np.all(layer1_forward[0] == np.array([[6], [3], [0]]))
    assert np.all(layer1_forward[1] == np.array([[6], [3], [-2]]))

    layer2_forward = nn_test._single_forward(nn_test._param_dict["W2"], 
                                             nn_test._param_dict["b2"], 
                                             layer1_forward[0], 
                                             nn_test.arch[1]["activation"])
    assert np.all(layer2_forward[0] == np.array([[3]]))
    assert np.all(layer2_forward[1] == np.array([[3]]))

    # Test on multiple samples
    A_test = np.array([[[3], [2]], 
                       [[-1], [-2]]])
    layer1_forward = nn_test._single_forward(nn_test._param_dict["W1"], 
                                             nn_test._param_dict["b1"], 
                                             A_test, 
                                             nn_test.arch[0]["activation"])
    assert np.all(layer1_forward[0] == np.array([[[6], [3], [0]], 
                                                 [[0], [3], [6]]]))
    assert np.all(layer1_forward[1] == np.array([[[6], [3], [-2]], 
                                                 [[-2], [3], [6]]]))

    layer2_forward = nn_test._single_forward(nn_test._param_dict["W2"], 
                                             nn_test._param_dict["b2"], 
                                             layer1_forward[0], 
                                             nn_test.arch[1]["activation"])
    assert np.all(layer2_forward[0] == np.array([[[3]], 
                                                 [[3]]]))
    assert np.all(layer2_forward[1] == np.array([[[3]],
                                                 [[3]]]))

def test_forward():
    nn_test = nn.NeuralNetwork([{'input_dim': 2, 'output_dim': 3, 'activation': "relu"}, 
                                {'input_dim': 3, 'output_dim': 1, 'activation': 'relu'}], 
                                lr=0.001, seed=42, batch_size=2, epochs=3, 
                                loss_function="mean_square_error")
    
    nn_test._param_dict["W1"] = np.array([[1, 1], [1, -1], [-1, -1]])
    nn_test._param_dict["b1"] = np.array([[1], [2], [3]])
    nn_test._param_dict["W2"] = np.array([[1, -1, 1]])
    nn_test._param_dict["b2"] = np.array([[0]])

    # Test on a single sample
    A_test = np.array([[3], [2]])
    forward = nn_test.forward(A_test)
    assert np.all(forward[0] == np.array([[3]]))
    assert np.all(forward[1]["A0"] == np.array([[3], [2]]))
    assert np.all(forward[1]["Z1"] == np.array([[6], [3], [-2]]))
    assert np.all(forward[1]["A1"] == np.array([[6], [3], [0]]))
    assert np.all(forward[1]["Z2"] == np.array([[3]]))
    assert np.all(forward[1]["A2"] == np.array([[3]]))

    # Test on multiple samples
    A_test = np.array([[[3], [2]], 
                       [[-1], [-2]]])
    forward = nn_test.forward(A_test)
    assert np.all(forward[0] == np.array([[[3]], 
                                          [[3]]]))
    assert np.all(forward[1]["A0"] == np.array([[[3], [2]], 
                                                [[-1], [-2]]]))
    assert np.all(forward[1]["Z1"] == np.array([[[6], [3], [-2]], 
                                                [[-2], [3], [6]]]))
    assert np.all(forward[1]["A1"] == np.array([[[6], [3], [0]], 
                                                [[0], [3], [6]]]))
    assert np.all(forward[1]["Z2"] == np.array([[[3]], 
                                                 [[3]]]))
    assert np.all(forward[1]["A2"] == np.array([[[3]], 
                                                 [[3]]]))

def test_single_backprop():
    nn_test = nn.NeuralNetwork([{'input_dim': 2, 'output_dim': 3, 'activation': "relu"}, 
                                {'input_dim': 3, 'output_dim': 1, 'activation': 'relu'}], 
                                lr=0.001, seed=42, batch_size=2, epochs=3, 
                                loss_function="mean_square_error")
    
    nn_test._param_dict["W1"] = np.array([[1, 1], [1, -1], [-1, -1]])
    nn_test._param_dict["b1"] = np.array([[1], [2], [3]])
    nn_test._param_dict["W2"] = np.array([[1, -1, 1]])
    nn_test._param_dict["b2"] = np.array([[0]])

    A_test = np.array([[[3], [2]], 
                       [[-1], [-2]]])
    pred, cache = nn_test.forward(A_test)
    backprop2 = nn_test._single_backprop(nn_test._param_dict["W2"], 
                                         nn_test._param_dict["b2"],
                                         cache["Z2"],
                                         cache["A1"],
                                         1,
                                         nn_test.arch[1]["activation"])
    
    assert np.all(backprop2[0] == np.array([[[1], [-1], [1]], 
                                            [[1], [-1], [1]]]))
    assert np.all(backprop2[1] == np.array([[[6, 3, 0]], [[0, 3, 6]]]))
    assert np.all(backprop2[2] == np.array([[[1]], [[1]]]))

    backprop1 = nn_test._single_backprop(nn_test._param_dict["W1"], 
                                         nn_test._param_dict["b1"],
                                         cache["Z1"],
                                         cache["A0"],
                                         backprop2[0],
                                         nn_test.arch[0]["activation"])
    assert np.all(backprop1[0] == np.array([[[0], [2]], [[-2], [0]]]))
    assert np.all(backprop1[1] == np.array([[[3, 2], [-3, -2], [0, 0]], 
                                            [[0, 0], [1, 2], [-1, -2]]]))
    assert np.all(backprop1[2] == np.array([[[1], [-1], [0]], [[0], [-1], [1]]]))

def test_predict():
    nn_test = nn.NeuralNetwork([{'input_dim': 2, 'output_dim': 3, 'activation': "relu"}, 
                            {'input_dim': 3, 'output_dim': 1, 'activation': 'relu'}], 
                            lr=0.001, seed=42, batch_size=2, epochs=3, 
                            loss_function="mean_square_error")
    
    nn_test._param_dict["W1"] = np.array([[1, 1], [1, -1], [-1, -1]])
    nn_test._param_dict["b1"] = np.array([[1], [2], [3]])
    nn_test._param_dict["W2"] = np.array([[1, -1, 1]])
    nn_test._param_dict["b2"] = np.array([[0]])

    A_test = np.array([[[3], [2]], 
                       [[-1], [-2]]])
    assert np.all(nn_test.predict(A_test) == np.array([[[3]], [[3]]]))

def test_binary_cross_entropy():
    nn_test = nn.NeuralNetwork([{'input_dim': 2, 'output_dim': 3, 'activation': "relu"}, 
                                {'input_dim': 3, 'output_dim': 1, 'activation': 'relu'}], 
                                lr=0.001, seed=42, batch_size=2, epochs=3, 
                                loss_function="binary_cross_entropy")
    y_pred = np.array([[[0.3]], [[0.7]], [[0.1]], [[0.9]]])
    y_true = np.array([[[0]], [[1]], [[0]], [[0]]])
    assert nn_test._binary_cross_entropy(y_true, y_pred) == 0.7803238741323343

def test_binary_cross_entropy_backprop():
    nn_test = nn.NeuralNetwork([{'input_dim': 2, 'output_dim': 3, 'activation': "relu"}, 
                        {'input_dim': 3, 'output_dim': 1, 'activation': 'relu'}], 
                        lr=0.001, seed=42, batch_size=2, epochs=3, 
                        loss_function="binary_cross_entropy")
    y_pred = np.array([[[0.3]], [[0.7]], [[0.1]], [[0.9]]])
    y_true = np.array([[[0]], [[1]], [[0]], [[0]]])
    backprop = nn_test._binary_cross_entropy_backprop(y_true, y_pred)
    assert np.all((backprop - np.array([[[1.42857143]], [[-1.42857143]], [[1.11111111]], [[10]]])) < 0.001)

def test_mean_squared_error():
    nn_test = nn.NeuralNetwork([{'input_dim': 2, 'output_dim': 3, 'activation': "relu"}, 
                        {'input_dim': 3, 'output_dim': 1, 'activation': 'relu'}], 
                        lr=0.001, seed=42, batch_size=2, epochs=3, 
                        loss_function="mean_squared_error")
    y_pred = np.array([[[0.3], [0.7]], [[0.7], [0.3]], [[0.1], [0.1]]])
    y_true = np.array([[[0], [1]], [[1], [0]], [[0], [1]]])
    assert nn_test._mean_squared_error(y_true, y_pred) == 0.39333333333333337

def test_mean_squared_error_backprop():
    nn_test = nn.NeuralNetwork([{'input_dim': 2, 'output_dim': 3, 'activation': "relu"}, 
                        {'input_dim': 3, 'output_dim': 1, 'activation': 'relu'}], 
                        lr=0.001, seed=42, batch_size=2, epochs=3, 
                        loss_function="mean_squared_error")
    y_pred = np.array([[[0.3], [0.7]], [[0.7], [0.3]], [[0.1], [0.1]]])
    y_true = np.array([[[0], [1]], [[1], [0]], [[0], [1]]])
    backprop = nn_test._mean_squared_error_backprop(y_true, y_pred)
    assert np.all((backprop - np.array([[[ 0.6], [-0.6]], [[-0.6], [ 0.6]], [[ 0.2], [-1.8]]])) < 0.0001)

def test_sample_seqs():
    seqs = ["AAA", "TTT", "CCC"]
    labels = [True, True, False]
    sampled_seqs, sampled_labels = preprocess.sample_seqs(seqs, labels)
    assert sampled_seqs == ["AAA", "TTT", "CCC", "CCC"]
    assert sampled_labels == [True, True, False, False]

def test_one_hot_encode_seqs():
    seqs = ["AAA", "ATC", "GCT", "GTA"]
    encoded_seqs = preprocess.one_hot_encode_seqs(seqs)
    assert encoded_seqs == [[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0]]

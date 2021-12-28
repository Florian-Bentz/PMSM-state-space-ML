import numpy as np
import pandas as pd
import tensorflow as tf
from cfg import *
from custom_layer import i_k1_calc_impl1, i_k1_calc_impl2, i_k1_calc_impl3
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.metrics import mean_squared_error as mse


def build_model(state_n, implementation=1):
    """
    Builds a model
    state_n: Defines switching state n
    implementation: Builds model with corresponding layer
    Returns a compiled model with 'Input' and 'Custom Layer'
    """

    input_layer = Input(len(column_lists["input_cols"]), name="input_layer")

    if implementation == 1:
        output_layer = i_k1_calc_impl1(
            state_n, i_k1_calc_params, name="ouput_layer")(input_layer)

    elif implementation == 2:
        output_layer = i_k1_calc_impl2(
            state_n, i_k1_calc_params, name="ouput_layer")(input_layer)

    elif implementation == 3:
        output_layer = i_k1_calc_impl3(
            state_n, i_k1_calc_params, name="ouput_layer")(input_layer)

    else:
        print("ERROR: 'implementation' must be 'int' between 1 and 3")

    model = tf.keras.Model(input_layer, output_layer)

    model.compile(**compile_params)

    return model


def fit_model(model, features, labels):
    """
    Fits the given model to given data
    EarlyStop and ReduceLRONPLateu integrated
    Returns fitted model and loss history
    """

    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-3,
                                          mode="min", patience=10, verbose=1)
    rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min",
                                                 verbose=1, factor=0.1,
                                                 patience=10, min_delta=1e-3)

    history_callback = model.fit(
        features, labels, callbacks=[es, rlrop], **fit_params)
    loss_history = history_callback.history["loss"]

    return (model, loss_history)


def build_nn_model(unit_list):
    """
    Builds a sequential model
    unit_list: every list entry is a layer with corresponding units
    Returns a compiled model
    """
    input_layer = Input(5)

    if len(unit_list) != 0:
        hidden_layer = Dense(units=unit_list[0], **dense_cfg_nn)(input_layer)
        for unit in unit_list[1:]:
            hidden_layer = Dense(units=unit, **dense_cfg_nn)(hidden_layer)

        output_layer = Dense(len(column_lists["output_cols"]),
                             dtype="float64")(hidden_layer)
    else:
        output_layer = Dense(len(column_lists["output_cols"]),
                             dtype="float64")(input_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(**compile_params_nn)
    return model


def fit_nn_model(model, features, labels):
    """
    Fits the given model to given data
    EarlyStop and ReduceLRONPLateu integrated
    Returns fitted model and loss history
    """

    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-3,
                                          mode="min", patience=15, verbose=1)
    rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min",
                                                 verbose=1, factor=0.1,
                                                 patience=5, min_delta=1e-3)
    history_callback = model.fit(
        features, labels, callbacks=[es, rlrop], **fit_params_nn)
    loss_history = history_callback.history["loss"]

    return (model, loss_history)


def evaluate_model(model, features, labels):
    """
    Evaluates the model with test data
    returns RMSE of id and iq predictions
    """

    i_pred = model.predict(features)
    id_pred = i_pred[:, 0] * scaling_cfg["const_values"]["id_k"]
    iq_pred = i_pred[:, 1] * scaling_cfg["const_values"]["iq_k"]

    id_set = labels.loc[:, "id_k1"] * scaling_cfg["const_values"]["id_k1"]
    iq_set = labels.loc[:, "iq_k1"] * scaling_cfg["const_values"]["iq_k1"]

    rmse_id = np.sqrt(mse(id_set, id_pred))
    rmse_iq = np.sqrt(mse(iq_set, iq_pred))

    return (rmse_id, rmse_iq)


def read_feature_data(path, delete=False, size=0):
    """
    simple function to read input data
    path: path to data
    delete: delete "iq_k" > 0
    size: set a custom size, returns all if "0"
    """

    data = pd.read_csv(path)
    if delete:
        data = data[(data["iq_k"] > 0)]

    if size != 0:
        size -= 1
        return data.loc[:size, column_lists["input_cols"]]

    return data.loc[:, column_lists["input_cols"]]


def read_feature_data_nn(path, delete=False, size=0):
    """
    simple function to read input data
    path: path to data
    delete: delete "iq_k" > 0
    size: set a custom size, returns all if "0"
    """

    data = pd.read_csv(path)
    if delete:
        data = data[(data["iq_k"] > 0)]

    if size != 0:
        size -= 1
        return data.loc[:size, column_lists["input_cols_nn"]]

    return data.loc[:, column_lists["input_cols_nn"]]


def read_labels_data(path, delete=False, size=0):
    """
    simple function to read labels data
    path: path to data
    delete: delete "iq_k" > 0
    size: set a custom size, returns all if "0"
    """

    data = pd.read_csv(path)
    if delete:
        data = data[(data["iq_k"] > 0)]

    if size != 0:
        size -= 1
        return data.loc[:size, column_lists["output_cols"]]

    return data.loc[:, column_lists["output_cols"]]


def multi_eval(runs, path_list, state_n, implementation, size=0):
    """
    Combines all above function for muliple evaluations
    runs: number of repitions
    path_list: path list to data [feature, label]
    state_n: Defines switching state n
    implementation: Builds model with corresponding layer
    size: set a custom data size, returns all if "0"
    returns a dictionary with the results
    """

    data = {"RMSE_id": [],
            "RMSE_iq": [],
            "loss": [],
            "parameter": [],
            "n_training_data": [],
            "features": []}

    train_features = read_feature_data(path_list[0], size=size)
    train_labels = read_labels_data(path_list[0], size=size)

    eval_features = read_feature_data(path_list[1])
    eval_labels = read_labels_data(path_list[1])

    for i in range(runs):

        model = build_model(state_n=state_n, implementation=implementation)
        model_params = model.count_params()

        try:

            trained_model = fit_model(
                model, train_features, train_labels, implementation=implementation, state_n=state_n, i=i)

            rmse_id, rmse_iq = evaluate_model(
                trained_model[0], eval_features, eval_labels)

            data["RMSE_id"].append(rmse_id)
            data["RMSE_iq"].append(rmse_iq)
            data["loss"].append(trained_model[1])
            data["parameter"].append(model_params)
            data["n_training_data"].append(train_features.shape[0])
            data["features"].append(column_lists["input_cols"])

            print("Finished Run:", i+1)

        except:
            pass

    results = pd.DataFrame(data)

    return results


def multi_eval_nn(runs, path_list, unit_list, size=0):
    """
    Combines all above function for muliple evaluations
    runs: number of repitions
    path_list: path list to data [feature, label]
    unit_list: every list entry is a layer with corresponding units
    size: set a custom data size, returns all if "0"
    returns a dictionary with the results
    """

    data = {"RMSE_id": [],
            "RMSE_iq": [],
            "loss": [],
            "parameter": [],
            "n_training_data": [],
            "features": []}

    train_features = read_feature_data_nn(
        path_list[0], size=size, delete=False)
    train_labels = read_labels_data(path_list[0], size=size, delete=False)

    eval_features = read_feature_data_nn(path_list[1], delete=False)
    eval_labels = read_labels_data(path_list[1], delete=False)

    for i in range(runs):

        model = build_nn_model(unit_list)

        trained_model = fit_nn_model(
            model, train_features, train_labels)

        model_params = model.count_params()

        rmse_id, rmse_iq = evaluate_model(
            trained_model[0], eval_features, eval_labels)

        data["RMSE_id"].append(rmse_id)
        data["RMSE_iq"].append(rmse_iq)
        data["loss"].append(trained_model[1])
        data["parameter"].append(model_params)
        data["n_training_data"].append(train_features.shape[0])
        data["features"].append(column_lists["input_cols_nn"])

        print("Finished Run:", i+1)

    results = pd.DataFrame(data)

    return results

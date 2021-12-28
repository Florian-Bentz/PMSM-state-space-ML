from tensorflow.keras.initializers import  RandomUniform
from tensorflow.keras.regularizers import l2

i_k1_calc_params = {"testbench_values":   {"supply_voltage": 300,
                                           "rotation_speed": 1000,
                                           "pole_pairs": 3,
                                           "perm_mag_flux": 66e-3,
                                           "sample_time": 50e-6,
                                           "stator_res": 18e-3},

                    "scaling_consts":     {"u_dc": 500,
                                           "w_el": 1900},

                    "switching_states":   {1: [-1, -1, -1],
                                           2: [1, -1, -1],
                                           3: [1, 1, -1],
                                           4: [-1, 1, -1],
                                           5: [-1, 1, 1],
                                           6: [-1, -1, 1],
                                           7: [1, -1, 1]}}

column_lists = {"input_cols": ["id_k", "iq_k", "i_norm_k", "sin_eps_k", "cos_eps_k"],
                "input_cols_nn": ["id_k", "iq_k", "i_norm_k", "sin_eps_k", "cos_eps_k"],
                "output_cols": ["id_k1", "iq_k1"]}


scaling_cfg = {"minmax": ["iq_sqrd"],
               "const": ["id_k", "iq_k", "id_k1", "iq_k1"],
               "const_values": {"id_k": 300,
                                "iq_k": 300,
                                "id_k1": 300,
                                "iq_k1": 300}}

dense_cfg = {"activation": "sigmoid",
             "kernel_initializer": RandomUniform}

dense_cfg_nn = {"activation": "tanh"}
                
"""                "kernel_initializer" : "he_uniform",
                "kernel_regularizer" : l2(1e-9),
                "activity_regularizer" : l2(1e-12)}"""

compile_params = {"loss": "mse",
                  "optimizer": "adam",
                  "run_eagerly": False}

compile_params_nn = {"loss": "mse",
                     "optimizer": "adam",
                     "run_eagerly": False}

fit_params = {"epochs": 800,
              "batch_size": 10000,
              "validation_split": 0.2,
              "verbose": 2}

fit_params_nn = {"epochs": 800,
                 "batch_size": 10000,
                 "validation_split": 0.2,
                 "verbose": 2}


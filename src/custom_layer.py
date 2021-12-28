import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from cfg import *


class i_k1_calc_impl1(Layer):
    def __init__(self, state_n, calc_params, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.state_n = state_n
        self.calc_params = calc_params

        # load testbench parameter / scaling consts
        tb_val = calc_params["testbench_values"]

        self.u_dc = tb_val["supply_voltage"]        # supply voltage
        self.r_s = tb_val["stator_res"]             # stator resistance
        self.sample_time = tb_val["sample_time"]    # load sample_time
        n_me = tb_val["rotation_speed"]
        # calculate angular velocity
        pp = tb_val["pole_pairs"]
        self.w_el = ((2*n_me*np.pi)/60)*pp

        # calculate switching matrix
        s_an, s_bn, s_cn = calc_params["switching_states"][state_n]
        s_abc = (2*s_an-s_bn-s_cn)/3
        s_bc = (s_bn - s_cn)/np.sqrt(3)
        self.s_mat = tf.constant(
            [[s_bc], [s_abc]], dtype=tf.float32)

        # define J matrix
        self.j = tf.constant([[0, -1], [1, 0]], dtype=tf.float32)

        # define layer for L_dq
        self.hidden1 = Dense(8, **dense_cfg)
        self.hidden2 = Dense(8, **dense_cfg)
        self.L_dq_mlp = Dense(2, activation="linear", name="L_dq")

        # define layer for Psi_p
        self.hiddenpsi = Dense(8, **dense_cfg)
        self.psi_p_mlp = Dense(1, activation="linear",
                               name="Psi_d")

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'state_n': self.state_n,
            'calc_params': self.calc_params,
        })
        return config

    def call(self, inputs):
        # load inputs
        i_d_k = inputs[:, 0]
        i_d_k = tf.reshape(i_d_k, [-1, 1])
        i_q_k = inputs[:, 1]
        i_q_k = tf.reshape(i_q_k, [-1, 1])
        i_dq_k = inputs[:, :2]
        i_dq_mul = tf.reshape(i_dq_k, [-1, 2, 1])
        sin_eps = inputs[:, 2]
        sin_eps = tf.reshape(sin_eps, [-1, 1])
        cos_eps = inputs[:, 3]
        cos_eps = tf.reshape(cos_eps, [-1, 1])

        # calculate rotation matrix
        q1 = tf.concat([cos_eps, -sin_eps], axis=1)
        q1 = tf.reshape(q1, [-1, 2, 1])
        q2 = tf.concat([sin_eps, cos_eps], axis=1)
        q2 = tf.reshape(q2, [-1, 2, 1])
        q = tf.concat([q1, q1], axis=2)

        i_norm = tf.reshape(tf.norm(i_dq_k, axis=1), (-1, 1))
        i_input = tf.concat([i_dq_k, i_norm], axis=1)

        # build psi_d
        psi_p = self.psi_p_mlp(self.hiddenpsi(i_input))
        psi_p = tf.concat([psi_p, tf.zeros_like(sin_eps)], axis=1)
        psi_p = tf.reshape(psi_p, [-1, 2, 1])

        # build L_d and L_q
        L_dq = self.L_dq_mlp(self.hidden2(self.hidden1(i_input)))
        L_dq = self.L_dq_mlp(i_input)
        L_d = L_dq[:, 0]
        L_d = tf.reshape(L_d, [-1, 1])
        L_q = L_dq[:, 1]
        L_q = tf.reshape(L_q, [-1, 1])

        # build matrix L_dq
        L_dq1 = tf.concat([L_d, tf.zeros_like(sin_eps)], axis=1)
        L_dq1 = tf.reshape(L_dq1, [-1, 2, 1])
        L_dq2 = tf.concat([tf.zeros_like(sin_eps), L_q], axis=1)
        L_dq2 = tf.reshape(L_dq2, [-1, 2, 1])
        L_dq = tf.concat([L_dq1, L_dq2], axis=2)
        L_dq_inv = tf.linalg.inv(L_dq)

        # idq_k1 calculation (3 Elements)
        elem_1 = -self.r_s*i_dq_mul
        elem_2 = (self.u_dc/2)*(q @ self.s_mat)
        elem_3 = self.j @ (psi_p + L_dq@i_dq_mul) * self.w_el
        elem = elem_1 + elem_2 - elem_3
        i_k1 = L_dq_inv @ elem

        i_k1 = tf.squeeze(i_k1)

        # forward euler
        i_k1 = i_k1 * self.sample_time + i_dq_k
        return i_k1


class i_k1_calc_impl2(Layer):
    def __init__(self, state_n, calc_params, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.state_n = state_n
        self.calc_params = calc_params

        # load testbench parameter / scaling consts
        tb_val = calc_params["testbench_values"]

        self.u_dc = tb_val["supply_voltage"]        # supply voltage
        self.r_s = tb_val["stator_res"]             # stator resistance
        self.sample_time = tb_val["sample_time"]    # load sample_time
        n_me = tb_val["rotation_speed"]
        # calculate angular velocity
        pp = tb_val["pole_pairs"]
        self.w_el = ((2*n_me*np.pi)/60)*pp

        # calculate switching matrix
        s_an, s_bn, s_cn = calc_params["switching_states"][state_n]
        s_abc = (2*s_an-s_bn-s_cn)/3
        s_bc = (s_bn - s_cn)/np.sqrt(3)
        self.s_mat = tf.constant(
            [[s_bc], [s_abc]], dtype=tf.float32)

        # define layer for L_diff
        self.hidden1 = Dense(8, **dense_cfg)
        self.hidden2 = Dense(8, **dense_cfg)
        self.L_diff_out = Dense(4, activation="linear",
                                name="L_diff")

        # define layer for Psi_dq
        self.hiddenpsi = Dense(4, **dense_cfg)

        self.psi_out = Dense(2, activation="linear",
                             name="Psi_out")

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'state_n': self.state_n,
            'calc_params': self.calc_params,
        })
        return config

    def call(self, inputs):
        # load inouts
        i_d_k = inputs[:, 0]
        i_d_k = tf.reshape(i_d_k, [-1, 1])
        i_q_k = inputs[:, 1]
        i_q_k = tf.reshape(i_q_k, [-1, 1])
        i_k_scaled = inputs[:, :2]
        i_norm = tf.reshape(tf.norm(i_k_scaled, axis=1), (-1, 1))
        i_input = tf.concat([i_k_scaled, i_norm], axis=1)

        # build Psi_dq
        psi = self.psi_out(self.hiddenpsi(i_input))
        psi_d = psi[:, :1]
        psi_q = psi[:, 1:2]
        inv_psi = tf.expand_dims(tf.concat([psi_q, -psi_d], axis=1), axis=-1)

        # build L_diff
        L_diff = self.L_diff_out(self.hidden2(self.hidden1(i_input)))
        L_diff = self.L_diff_out(i_input)
        L_diff = tf.reshape(L_diff, [tf.shape(psi)[0], 2, 2])
        inv_L = tf.linalg.inv(L_diff)

        # calculate i_k1
        rhs = tf.concat([tf.fill([tf.shape(inv_psi)[0], 2, 2], -self.r_s),
                         tf.repeat(tf.expand_dims(
                             self.u_dc/2 * self.s_mat, axis=0), tf.shape(inv_psi)[0], axis=0),
                         inv_psi*self.w_el],
                        axis=-1)
        vec = tf.expand_dims(
            tf.concat([inputs[:, :4],
                       tf.ones_like(inputs[:, :1])],
                      axis=1), axis=-1)
        # batch-aware matmul
        i_k1 = inv_L @ rhs @ vec
        i_k1 = tf.squeeze(i_k1)

        # forward euler
        i_k1 = i_k1 * self.sample_time + i_k_scaled
        return i_k1


class i_k1_calc_impl3(Layer):
    def __init__(self, state_n, calc_params, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.state_n = state_n
        self.calc_params = calc_params

        # load testbench parameter / scaling consts
        tb_val = calc_params["testbench_values"]

        self.u_dc = tb_val["supply_voltage"]        # supply voltage
        self.r_s = tb_val["stator_res"]             # stator resistance
        self.sample_time = tb_val["sample_time"]    # load sample_time
        n_me = tb_val["rotation_speed"]
        # calculate angular velocity
        pp = tb_val["pole_pairs"]
        self.w_el = ((2*n_me*np.pi)/60)*pp

        # calculate switching matrix
        s_an, s_bn, s_cn = calc_params["switching_states"][state_n]
        s_abc = (2*s_an-s_bn-s_cn)/3
        s_bc = (s_bn - s_cn)/np.sqrt(3)
        self.s_mat = tf.constant(
            [[s_bc], [s_abc]], dtype=tf.float32)

        # define layer for Psi_dq
        self.hidden_1 = Dense(12, **dense_cfg)
        self.hidden_2 = Dense(8, **dense_cfg)
        self.psi_out = Dense(2, activation="linear",
                             name="Psi_out")

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'state_n': self.state_n,
            'calc_params': self.calc_params,
        })
        return config

    def call(self, inputs):

        i_d_k = inputs[:, 0]
        i_d_k = tf.reshape(i_d_k, [-1, 1])
        i_q_k = inputs[:, 1]
        i_q_k = tf.reshape(i_q_k, [-1, 1])
        i_k_scaled = inputs[:, :2]

        # build L_diff with derivation through GradientTape()
        with tf.GradientTape() as dpsi:
            dpsi.watch(i_k_scaled)
            i_norm = tf.reshape(tf.norm(i_k_scaled, axis=1), (-1, 1))
            i_all = tf.concat([i_k_scaled, i_norm], axis=1)
            psi = self.psi_out(self.hidden_2(self.hidden_1(i_all)))

        L_diff = dpsi.batch_jacobian(psi, i_k_scaled)
        inv_L = tf.linalg.inv(L_diff)

        # build Psi_dq
        psi_d = psi[:, :1]
        psi_q = psi[:, 1:2]
        inv_psi = tf.expand_dims(tf.concat([psi_q, -psi_d], axis=1), axis=-1)

        # calculate i_k1
        rhs = tf.concat([tf.fill([tf.shape(inv_psi)[0], 2, 2], -self.r_s),
                         tf.repeat(tf.expand_dims(
                             self.u_dc/2 * self.s_mat, axis=0), tf.shape(inv_psi)[0], axis=0),
                         inv_psi*self.w_el],
                        axis=-1)
        vec = tf.expand_dims(
            tf.concat([inputs[:, :4],
                       tf.ones_like(inputs[:, :1])],
                      axis=1), axis=-1)
        # batch-aware matmul
        i_k1 = inv_L @ rhs @ vec
        i_k1 = tf.squeeze(i_k1)

        # forward euler
        i_k1 = i_k1 * self.sample_time + i_k_scaled

        return i_k1

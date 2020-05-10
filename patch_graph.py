# coding: utf-8


import os
import tensorflow as tf


def read_constant_graph(pbfile, create_session=True):
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        from tensorflow.python.platform import gfile
        with gfile.FastGFile(pbfile, "rb") as f:
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")

    if create_session:
        session = tf.Session(graph=graph)
        return graph, session
    else:
        return graph


def write_constant_graph(session, output_names):

    constant_graph = tf.graph_util.convert_variables_to_constants(session,
        session.graph.as_graph_def(), output_names)
    graph_dir = './'
    # graph_path = os.path.normpath(os.path.abspath(graph_path))
    # graph_dir, graph_name = os.path.split(graph_path)
    # if not os.path.exists(graph_dir):
    #     os.makedirs(graph_dir)
    # if os.path.exists(graph_path):
    #     os.remove(graph_path)

    # tf.train.write_graph(constant_graph, graph_dir, graph_name)
    tf.train.write_graph(constant_graph, graph_dir, "pid-er_ATmodel_V2patched.pb", as_text=False)
    tf.train.write_graph(constant_graph, graph_dir, "pid-er_ATmodel_V2patched.pbtxt", as_text=True)


graph, sess = read_constant_graph("pid-er_ATmodel_V2.pb")

id_op = graph.get_operation_by_name("pid_output/Softmax")
er_op = graph.get_operation_by_name("enreg_output/BiasAdd")

with tf.variable_scope("patch"):
    # e_mean = 213.90352475881576
    # e_std = 108.05413626100672
    e_mean = 123.61015624208174
    e_std =  149.02813697719492
    e_rescaled = er_op.outputs[0] * e_std + e_mean

    id_t = id_op.outputs[0]
    id_ph_el_mu = id_t[:, 0:3]
    id_pi0 = id_t[:, 0:1] * 0.
    id_ch_nh = id_t[:, 3:5]
    id_am = id_t[:, 0:1] * 0.
    id_un = id_t[:, 5:6]
    id_concat = tf.concat([id_ph_el_mu, id_pi0, id_ch_nh, id_am, id_un], axis=1)

tf.identity(e_rescaled, name="output/regressed_energy")
tf.identity(id_concat, name="output/id_probabilities")

write_constant_graph(sess, ["output/id_probabilities", "output/regressed_energy"])


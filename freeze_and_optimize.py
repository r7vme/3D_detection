#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.tools.graph_transforms import TransformGraph
from keras import backend as K

from net.bbox_3D_net import bbox_3D_net

def freeze_and_optimize_session(session, keep_var_names=None, input_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        graph = tf.graph_util.remove_training_nodes(
            input_graph_def, protected_nodes=output_names)
        graph = tf.graph_util.convert_variables_to_constants(
            session, graph, output_names, freeze_var_names)
        transforms = [
          'remove_nodes(op=Identity)',
          'merge_duplicate_nodes',
          'strip_unused_nodes',
          'fold_constants(ignore_errors=true)',
          'fold_batch_norms',
         ]
        graph = TransformGraph(
            graph, input_names, output_names, transforms)
        return graph

if __name__ == "__main__":
  # freeze Keras session - converts all variables to constants
  K.set_learning_phase(0)
  model = bbox_3D_net((224, 224, 3), freeze_vgg=True)
  model.load_weights("3dmodel.h5")
  frozen_graph = freeze_and_optimize_session(K.get_session(),
                                             input_names=[inp.op.name for inp in model.inputs],
                                             output_names=[out.op.name for out in model.outputs])
  tf.train.write_graph(frozen_graph,
                       logdir="",
                       as_text=False,
                       name='box3d.pb')
  # To check graph in text editor
  #tf.train.write_graph(frozen_graph,
  #                     logdir="",
  #                     as_text=True,
  #                     name='box3d.pbtxt')

#!/usr/bin/env python
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluates a trained network."""

import argparse
import cv2
import logging
import numpy as np
import os
import re
import setproctitle
import skimage
import skimage.io
import skimage.transform
import sys

sys.path.append("$PATH/DeepUPE")

import time
import tensorflow as tf

import main.models as models
import main.utils as utils


logging.basicConfig(format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
log = logging.getLogger("train")
log.setLevel(logging.INFO)

os.environ["CUDA_VISIBLE_DEVICES"]="1"
def get_input_list(path):
  regex = re.compile(".*.(png|jpeg|jpg|tif|tiff)")
  if os.path.isdir(path):
    inputs = os.listdir(path)
    inputs = [os.path.join(path, f) for f in inputs if regex.match(f)]
    log.info("Directory input {}, with {} images".format(path, len(inputs)))

  elif os.path.splitext(path)[-1] == ".txt":
    dirname = os.path.dirname(path)
    with open(path, 'r') as fid:
      inputs = [l.strip() for l in fid.readlines()]
    inputs = [os.path.join(dirname, 'input', im) for im in inputs]
    log.info("Filelist input {}, with {} images".format(path, len(inputs)))
  elif regex.match(path):
    inputs = [path]
    log.info("Single input {}".format(path))
  return inputs


def main(args):
  # -------- Load params ----------------------------------------------------
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:

    checkpoint_path = tf.train.latest_checkpoint(args.checkpoint_dir)
    if checkpoint_path is None:
      log.error('Could not find a checkpoint in {}'.format(args.checkpoint_dir))
      return

    metapath = ".".join([checkpoint_path, "meta"])
    log.info('Loading graph from {}'.format(metapath))
    tf.train.import_meta_graph(metapath)

    model_params = utils.get_model_params(sess)

    # -------- Setup graph ----------------------------------------------------
  if not hasattr(models, "HDRNetCurves"):
    log.error("Model {} does not exist".format(params.model_name))
    return
  mdl = getattr(models, "HDRNetCurves")

  with tf.Graph().as_default() as graph:
    net_shape = model_params['net_input_size']
    t_fullres_input = tf.placeholder(tf.float32, (1, None, None, 3), name="input_high")
    t_lowres_input = tf.placeholder(tf.float32, (1, net_shape, net_shape, 3), name="input_low")

    with tf.variable_scope('inference'):
      prediction = mdl.inference(
          t_lowres_input, t_fullres_input, model_params, is_training=False)


    variable_map = {}
    for variable in tf.global_variables():
        tmp = variable.name.split('/')
        if tmp[0] == 'inference':
            print (variable.name.replace(':0', ''))
            variable_map[variable.name.replace(':0', '')] = variable

    saver = tf.train.Saver(var_list=variable_map)

    output = tf.cast(255.0*tf.squeeze(tf.clip_by_value(prediction, 0, 1)), tf.uint8, name="output")

    input_graph_def = graph.as_graph_def()

    output_node_names = "output"

    log.info('Restoring weights from {}'.format(checkpoint_path))

    with tf.Session(config=config) as sess:
      sess.run(tf.global_variables_initializer())
      saver.restore(sess, checkpoint_path)
      output_graph = "./de_dark_light.pb"
      output_graph_def = tf.graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names.split(","))

      with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('checkpoint_dir', default=None, help='path to the saved model variables')
  args = parser.parse_args()
  main(args)

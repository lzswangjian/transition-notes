#!/usr/bin/env python

"""A program to annotate a conll file with a tensorflow neural net parser."""

import os
import os.path
import time


def Eval(sess, num_actions, feature_sizes, domain_sizes,  embedding_dims):
  """
  """
  hidden_layer_sizes = map(int, FLAGS.hidden_layer_sizes.split(','))

  sink_documents = tf.placeholder(tf.string)
  sink = gen_parser_ops.document_sink(sink_documents,
      task_context=FLAGS.task_context,
      corpus_name=FLAGS.output)

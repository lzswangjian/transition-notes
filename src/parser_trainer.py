#!/usr/bin/env python

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('tf_master', '',
    'Tensorflow execution engine to connect to.')
flags.DEFINE_string('output_path', '', 'Top level for output.')
flags.DEFINE_string('task_context', )
flags.DEFINE_string('arg_prefix')
flags.DEFINE_string('params', '0', 'Unique identifier of parameter grid point.')
flags.DEFINE_string('training_corpus', )
flags.DEFINE_string('tuning_corpus', )
flags.DEFINE_string('word_embeddings', None,
    'Recordio containing pretrained word embeddings, will be '
    'loaded as the first embedding matrix.')
flags.DEFINE_integer('batch_size', 32)
flags.DEFINE_integer('beam_size', 10)
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs to train for.')
flags.DEFINE_integer('max_steps', 50, 'Max number of parser steps during a training step.')
flags.DEFINE_integer('report_every', 100,
    'Report cost and training accuracy every this many steps.')
flags.DEFINE_integer('checkpoint_every', 5000)
flags.DEFINE_float('learning_rate')
flags.DEFINE_float('momentum')


def RewriteContext():
  pass


def Eval(sess, parser, num_steps, best_eval_metric):
  """Evaluates a network and checkpoints it to disk.
  Args:
    sess: tensorflow session to use
    parser: graph builder containing all ops references.
    num_steps: number of training steps taken, for logging
    best_eval_metric: current best eval metric, to decide whether this model
      is the best so far
  """

def Train(sess, num_actions, feature_sizes, domain_sizes, embedding_dims):
  """Builds and trains the network.

  Args:
    sess:
    num_actions:
    feature_sizes:
    domain_sizes: number of possible ids in each feature vector.
    embedding_dims:
  """
  parser = graph_builder.GreedyParser(num_actions,
      feature_sizes,
      domain_sizes,
      embedding_dims,
      hidden_layer_sizes,
      seed=int(FLAGS.seed),
      gate_gradients=True,
      averaging_decay=FLAGS.averaging_decay,
      arg_prefix=FLAGS.arg_prefix)

  task_context = OutputPath('context')
  corpus_name = FLAGS.training_corpus
  parser.AddTraining()
  parser.AddEvaluation()
  parser.AddSaver(FLAGS.slim_model)

  # Saving graph
  logging.info('Training...')
  while num_epochs < FLAGS.num_epochs:
      pass
    


#!/usr/bin/env python

def BatchedSparseToDense(sparse_indices, output_size):
  """Batch compatible sparse to dense conversion.

  This is useful for one-hot coded target labels.

  Args:
    sparse_indices: [batch_size] tensor containing one index per batch
    output_size: needed in order to generate the correct dense output

  Returns:
    A [batch_size, output_size] dense tensor.
  """
  eye = tf.diag(tf.fill([output_size], tf.constant(1, tf.float32)))
  return tf.nn.embedding_lookup(eye, sparse_indices)


def EmbeddingLookupFeatures(params, sparse_features, allow_weights):
  """Computes embeddings for each entry of sparse features sparse_features.
  Args:
    parms: list of 2D tensors containing vector embeddings.
  """
  if not isinstance(params, list):
    params = [params]
  # Lookup embeddings
  sparse_features = tf.convert_to_tensor(sparse_features)
  indices, ids, weights = gen_parser_ops.unpack_sparse_features(sparse_features)
  embeddings = tf.nn.embedding_lookup(params, ids)

  if allow_weights:
    # Multiply by weights, reshaping to allow broadcast.
    broadcast_weights_shape = tf.concat(0, [tf.shape(weights), [1]])
    embeddings *= tf.reshape(weights, broadcast_weights_shape)

  # Sum embedding by index.
  return tf.unsorted_segment_sum(embeddings, indices, tf.size(sparse_features))


class GreedyParser(object):
  """Builds a Chen & Manning style greedy neural net parser.

  Builds a graph with an optional reader op connected at one end
  and operations needed to train the network on the other. Supports
  multiple network instantiations sharing the same parameters and
  network topology.

  The following named nodes are added to the training and eval networks:
    epochs: a tensor containing the current epoch number
    cost: a tensor containing the current training step cost
    gold_actions: a tensor containing actions from gold decoding
    feature_endpoints: a list of sparse feature vectors
    logits: output of the final layer before computing softmax
  The training network also contains:
    train_op: an op that executes a single training step
  """
  def __init__(self,
      num_actions,
      num_features,
      num_feature_ids,
      embedding_sizes,
      hidden_layer_sizes,
      seed=None,
      gate_gradients=False,
      use_locking=False,
      embedding_init=1.0,
      relu_init=1e-4,
      bias_init=0.2,
      softmax_init=1e-4,
      averaging_decay=0.9999,
      use_averaging=True,
      check_parameters=True,
      check_every=1,
      allow_feature_weights=False,
      only_train='',
      arg_prefix=None,
      **unused_kwargs):
    """Initalize the graph builder with parameters defining the network.

    Args:
      num_actions: int size of the set of parser actions.
      num_features: int list of dimensions of the feature vectors.
      num_feature_ids: int list of same length as num_features corresponding to
        the sizes of the input feature spaces.
      embedding_sizes: int list of same length as num_features of the desired
        embedding layer sizes.
    """

    self._num_actions = num_actions
    self._num_features = num_features

    # Parameters of the network w.r.t which training is done.
    self.params = {}

    # Other variables, w.r.t which no training is done, but which
    # we nonetheless need to save in order to capture the state of
    # the graph.
    self.variables = {}

    # Operations to initialize any nodes that require initialization.
    self.inits = {}

    # Training and eval releated nodes.
    self.training = {}
    self.evaluation = {}
    self.saver = None

    self._averaging = {}
    self._averaging_decay = averaging_decay

    self._pretrained_embeddings = {}

    with tf.name_scope('params') as self._param_scope:
      self._relu_bais_init = tf.constant_initializer(bias_init)

  @property
  def embedding_size(self):
    size = 0
    for i in range(self._feature_size):
      size += self._num_features[i] * self._embedding_sizes[i]
    return size

  def _AddParam(self, shape, dtype, name, initializer=None, return_average=False):
    """Add a model parameter w.r.t we expect to compute gradients.
    _AddParam creates both regular parameters (usually for tranining) and averaged
    nodes (usually for inference). It returns one or the other based on the
    'return_average' arg.
    """

    if name not in self.params:
      step = tf.cast(self.GetStep(), tf.float32)
      # Pull all parameters and their initalizing ops in their own scope
      # irrespective of current scope (training or eval).

  def GetStep(self):
    def OnesInitializer(shape, dtype=tf.float32):
      return tf.ones(shape, dtype)
    return self._AddVariable([], tf.int32, 'step', OnesInitializer)

  def _AddVariable(self, shape, dtype, name, initializer):
    if name in self.variables:
      return self.variables[name]
    self.variables[name] = tf.get_variable(name, shape, dtype, initializer)
    if initializer is not None:
      self.inits[name] = state_ops.init_variable(self.variables[name], initializer)

    return self.variables[name]

  def _ReluWeightInitializer(self):
    with tf.name_scope(self._param_scope):
      # Returns an intializer that generates tensors with a normal distribution
      return tf.random_normal_initializer(stddev=self._relu_init,
          seed=self._seed)

  def _EmbeddingMatrixInitializer(self, index, embedding_size):
    if index in self._pretrained_embeddings:
      return self._pretrained_embeddings[index]
    else:
      return tf.random_normal_initializer(
          stddev=self._embedding_init / embedding_size**.5,
          seed=self._seed)

  def _AddEmbedding(self, features, num_features, num_ids, embedding_size,
      index, return_average=False):
    pass

  def _BuildNetwork(self, feature_endpoints, return_average=False):
    """Builds a feed-forward part of the net given features as input.

    The network topology is already defined in the constructor, so multiple
    calls to BuildForward build multiple networks whose parameter are all
    shared. It is the source of the input features and the use of of the output
    that distinguishes each network.

    Args:
      feature_endpoints: tensors with input features to the network.
      return_average: whether to use moving averages as model parameters.

    Returns:
      logits: output of the final layer before computing softmax.
    """

    assert len(feature_endpoints) == self._feature_size

    # Creating embedding layer.
    embeddings = []
    for i in range(self._feature_size):
      embeddings.append(self._AddEmbedding(feature_endpoints[i],
        self._num_features[i], self._num_feature_ids[i], self._embedding_sizes[i],
        i, return_average=return_average))

    last_layer = tf.concat(1, embeddings)
    last_layer_size = self.embedding_size

    # Create ReLU layers.
    for i, hidden_layer_size in enumerate(self._hidden_layer_sizes):
      weights = self._AddParam(
          [last_layer_size, hidden_layer_size],
          tf.float32,
          'weights_%d' % i,
          self._ReluWeightInitializer(),
          return_average=return_average)

      bias = self._AddParam(
          [hidden_layer_size],
          tf.float32,
          'bias_%d' % i,
          self._relu_bais_init,
          return_average=return_average)

      last_layer = tf.nn.relu_layer(last_layer, weights, bias, name='layer_%d' % i)

      last_layer_size = hidden_layer_size

    # Create softmax layer.
    softmax_weight = self._AddParam(
        [last_layer_size, self._num_actions],
        tf.float32,
        'softmax_weight',
        tf.random_normal_initializer(stddev=self._softmax_init, seed=self._seed),
        return_average=return_average)

    softmax_bias = self._AddParam(
        [self._num_actions],
        tf.float32,
        'softmax_bias',
        tf.zeros_initializer,
        return_average=return_average)

    logits = tf.nn.xw_plus_b(layer, softmax_weight, softmax_bias, name='logits')
    return {'logits' : logits}

  def _AddGoldReader(self, task_context, batch_size, corpus_name):
    features, epoches, gold_actions = (
        gen_parser_ops.gold_parse_reader(task_context,
          self._feature_size,
          batch_size,
          corpus_name=corpus_name,
          arg_prefix=self._arg_prefix))

    return {'gold_actions' : tf.identity(gold_actions, name='gold_actions'),
            'epochs' : tf.identity(epochs, name='epochs'),
            'feature_endpoints': features}

  def _AddDecodedReader(self, task_context, batch_size, transition_scores, corpus_name):
    pass


  def _AddCostFunction(self, batch_size, gold_actions, logits):
    """Cross entropy plus L2 loss on weights and bias of the hidden layers.
    """
    dense_gloden = BatchedSparseToDense(gold_actions, self._num_actions)
    cross_entropy = tf.div(
      tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
        logits, dense_gloden)), batch_size)
    regularized_params = [tf.nn.l2_loss(p)
                          for k, p in self.params.items()
                          if k.startswith('weights') or k.startswith('bias')]
    l2_loss = 1e-4 * tf.add_n(regularized_params) if regularized_params else 0
    return {'cost': tf.add(cross_entropy, l2_loss, name='cost')}

  def AddPretrainedEmbeddings(self, index, embedding_path, task_context):
    """Embeddings at the given index will be set to pretrained values.
    """
    def _Initializer(shape, dtype=tf.float32):
      unused_dtype = dtype
      t = gen_parser_ops.word_embedding_initializer(
        vectors=embedding_path,
        task_context=task_context,
        embedding_init=self._embedding_init)

      t.set_shape(shape)
      return t

    self._pretrained_embeddings[index] = _Initializer

  def AddTraining(self, task_context, batch_size, learning_rate=0.1,
      decay_steps=4000, momentum=0.9, corpus_name='documents'):
    """
    """
    with tf.name_scope('training'):
      nodes = self.training
      nodes.update(self._AddGoldReader(task_context, batch_size, corpus_name))
      nodes.update(self._BuildForward(nodes['feature_endpoints'], return_average=False))
      nodes.update(self._AddCostFunction(batch_size, nodes['gold_actions'], nodes['logits']))

      # Add the optimizer


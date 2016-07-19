#include <algorithm>
#include <deque>
#include <map>
#include <memory>
#include <string>


class ParserStateWithHistory {
public:
  explicit ParserStateWithHistory(const ParserState &s) : state(s.Clone()) {}

  ParserStateWithHistory(const ParserStateWithHistory &next,
                         const ParserTransitionSystem &transitions, int32 slot,
                         int32 action, float score)
    : state(next.state->Clone()),
      slot_history(next.slot_history),
      action_history(next.action_history),
      score_history(next.score_history) {
    transitions.PerformAction(action, state.get());
    slot_history.push_back(slot);
    action_history.push_back(action);
    score_history.push_back(score);
  }

  std::unique_ptr<ParserState> state;
  std::vector<int32> slot_history;
  std::vector<int32> action_history;
  std::vector<float> score_history;
};


struct BatchStateOptions {
  // Maximum number of parser states in a beam.
  int max_beam_size;

  // Number of parallel sentences to decode.
  int batch_size;

  // Argument prefix for context parameters.
  string arg_prefix;

  // Corpus name to read from context inputs.
  string corpus_name;

  // Whether we allow weights in SparseFeatures protos.
  bool allow_feature_weights;

  // Whether beams should be considered alive until all states are final,
  // or until the gold path falls off.
  bool continue_until_all_final;

  // Whether to skip to a new sentence after each training step.
  bool always_start_new_sentences;

  // Parameter for deciding which tokens to score.
  string scoring_type;
};


/*!
 * \brief Encapsulates the environment needed to parse with a beam, keeping a 
 * record of path histories.
 */
class BeamState {
public:
  // The agenda is keyed by a tuple that is the score followed by an int
  // that is -1 if the path coincides with the gold path and 0 otherwise.
  // The lexicographic ordering of the keys therefore ensures that for all
  // pahts sharing the same score, the gold path will always be at the
  // bottom.
  typedef std::pair<double, int> KeyType;
  typedef std::multimap<KeyType, std::unique_ptr<ParserStateWithHistory>> AgendaType;
  typedef std::pair<KeyType, std::unique_ptr<ParserStateWithHistory>> AgendaItem;
  typedef Eigen::Tensor<float, 2, Eigen::RowMajor, Eigen::DenseIndex> ScoreMatrixType;

  // The beam can be
  //   - ALIVE: parsing is still active, features are being output for at least
  //     some slots in the beam.
  //   - DYING:
  //   - DEAD:
  enum State { ALIVE = 0, DYING = 1, DEAD = 2 };

  explicit BeamState(const BatchStateOptions &options) : options_(options) {}

  void Reset() {
    if (options_.always_start_new_sentences ||
        gold_ == nullptr || transition_system_->IsFinalState(*gold_)) {
      AdvanceSentence();
    }
    slots.clear();
    if (gold_ == nullptr) {
      state_ = DEAD; // EOF has been reached.
    } else {
      gold_->set_is_gold(true);
      slots_.emplace(KeyType(0.0, -1),
                     std::unique_ptr<ParserStateWithHistory>(new ParserStateWithHistory(*gold_)));
      state_ = ALIVE;
    }
  }

  // Check whether all states in the beam have reached final state.
  void UpdateAllFinal() {
    all_final_ = true;
    for (const AgendaItem &item : slots_) {
      if (!transition_system_->IsFinalState(*item.second->state)) {
        all_final_ = false;
        break;
      }
    }
    if (all_final_) {
      state_ = DEAD;
    }
  }

  // This method updates the beam. For all elements of the beam, all allowed transitions
  // are scored into a new beam. The beam size is capped by discarding the lowest scoring
  // slots at any given time. There is one exception to this process: the gold path is forced
  // to remain in the beam at all times, even if it scores low. This is to ensure that the gold
  // path can be used for training at the moment it would otherwise fall off (can be absent from)
  // the beam.
  void Advance(const ScoreMatrixType &scores) {
    if (state_ == DYING) state_ = DEAD;

    // When to stop advancing
    if (!IsAlive() || gold_ == nullptr) return;

    AdvanceGold();

    const int score_rows = scores.dimension(0);
    const int num_actions = scores.dimension(1);

    // Advance beam.
    AgendaType previous_slots;
    previous_slots.swap(slots_);
    CHECK_EQ(state_, ALIVE);

    int slot = 0;
    for (AgendaItem &item : previous_slots) {
      {
        ParseState *current = item.second->state.get();
        VLOG(2) << "Slot: " << slot;
      }
      if (!transition_system_->IsFinalState(*item.second->state)) {
        // Not a final state.
        for (int action = 0; action < num_actions; ++action) {
          // Is action allowed?
          if (!transition_system_->IsAllowedAction(action, *item.second->state)) {
            continue;
          }
          CHECK_LT(slot, score_rows);
          MaybeInsertWithNewAction(item, slot, scores(slot, action), action);
          PruneBeam();
        }
      } else {
        // Final state: no need to advance.
        MaybeInsert(&item);
        PruneBeam();
      }
      ++slot;
    }
    UpdateAllFinal();
  }

  void PopulateFeatureOutputs(std::vector<std::vector< std::vector<SparseFeatures> > > *features) {
  }

  int BeamSize() const { return slots_.size(); }

  bool IsAlive() const { return state_ == ALIVE; }

  bool IsDead() const { return state_ == DEAD; }

  bool AllFinal() const { return all_final_; }

  // The current contents of the beam.
  AgendaType slots_;

  // Which batch this refers to.
  int beam_id_ = 0;

  SentenceBatch *sentence_batch_ = nullptr;

  const TermFrequencyMap *label_map_ = nullptr;

  const ParserTransitionSystem *transition_system_ = nullptr;

  // Feature extractor.
  const ParserEmbeddingFeatureExtractor *features_ = nullptr;

  WorkspaceSet *workspace_ = nullptr;

  WorkspaceRegistry *workspace_registry_ = nullptr;

  // ParserState used to get gold actions.
  std::unique_ptr<ParserState> gold_;

private:
  // Creates a new ParserState if there's another sentence to be read.
  void AdvanceSentence() {
    gold_.reset();
  }

  void AdvanceGold() {
    gold_action_ = -1;
    if (!transition_system_->IsFinalState(*gold_)) {
      gold_action_ = transition_system_->GetNextGoldAction(*gold_);
      if (transition_system_->IsAllowedAction(gold_action_, *gold_)) {
        // In cases where the gold annotation is incompatible with the
        // transition system, the action returned as gold might be not allowed.
        transition_system_->PerformAction(gold_action_, gold_.get());
      }
    }
  }

  // Removes the first non-gold beam element if the beam is larger than
  // the maximum beam size. If the gold element was at the bottom of the
  // beam, stes the beam state to DYING, otherwise leaves the state alone.
  void PruneBeam() {
    if (static_cast<int>(slots_.size()) > options_.max_beam_size) {
      auto bottom = slots_.begin();
      if (!options_.continue_until_all_final &&
          bottom->second->state->is_gold()) {
        state_ = DYING;
        ++bottom;
      }
      slots_erase(bottom);
    }
  }

  // Inserts an item in the beam if
  //   - the item is gold,
  //   - the beam is not full, or
  //   - the item's new score is greater than the lowest score in the beam
  //     after the score has been incremented by given delta_score.
  // Inserted items have slot, delta_score and action appended to their history.
  void MaybeInsertWithNewAction(const AgendaItem &item, const int slot,
                                const double delta_score, const int action) {
    const double score = item.first.first + delta_score;
    const bool is_gold =
      item.second->state->is_gold() && action == gold_action_;
    if (is_gold || static_cast<int>(slots_.size()) < options_.max_beam_size ||
        score > slots_.begin()->first.first) {
      const KeyType key{score, -static_cast<int>(is_gold)};
      slots_.emplace(key, std::unique_ptr<ParserStateWithHistory>(
                              new ParserStateWithHistory(
                                  *item.second, *transition_system_, slot,
                                  action, delta_score)))
        ->second->state->set_is_gold(is_gold);
    }
  }

  // Inserts an item in the beam if
  //   - the item is gold,
  //   - the beam is not full, or
  //   - the item's new score is greater than the lowest score in the beam.
  // The history of inserted items is left untouched.
  void MaybeInsert(AgendaItem *item) {
    const bool is_gold = item->second->state->is_gold();
    const double score = item->first.first;
    if (is_gold || static_cast<int>(slots_.size()) < options_.max_beam_size ||
        score > slots_.begin()->first.first) {
      slots_.emplace(item->first, std::move(item->second));
    }
  }

  // Limits the number of slots on the beam.
  const BatchStateOptions &options_;

  it gold_action_ = -1;
  State state_ = ALIVE;
  bool all_final_ = false;
};

// Encapsulates the state of a batch of beams. It is an object of this
// type that will persist through repeated Op evaluations as the multiple
// steps are computed in sequence.
class BatchState {
public:
  explicit BatchState(const BatchStateOptions &options)
    : options_(options), features_(options.arg_prefix) {}

  ~BatchState() { }

  void Init(TaskContext *task_context) {
    // Create sentence bath
  }

  void ResetBeams() {
  }

  void ResetOffsets() {
  }

  void AdvanceBeam(const int beam_id,
                   const TTypes<float>::ConstMatrix &scores) {
  }

  void UpdateOffsets() {
  }

  tensorflow::Status PopulateFeatureOutputs(OpKernelContext *context) {
  }

  int GetOffset(const int step, const int beam_id) const {
  }

  int FeatureSize() const {
  }

  int NumActions() const {
  }

  int BatchSize() const {
  }

  int Epoch() const { return epoch_; }

  const string &ScoringType() const { return options_.scoring_type; }

private:
  const BatchStateOptions options_;

  // How many times the document source has been rewound.
  int epoch_ = 0;

  // Batch of sentences, and the corresponding parser states.
  std::unique_ptr<SentenceBatch> sentence_batch_;

  // Transition system.
  std::unique_ptr<ParserTransitionSystem> transition_system_;

  // Label map for transition system..
  const TermFrequencyMap *label_map_;

  // Typed feature extractor for embeddings.
  ParserEmbeddingFeatureExtractor features_;

  // Batch: WorkspaceSet objects.
  std::vector<WorkspaceSet> workspaces_;

  // Internal workspace registry for use in feature extraction.
  WorkspaceRegistry workspace_registry_;

  std::deque<BeamState> beams_;
  std::vector<std::vector<int>> beam_offsets_;

  // Keeps track of the slot offset of each step.
  std::vector<int> step_offsets_;
};


// Creates a BeamState and hooks it up with a parser. This Op needs to
// remain alive for the duration of the parse.
class BeamParseReader : public OpKernel {
public:
  explicit BeamParseReader(OpKernelConstruction *context)
    : OpKernel(context) {
    string file_path;
    int feature_size;
    BatchStateOptions options;
  }

  void Compute(OpKernelContext *context) override {
  }

private:
  mutext mu_;

  std::unique_ptr<BatchState> batch_state_;
};

// Updates the beam based on incoming scores and outputs new feature vectors
// based on the updated beam.
class BeamParser : public OpKernel {
public:
  explicit BeamParser(OpKernelConstruction *context)
    :  OpKernel(context) {
    int feature_size;
    OP_REQUIRES_OK(context, context->GetAttr("feature_size", &feature_size));

    // Set expected signature.
    std::vector<DataType> output_types(feature_size, DT_STRING);
    output_types.push_back(DT_INT64);
    output_types.push_back(DT_BOOL);
    OP_REQUIRES_OK(context,
                   context->MatchSignature({DT_INT64, DT_FLOAT}, output_types));
  }

  void Compute(OpKernelContext *context) override {
    BatchState *batch_state =
      reinterpret_cast<BatchState *>(context->input(0).scalar<int64>());
    const TTypes<float>::ConstMatrix scores = context->input(1).matrix<float>();

    // In AdvanceBeam we use beam_offsets_[beam_id] to determine the slice of
    // scores that should be used for advancing, but beam_offsets_[beam_id]
    // exists for beams that have a sentence loaded.
    const int batch_size = batch_state->BatchSize();
    for (int beam_id = 0; beam_id < batch_size; ++beam_id) {
      batch_state->AdvanceBeam(beam_id, scores);
    }
    batch_state->UpdateOffsets();

    // Forward the beam state unmodified.
    Tensor *output;
    const int feature_size = batch_state->FeatureSize();
    OP_REQUIRES_OK(context, context->allocate_output(feature_size,
                                                     TensorShape({}), &output));
    output->scalar<int64>()() = context->input(0).scalar<int64>()();

    // Output the new features of all the slots in all the beams.
    OP_REQUIRES_OK(context, batch_state->PopulateFeatureOutputs(context));

    // Output whether the beams are alive.
    OP_REQUIRES_OK(
        context, context->allocate_output(feature_size + 1,
                                          TensorShape({batch_size}), &output));
    for (int beam_id = 0; beam_id < batch_size; ++beam_id) {
      output->vec<bool>()(beam_id) = batch_state->Beam(beam_id).IsAlive();
    }
  }
};


// Extracts the paths for the elements of the current beams and returns
// indices into a scoring matrix that is assumed to have been
// constructed along with the beam search.
class BeamParserOutput : public OpKernel {
public:
  explicit BeamParserOutput(OpKernelConstruction *context) : OpKernel(context) {
  }

  void Compute(OpKernelContext *context) override {
    BatchState *batch_state =
      reinterpret_cast<BatchState *>(context->input(0).scalar<int64>()());

    // Vectors for output.
    //
    // Each step of each batch: path gets its index computed and a
    // unique path id assigned.
    std::vector<int32> indices;
    std::vector<int32> path_ids;

    // Each unique path gets a batch id and a slot (in the beam)
    // id. These are in effect the row and column of the final
    // 'logits' matrix going to CrossEntropy.
    std::vector<int32> beam_ids;
    std::vector<int32> slot_ids;

    // To compute the cross entropy we also need the slot id of the
    // gold path , one per batch.
    std::vector<int32> gold_slot(batch_size, -1);

    std::vector<float> path_scores;

    int path_id = 0;
    for (int beam_id = 0; beam_id < batch_size; ++beam_id) {
      // This occurs at the end of the corpus, when there aren't enough
      // sentences to fill the batch.
      if (batch_state->Beam(beam_id).gold_ == nullptr) continue;

      int slot = 0;
      for (const auto &item : batch_state->Beam(beam_id).slots_) {
        beam_ids.push_back(beam_id);
        slot_ids.push_back(slot);
        path_scores.push_back(item.first.first);

        if (item.second->state->is_gold()) {
          CHECK_EQ(gold_slot[beam_id], -1);
          gold_slot[beam_id] = slot;
        }
      }
    }
  }
};

// Computes eval metrics for the best path in the input beams.
class BeamEvalOutput : public OpKernel {
};

#include <algorithm>
#include <deque>
#include <map>
#include <memory>
#include <string>

#include "utils/utils.h"
#include "sentence_batch.h"
#include "parser/parser_state.h"
#include "utils/task_context.h"
#include "utils/work_space.h"
#include "feature/embedding_feature_extractor.h"
#include "utils/shared_store.h"
#include "parser/arc_standard_transitions.cc"
#include "model/model_predict.cc"

typedef Matrix ScoreMatrixType;

/*!
 * \brief ParserStateWithHistory
 * Wraps ParserState so that the history of transitions (actions
 * performed and the beam slot they were performed in) are recorded.
 */
class ParserStateWithHistory {
public:
  explicit ParserStateWithHistory(const ParserState &s) : state(s.Clone()) {}
  
  // New state obtained by cloning the given state and applying the given action.
  // The given beam slot and action are appended to the history.
  ParserStateWithHistory(const ParserStateWithHistory &next,
                         const ParserTransitionSystem &transitions, int32_t slot,
                         int32_t action, float score)
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
  std::vector<int32_t> slot_history;
  std::vector<int32_t> action_history;
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
  typedef std::pair<const KeyType, std::unique_ptr<ParserStateWithHistory>> AgendaItem;

  // The beam can be
  //   - ALIVE: parsing is still active, features are being output for at least
  //     some slots in the beam.
  //   - DYING: features should be output for this beam only one more time, then
  //     the beam will be DEAD. This state is reached when the gold path falls out
  //     of the beam and features have to be output one last time.
  //   - DEAD: parsing is not active, features are not being output and the no actions
  //     are taken on the states.
  enum State { ALIVE = 0, DYING = 1, DEAD = 2 };

  explicit BeamState(const BatchStateOptions &options) : options_(options) {}

  void Reset() {
    if (options_.always_start_new_sentences ||
        gold_ == nullptr || transition_system_->IsFinalState(*gold_)) {
      AdvanceSentence();
    }
    slots_.clear();
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
    for (auto &item : slots_) {
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
  void Advance(ScoreMatrixType &scores) {
    if (state_ == DYING) state_ = DEAD;

    // When to stop advancing
    if (!IsAlive() || gold_ == nullptr) return;

    AdvanceGold();

    const int score_rows = scores.row();
    const int num_actions = scores.col();

    // Advance beam.
    AgendaType previous_slots;
    previous_slots.swap(slots_);
    CHECK_EQ(state_, ALIVE);

    int slot = 0;
    for (AgendaItem &item : previous_slots) {
      {
        ParserState *current = item.second->state.get();
        VLOG(2) << "Slot: " << slot;
        VLOG(2) << "Parser state: " << current->ToString();
        VLOG(2) << "Parser state cumulative score:  " << item.first.first << " "
                << (item.first.second < 0 ? "golden" : "");
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

  void PopulateFeatureOutputs(vector<vector<vector<SparseFeatures>>> *features) {
    for (AgendaItem &item : slots_) {
      vector<vector<SparseFeatures> > f =
        features_->ExtractSparseFeatures(*workspace_, *item.second->state);
      for (size_t i = 0; i < f.size(); ++i) {
        (*features)[i].push_back(f[i]);
      }
    }
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
    if (sentence_batch_->AdvanceSentence(beam_id_)) {
      gold_.reset(new ParserState(sentence_batch_->sentence(beam_id_),
                                  transition_system_->NewTransitionState(true),
                                  label_map_));
      workspace_->Reset(*workspace_registry_);
      features_->Preprocess(workspace_, gold_.get());
    }
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
      slots_.erase(bottom);
    }
  }

  // Inserts an item in the beam if
  //   - the item is gold,
  //   - the beam is not full, or
  //   - the item's new score is greater than the lowest score in the beam
  //     after the score has been incremented by given delta_score.
  // Inserted items have slot, delta_score and action appended to their history.
  void MaybeInsertWithNewAction(AgendaItem &item, const int slot,
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

  int gold_action_ = -1;
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

  ~BatchState() { SharedStore::Release(label_map_); }

  void Init(TaskContext *task_context) {
    // Create sentence batch
    sentence_batch_.reset(new SentenceBatch(BatchSize(),
                                            options_.corpus_name));
    sentence_batch_->Init(task_context);

    // Create transition system.
    // transition_system_.reset(ParserTransitionSystem::Create(task_context->Get(
    //  options_.arg_prefix+"_transition_system", "arc-standard")));
    transition_system_.reset(new ArcStandardTransitionSystem());
    transition_system_->Setup(task_context);
    transition_system_->Init(task_context);

    // Create label map.
    string label_map_path =
      TaskContext::InputFile(*task_context->GetInput("label-map"));
    label_map_ = SharedStoreUtils::GetWithDefaultName<TermFrequencyMap>(label_map_path, 0, 0);

    // Setup features.
    features_.Setup(task_context);
    features_.Init(task_context);
    features_.RequestWorkspaces(&workspace_registry_);

    // Create workspaces.
    workspaces_.resize(BatchSize());

    // Create beams.
    beams_.clear();
    for (int beam_id = 0; beam_id < BatchSize(); ++beam_id) {
      beams_.emplace_back(options_);
      beams_[beam_id].beam_id_ = beam_id;
      beams_[beam_id].sentence_batch_ = sentence_batch_.get();
      beams_[beam_id].transition_system_ = transition_system_.get();
      beams_[beam_id].label_map_ = label_map_;
      beams_[beam_id].features_ = &features_;
      beams_[beam_id].workspace_ = &workspaces_[beam_id];
      beams_[beam_id].workspace_registry_ = &workspace_registry_;
    }
  }

  void ResetBeams() {
    for (BeamState &beam : beams_) {
      beam.Reset();
    }

    // Rewind if no states remain in the batch (we need to rewind the corpus).
    if (sentence_batch_->size() == 0) {
      ++epoch_;
      VLOG(2) << "Starting epoch " << epoch_;
      sentence_batch_->Rewind();
    }
  }

  // Resets the offset vectors required for a single run because we're starting
  // a new matrix of scores.
  void ResetOffsets() {
    beam_offsets_.clear();
    step_offsets_ = {0};
    UpdateOffsets();
  }

  void AdvanceBeam(const int beam_id, ScoreMatrixType &scores) {
    const int offset = beam_offsets_.back()[beam_id];
    // slice scores.

    beams_[beam_id].Advance(scores);
  }

  void UpdateOffsets() {
    beam_offsets_.emplace_back(BatchSize() + 1, 0);
    vector<int> &offsets = beam_offsets_.back();
    for (int beam_id = 0; beam_id < BatchSize(); ++beam_id) {
      // If the beam is ALIVE or DYING (but not DEAD), we want to
      // output the activations.
      const BeamState &beam = beams_[beam_id];
      const int beam_size = beam.IsDead() ? 0 : beam.BeamSize();
      offsets[beam_id + 1] = offsets[beam_id] + beam_size;
    }
    const int output_size = offsets.back();
    step_offsets_.push_back(step_offsets_.back() + output_size);
  }

  bool PopulateFeatureOutputs(TaskContext *context) {
    const int feature_size = FeatureSize();
    vector<vector<vector<SparseFeatures> > > features(feature_size);
    for (int beam_id = 0; beam_id < BatchSize(); ++beam_id) {
      if (!beams_[beam_id].IsDead()) {
        beams_[beam_id].PopulateFeatureOutputs(&features);
      }
    }
    /*for (size_t i = 0; i < features.size(); ++i) {
      for (size_t j = 0; j < features[i].size(); ++j) {
        for (size_t k = 0; k < features[i][j].size(); ++k) {
          LOG(INFO) << features[i][j][k].description_[0];
        }
      }
    }*/
    return true;
  }
  
  // Returns the offset (i.e. row number) of a particular beam at a
  // particular step in the final concatenated score matrix.
  int GetOffset(const int step, const int beam_id) const {
    return step_offsets_[step] + beam_offsets_[step][beam_id];
  }

  int FeatureSize() const {
    return features_.embedding_dims().size();
  }

  int NumActions() const {
    return transition_system_->NumActions(label_map_->Size());
  }

  int BatchSize() const {
    return options_.batch_size;
  }

  const BeamState &Beam(const int i) const { return beams_[i]; }

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
class BeamParseReader {
public:
  explicit BeamParseReader(TaskContext *context) {
    BatchStateOptions options;
    options.max_beam_size = 25;
    options.batch_size = 32;
    options.corpus_name = "training-corpus";
    options.arg_prefix = "beam_parser";
    
    // Create batch state.
    batch_state_.reset(new BatchState(options));
    batch_state_->Init(context);

    // Check number of feature groups matches the task context.
    const int required_size = batch_state_->FeatureSize();
  }

  void Compute(TaskContext *context) {
    // Write features.
    batch_state_->ResetBeams();
    batch_state_->ResetOffsets();
    batch_state_->PopulateFeatureOutputs(context);

    // Forward the beam state vector.
    const int feature_size = batch_state_->FeatureSize();
    VLOG(2) << "feature_size [" << feature_size << "]";

    // Ouput number of epochs.
    VLOG(2) << "epoch:" << batch_state_->Epoch();
  }

public:
  std::unique_ptr<BatchState> batch_state_;
};

// Updates the beam based on incoming scores and outputs new feature vectors
// based on the updated beam.
class BeamParser {
public:
  explicit BeamParser(TaskContext *context) {
    string symbol = "mxnet/greedy-symbol.json";
    string params = "mxnet/greedy-0009.params";
    int max_batch_size_ = 32;
    global_model_ = new Model(max_batch_size_);
    global_model_->Load(symbol, params);
    global_model_->Init(context);
    parser_reader_.reset(new BeamParseReader(context));
  }

  void Compute(TaskContext *context) {
    // Read Sentence & Populate Features.
    parser_reader_->Compute(context);

    // Extract BeamState Features.
    BatchState *batch_state = parser_reader_->batch_state_.get();

    // Compute BeamState Score via GlobalNormalization Model.
    ComputeMatrix();

    // In AdvanceBeam we use beam_offsets_[beam_id] to determine the slice of
    // scores that should be used for advancing, but beam_offsets_[beam_id]
    // exists for beams that have a sentence loaded.
    const int batch_size = batch_state->BatchSize();
    for (int beam_id = 0; beam_id < batch_size; ++beam_id) {
      batch_state->AdvanceBeam(beam_id, scores_matrix_);
    }
    batch_state->UpdateOffsets();

    // Forward the beam state unmodified.
    const int feature_size = batch_state->FeatureSize();

    // Output the new features of all the slots in all the beams.
    batch_state->PopulateFeatureOutputs(context);

    // Output whether the beams are alive.
    for (int beam_id = 0; beam_id < batch_size; ++beam_id) {
      batch_state->Beam(beam_id).IsAlive();
    }
  }
  
  void ComputeMatrix() {
    vector<string> feature_names = {"feature_0_data", "feature_1_data", "feature_2_data"};
    vector<int> feature_sizes = {20, 20, 12};
    // padding.
    /*int sentence_size = this->batch_size();
    int max_batch_size = this->max_batch_size();
    for (int pad_size = sentence_size; pad_size < max_batch_size; ++pad_size) {
      for (size_t k = 0; k < feature_sizes.size(); ++k) {
        for (int fsize = 0; fsize < feature_sizes[k]; ++fsize) {
          feature_outputs_[k].push_back(0);
        }
      }
    }*/
    
    global_model_->DoPredict(feature_outputs_, feature_names, feature_sizes, &scores_matrix_);
  }

private:
  std::unique_ptr<BeamParseReader> parser_reader_;
  Model *global_model_;
  vector<vector<float>> feature_outputs_;
  ScoreMatrixType scores_matrix_;
};


// Extracts the paths for the elements of the current beams and returns
// indices into a scoring matrix that is assumed to have been
// constructed along with the beam search.
class BeamParserOutput {
public:
  explicit BeamParserOutput(TaskContext *context) {
  }

  void Compute(TaskContext *context) {
    BatchState *batch_state = nullptr;
    int batch_size = 32;  // Get value from context.
    int num_actions = 10;

    // Vectors for output.
    //
    // Each step of each batch: path gets its index computed and a
    // unique path id assigned.
    std::vector<int32_t> indices;
    std::vector<int32_t> path_ids;

    // Each unique path gets a batch id and a slot (in the beam)
    // id. These are in effect the row and column of the final
    // 'logits' matrix going to CrossEntropy.
    std::vector<int32_t> beam_ids;
    std::vector<int32_t> slot_ids;

    // To compute the cross entropy we also need the slot id of the
    // gold path , one per batch.
    std::vector<int32_t> gold_slot(batch_size, -1);

    // For good measure we also output the path scores as computed by
    // the beam decoder, so it can be compared in tests with the path
    // scores computed via the indices in TF. This has the same length
    // as beam_ids and slot_ids.
    std::vector<float> path_scores;

    // The scores tensor has, conceptually, four dimensions: 1. number of steps,
    // 2. batch size, 3. number of paths on the beam at that step, and 4. the number
    // of actions scored. However this is not a true tensor since the size of the
    // beam at each step may not be equal among all steps and among all batches.
    // Only the batch size and number of actions are fixed.
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

        for (size_t step = 0; step < item.second->slot_history.size(); ++step) {
          const int step_beam_offset = batch_state->GetOffset(step, beam_id);
          const int slot_index = item.second->slot_history[step];
          const int action_index = item.second->action_history[step];
          indices.push_back(num_actions * (step_beam_offset + slot_index) +
                            action_index);
          path_ids.push_back(path_id);
        }
        ++slot;
        ++path_id;
      }
    }
  }
};

// Computes eval metrics for the best path in the input beams.
class BeamEvalOutput {
public:
  explicit BeamEvalOutput(TaskContext *context) {
  }

  void Compute(TaskContext *context) {
    int num_tokens = 0;
    int num_correct = 0;
    int all_final = 0;
    BatchState *batch_state = nullptr;
    const int batch_size = batch_state->BatchSize();
    vector<Sentence> documents;
    for (int beam_id = 0; beam_id < batch_size; ++beam_id) {
      if (batch_state->Beam(beam_id).gold_ != nullptr &&
          batch_state->Beam(beam_id).AllFinal()) {
        ++all_final;
        const auto &item = *batch_state->Beam(beam_id).slots_.rbegin();
        // ComputeTokenAccuracy();
        documents.push_back(item.second->state->sentence());
        item.second->state->AddParseToDocument(&documents.back());
      }
    }
  }

private:
  void ComputeTokenAccuracy(const ParserState &state,
                            const string &scoring_type,
                            int *num_tokens, int *num_correct) {
  }
};

#include <deque>
#include "utils/utils.h"
#include "sentence_batch.h"
#include "parser/parser_state.h"
#include "utils/task_context.h"
#include "utils/work_space.h"
#include "feature/embedding_feature_extractor.h"
#include "utils/shared_store.h"
#include "parser/arc_standard_transitions.cc"
#include "model/model_predict.cc"

class ParsingReader {
public:
    explicit ParsingReader(TaskContext *context) {
        string corpus_name;
        arg_prefix_ = "parser";
        corpus_name = "training-corpus";
        max_batch_size_ = 32;

        // Set up the batch reader.
        sentence_batch_.reset(
                new SentenceBatch(max_batch_size_, corpus_name));
        sentence_batch_->Init(context);


        // Set up the parsing features and transition system.
        states_.resize(max_batch_size_);
        workspaces_.resize(max_batch_size_);
        features_.reset(new ParserEmbeddingFeatureExtractor(arg_prefix_));
        features_->Setup(context);
        transition_system_.reset(new ArcStandardTransitionSystem());
        transition_system_->Setup(context);

        features_->Init(context);
        features_->RequestWorkspaces(&workspace_registry_);

        transition_system_->Init(context);

        string label_map_path = "/Users/Sheng/WorkSpace/transition-notes/label-map";
                // TaskContext::InputFile(*context->GetInput("label_map"));
        label_map_ = SharedStoreUtils::GetWithDefaultName<TermFrequencyMap>(label_map_path, 0, 0);

        // Checks number of feature groups matches the task context.

    }

    ~ParsingReader() { SharedStore::Release(label_map_); }

    // Creates a new ParserState if there's another sentence to be read.
    virtual void AdvanceSentence(int index) {

        states_[index].reset();
        if (sentence_batch_->AdvanceSentence(index)) {
            states_[index].reset(new ParserState(
                    sentence_batch_->sentence(index),
                    transition_system_->NewTransitionState(true),
                    label_map_));
            workspaces_[index].Reset(workspace_registry_);
            features_->Preprocess(&workspaces_[index], states_[index].get());
        }
    }

    virtual void Compute() {
        mu_.lock();

        // Advances states to the next position.
        PerformActions();

        // Advances any final states to the next sentences.
        for (int i = 0; i < max_batch_size_; ++i) {
            if (state(i) == nullptr) continue;

            // Switches to the next sentence if we're at a final state.
            while (transition_system_->IsFinalState(*state(i))) {
                VLOG(2) << "Advance sentence" << i;
                AdvanceSentence(i);
                if (state(i) == nullptr) break;  // EOF has been reached.
            }
        }

        // Rewinds if no states remain in the batch (we need to re-wind the corpus).
        if (sentence_batch_->size() == 0) {
            ++num_epochs_;
            LOG(INFO) << "Starting epoch " << num_epochs_;
            sentence_batch_->Rewind();
            for (int i = 0; i < max_batch_size_; ++i) {
                AdvanceSentence(i);
            }
        }

        // Create the outputs for each feature space.
        feature_outputs_.clear();
        feature_outputs_.resize(features_->NumEmbeddings());

        // Populate feature outputs.
        for (int i = 0, index = 0; i < max_batch_size_; ++i) {
            if (states_[i] == nullptr) continue;

            // Extract features from the current parser state, and fill up the
            // available batch slots.
            vector<vector<SparseFeatures>> features =
                    features_->ExtractSparseFeatures(workspaces_[i], *states_[i]);

            for (size_t j = 0; j < features.size(); ++j) {
                for (size_t k = 0; k < features[j].size(); ++k) {
                    feature_outputs_[j].push_back(features[j][k].id_[0]);
                }
            }
            ++index;
        }

        // Return the number of epochs.

        // Create outputs specific to this reader.
        AddAdditionalOutputs();
        mu_.unlock();
        LOG(INFO) << "feature outputs[0] size " << feature_outputs_[0].size();
    }

protected:
    // Performs any relevant actions on the parser state, typically either the gold
    // action or a predicated action from decoding.
    virtual void PerformActions() = 0;

    virtual void AddAdditionalOutputs() const = 0;

    // Accessors.
    int max_batch_size() const { return max_batch_size_; }

    int batch_size() const { return sentence_batch_->size(); }

    ParserState *state(int i) const { return states_[i].get(); }

    const ParserTransitionSystem &transition_system() const {
        return *transition_system_.get();
    }

private:
    TaskContext task_context_;

    string arg_prefix_;

    // mutex to synchronize access to Compute.
    mutex mu_;

    // How many times the document source has been rewinded.
    int num_epochs_ = 0;

    // How many sentences this op can be processing at any given time.
    int max_batch_size_ = 1;

    // Number of feature groups in the brain parser features.
    int feature_size_;

    // Batch of sentences, and the corresponding parser states.
    std::unique_ptr<SentenceBatch> sentence_batch_;

    std::vector<std::unique_ptr<ParserState>> states_;

    // Batch: WorkspaceSet objects.
    std::vector<WorkspaceSet> workspaces_;

    const TermFrequencyMap *label_map_;

    std::unique_ptr<ParserTransitionSystem> transition_system_;

    std::unique_ptr<ParserEmbeddingFeatureExtractor> features_;

    WorkspaceRegistry workspace_registry_;

public:
    vector<vector<float> > feature_outputs_;
};

class GoldParseReader : public ParsingReader {
public:
    explicit GoldParseReader(TaskContext *context)
            : ParsingReader(context) {
    }

private:
    // Always performs the next gold action for each state.
    void PerformActions() override {
        for (int i = 0; i < max_batch_size(); ++i) {
            if (state(i) != nullptr) {
                transition_system().PerformAction(
                        transition_system().GetNextGoldAction(*state(i)), state(i));
            }
        }
    }

    // Adds the list of gold actions for each state as an additional output.
    void AddAdditionalOutputs() const override {
    }
};

/*!
 * \brief DecodedParseReader parses sentences using transition scores computed by neural
 * network. This op additionally computes a token correctness evaluation metric which can
 * be used to select hyperparameter settings and training stopping point.
 *
 * The notion of correct token is determined by the transition system. e.g. a tagger will
 * return POS tag accuracy, while an arc-standard parser will return UAS.
 */
class DecodedParseReader : public ParsingReader {
public:
    explicit DecodedParseReader(TaskContext *context)
            : ParsingReader(context) {
        string symbol = "/Users/Sheng/WorkSpace/transition-notes/mxnet/greedy-symbol.json";
        string params = "/Users/Sheng/WorkSpace/transition-notes/mxnet/greedy-0005.params";
        greedy_model_ = new Model(max_batch_size());
        greedy_model_->Load(symbol, params);
        greedy_model_->Init(context);
    }

private:
    void AdvanceSentence(int index) override {
        ParsingReader::AdvanceSentence(index);
        if (state(index)) {
            docids_.push_front(state(index)->sentence().docid());
        }
    }
    
public:
    void ComputeMatrix() {
        vector<string> feature_names = {"feature_0_data", "feature_1_data", "feature_2_data"};
        vector<int> feature_sizes = {20, 20, 12};
        greedy_model_->DoPredict(feature_outputs_, feature_names, feature_sizes, &scores_matrix_);
        /*for (size_t i = 0; i < scores_matrix_.row_; ++i) {
            for (size_t j = 0; j < scores_matrix_.col_; ++j) {
                LOG(INFO) << "matrix(" << i << "," << j << ")=" << scores_matrix_(i, j);
            }
        }*/
        this->PerformActions();
    }

    void ComputeTokenAccuracy(const ParserState &state) {
    }

    // Performs the allowed action with the highest score on the given state.
    // Also records the accuracy whenever a terminal actions is taken.
    void PerformActions() override {
        num_tokens_ = 0;
        num_correct_ = 0;
        for (int i = 0, batch_index = 0; i < max_batch_size(); ++i) {
            ParserState *state = this->state(i);
            if (state != nullptr) {
                int best_action = 0;
                float best_score = -std::numeric_limits<float>::max();
                for (int action = 0; action < scores_matrix_.col_; ++action) {
                    float score = scores_matrix_(batch_index, action);
                    if (score > best_score &&
                        transition_system().IsAllowedAction(action, *state)) {
                        best_action = action;
                        best_score = score;
                    }
                }
                transition_system().PerformAction(best_action, state);

                // Update the # of scored correct tokens if this is the last state
                // in the sentence and save the annotated document.
                if (transition_system().IsFinalState(*state)) {
                    cout << "final state" << endl;
                    sentence_map_[state->sentence().docid()] = state->sentence();
                    state->AddParseToDocument(&sentence_map_[state->sentence().docid()]);
                }
                ++batch_index;
            }
        }
    }

    void AddAdditionalOutputs() const override {
    }

public:
    int num_tokens_ = 0;
    int num_correct_ = 0;

    string scoring_type_;

    std::deque<string> docids_;
    mutable map<string, Sentence> sentence_map_;
    typedef Matrix ScoreMatrix;

    ScoreMatrix scores_matrix_;
    Model *greedy_model_;
};

class WordEmbeddingInitializer {
public:
    explicit WordEmbeddingInitializer() {}

    void Compute()  {}
};

#include <deque>
#include "utils.h"
#include "sentence_batch.h"
#include "parser_state.h"
#include "task_context.h"
#include "work_space.h"
#include "embedding_feature_extractor.h"

class ParsingReader : public OpKernel {
  public:
    explicit ParsingReader(OpKernelConstruction *context) : OpKernel(context) {
      string file_path, corpus_name;
      OP_REQUIRES_OK(context, context->GetAttr("task_context", &file_path));
      OP_REQUIRES_OK(context, context->GetAttr("feature_size", &feature_size_));
      OP_REQUIRES_OK(context, context->GetAttr("batch_size", &max_batch_size_));
      OP_REQUIRES_OK(context, context->GetAttr("corpus_name", &corpus_name));
      OP_REQUIRES_OK(context, context->GetAttr("arg_prefix_", &arg_prefix_));

      // Reads task context from file.
      string data;
      OP_REQUIRES_OK(context, ReadFileToString(tensorflow::Env::Default(),
            file_path, &data));
      OP_REQUIRES(context,
          TextFormat::ParseFromString(data, task_context_.mutable_spec()),
          InvalidArgument("Could not parse task context at ", file_path));

      // Set up the batch reader.
      sentence_batch_.reset(
          new SentenceBatch(max_batch_size_, corpus_name));
      sentence_batch_.Init(&task_context_);
    }

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

    void Compute(OpKernelContext *context) {
      mutex_lock lock(mu_);

      // Advances states to the next position.
      PerformActions(context);

      // Advances any final states to the next sentences.
      for (int i = 0; i < max_batch_size_; ++i) {
        if (state(i) == nullptr) continue;

        // Switches to the next sentence if we're at a final state.
        while(transition_system_->IsFinalState(*state(i))) {
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
      vector<Tensor *> feature_outputs(features_->NumEmbeddings());
      for (size_t i = 0; i < feature_outputs.size(); ++i) {
        OP_REQUIRES_OK(context, context->allocate_output(
              i, TensorShape({sentence_batch_->size(),
                features_->FeatureSize(i)}), 
              &feature_outputs[i]));
      }

      // Populate feature outputs.
      for (int i = 0, index = 0; i < max_batch_size_; ++i) {
        if (states_[i] == nullptr) continue;

        // Extract features from the current parser state, and fill up the
        // avaiable batch slots.
        vector<vector<SparseFeatures>> features = 
          features_->ExtractSparseFeatures(workspaces_[i], *state[i]);

        for (size_t feature_space = 0; feature_space < features.size();
            ++feature_space) {
          int feature_size = features[feature_space].size();
          CHECK(feature_size == features_->FeatureSize(feature_space));
          auto features_output = feature_outputs[feature_space]->matrix<string>();
          for (int k = 0; k < feature_size; ++k) {
            feature_outputs(index, k) =
              features[feature_space][k].SerializeAsString();
          }
        }
        ++index;
      }

      // Return the number of epochs.
      Tensor *epoch_output;
      OP_REQUIRES_OK();
      auto num_epochs = epoch_output->scalar<int32>();
      num_epochs() = num_epochs_;

      // Create outputs specific to this reader.
      AddAdditionalOutputs(context);
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

    std::unique_ptr<SentenceBatch> sentence_batch_;

    std::vector<std::unique_ptr<ParserState>> states_;

    // Batch: WorkspaceSet objects.
    std::vector<WorkspaceSet> workspaces_;

    const TermFrequencyMap *label_map_;

    std::unique_ptr<ParserTransitionSystem> transition_system_;
    
    std::unique_ptr<ParserEmbeddingFeatureExtractor> features_;

    WorkspaceRegistry workspace_registry_;
};

class GoldParseReader : public ParsingReader {
  public:
    explicit GoldParseReader(OpKernelConstruction *context)
      : ParsingReader(context) {
    }

  private:
    // Always performs the next gold action for each state.
    void PerformActions(OpKernelContext *context) override {
      for (int i = 0; i < max_batch_size(); ++i) {
        if (state(i) != nullptr) {
          transition_system().PerformAction(
              transition_system().GetNextGoldAction(*state(i)), state(i));
        }
      }
    }

    // Adds the list of gold actions for each state as an additional output.
    void AddAdditionalOutputs(OpKernelContext *context) const override {
    }
};

class DecodedParseReader : public ParsingReader {
public:
  explicit DecodedParseReader(OpKernelConstruction *context)
    : ParsingReader(context) {

  }

private:
  void AdvanceSentence(int index) override {
  }

  void ComputeTokenAccuracy(const ParserState &state) {
  }

  void PerformActions(OpKernelContext *context) override {
  }

  void AddAdditionalOutputs(OpKernelContext *context) const override {
  }

  int num_tokens_ = 0;
  int num_correct_ = 0;

  string scoring_type_;

  mutable std::deque<string> docids_;
  mutable map<string, Sentence> sentence_map_;
};

class WordEmbeddingInitializer : public OpKernel {
public:
  explicit WordEmbeddingInitializer(OpKernelConstruction *context)
    : OpKernel(context) {
    string file_path, data;
    OP_REQUIRES_OK(context, context->GetAttr("task_context", &file_path));
    OP_REQUIRES_OK(context, ReadFileToString(tensorflow::Env::Default(),
                                             file_path, &data));
  }

  void Compute(OpKernelContext *context) override {
    // Loads words from vocabulary with mapping to ids.
    string path = TaskContext::InputFile(*task_context_.GetInput("word_map"));
    const TermFrequencyMap *word_map =
      SharedStoreUtils::GetWithDefaultName<TermFrequencyMap>(path, 0, 0);
    unordered_map<string, int64_t> vocab;
    for (int i = 0; i < word_map->size(); ++i) {
      vocab[word_map->GetTerm(i)] = i;
    }

    // Creates a reader pointing to a local copy of the vectors recordio.
    string tmp_vectors_path;
    OP_REQUIRES_OK(context, CopyToTmpPath(vectors_path_, &tmp_vectors_path));
    ProtoRecordReader reader(tmp_vectors_path);
    
  }
};

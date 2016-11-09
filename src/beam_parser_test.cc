#include <iostream>
#include "options.h"
#include "model/structured_parser.h"

using namespace std;

int TestReaderOP(int argc, char *argv[]) {
  // Init Parser Config.
  TaskContext *context = new TaskContext();
  TaskSpec *spec = context->mutable_spec();

  TaskInput *input = spec->add_input();
  input->set_name("training-corpus");
  TaskInput::Part *input_part = input->add_part();
  input_part->set_file_pattern("test/dev");

  TaskInput *label_map_input = spec->add_input();
  label_map_input->set_name("label-map");
  TaskInput::Part *label_part = label_map_input->add_part();
  label_part->set_file_pattern("label-map");

  TaskInput *tag_map_input = spec->add_input();
  tag_map_input->set_name("tag-map");
  TaskInput::Part *tag_part = tag_map_input->add_part();
  tag_part->set_file_pattern("tag-map");

  TaskInput *word_map_input = spec->add_input();
  word_map_input->set_name("word-map");
  TaskInput::Part *word_part = word_map_input->add_part();
  word_part->set_file_pattern("word-map");

  TaskSpec::Parameter *feature_param = spec->add_parameter();
  feature_param->set_name("beam_parser_features");
  feature_param->set_value("input.word input(1).word input(2).word input(3).word stack.word stack(1).word stack(2).word stack(3).word stack.child(1).word stack.child(1).sibling(-1).word stack.child(-1).word stack.child(-1).sibling(1).word stack(1).child(1).word stack(1).child(1).sibling(-1).word stack(1).child(-1).word stack(1).child(-1).sibling(1).word stack.child(2).word stack.child(-2).word stack(1).child(2).word stack(1).child(-2).word;input.tag input(1).tag input(2).tag input(3).tag stack.tag stack(1).tag stack(2).tag stack(3).tag stack.child(1).tag stack.child(1).sibling(-1).tag stack.child(-1).tag stack.child(-1).sibling(1).tag stack(1).child(1).tag stack(1).child(1).sibling(-1).tag stack(1).child(-1).tag stack(1).child(-1).sibling(1).tag stack.child(2).tag stack.child(-2).tag stack(1).child(2).tag stack(1).child(-2).tag;stack.child(1).label stack.child(1).sibling(-1).label stack.child(-1).label stack.child(-1).sibling(1).label stack(1).child(1).label stack(1).child(1).sibling(-1).label stack(1).child(-1).label stack(1).child(-1).sibling(1).label stack.child(2).label stack.child(-2).label stack(1).child(2).label stack(1).child(-2).label");

  TaskSpec::Parameter *embedding_name = spec->add_parameter();
  embedding_name->set_name("beam_parser_embedding_names");
  embedding_name->set_value("words;tags;labels");

  TaskSpec::Parameter *embedding_dims = spec->add_parameter();
  embedding_dims->set_name("beam_parser_embedding_dims");
  embedding_dims->set_value("64;32;32");

  // Init ParserEmbeddingFeatureExtractor.
  ParserEmbeddingFeatureExtractor *features_ = new ParserEmbeddingFeatureExtractor("beam_parser");
  features_->Setup(context);
  features_->Init(context);

  // Init Transition System.
  // Initializes label map and tag map.
  Options options;

  // Initializes TaggerTransitionSystem.
  ArcStandardTransitionSystem *transition_system_ = new ArcStandardTransitionSystem();
  transition_system_->Setup(context);
  transition_system_->Init(context);

  string label_map_path = TaskContext::InputFile(*context->GetInput("label-map"));
  const TermFrequencyMap *label_map = SharedStoreUtils::GetWithDefaultName<TermFrequencyMap>(
    label_map_path, 0, 0);

  int num_embeddings = features_->NumEmbeddings();
  vector<int> feature_sizes_;
  vector<int> domain_sizes_;
  vector<int> embedding_dims_;
  for (int i = 0; i < num_embeddings; ++i) {
    feature_sizes_.push_back(features_->FeatureSize(i));
    domain_sizes_.push_back(features_->EmbeddingSize(i));
    embedding_dims_.push_back(features_->EmbeddingDims(i));
  }
  // Compute number of actions in the transition system.
  int num_actions = transition_system_->NumActions(label_map->Size());

//  for (int i = 0; i < num_embeddings; ++i) {
//      LOG(INFO) << "feature size [" << i << "] " << feature_sizes_[i];
//      LOG(INFO) << "domain size [" << i << "] " << domain_sizes_[i];
//      LOG(INFO) << "embedding dim [" << i << "] " << embedding_dims_[i];
//  }
  LOG(INFO) << "num actions " << num_actions;
  vector<int> hidden_layer_sizes_({50, 50});

  int batch_size = 4;
  StructuredParser *structured_parser = new StructuredParser(batch_size, num_actions, feature_sizes_,
                                                             domain_sizes_, embedding_dims_, hidden_layer_sizes_);
  BeamParseReader *beam_reader = new BeamParseReader(context);
  BeamParser *beam_parser = new BeamParser(context);
  BeamParserOutput *beam_parser_output = new BeamParserOutput(context);

  // Setup external context.
  structured_parser->beam_reader_ = beam_reader;
  structured_parser->beam_parser_ = beam_parser;
  structured_parser->beam_parser_output_ = beam_parser_output;
  structured_parser->context = context;

  structured_parser->CreateOptimizer();
  structured_parser->BuildSequence();
  // structured_parser->InitFreshParameters();
  structured_parser->InitWithPreTrainedParameters("models/param-0001.params");

  // Start Training.
  while (1) {
    int epoch = structured_parser->TrainIter();
    if (epoch >= 1) {
        break;
    }
  }

  cout << "Can not stop?" << endl;

  delete beam_reader;
  delete beam_parser;
  delete beam_parser_output;
  delete structured_parser;

  return 0;
}


int main(int argc, char *argv[]) {
  TestReaderOP(argc, argv);
  return 0;
}

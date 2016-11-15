#include <iostream>
#include "reader_ops.h"
#include "options.h"
#include "model/greedy_parser.h"

using namespace std;

/*int TestTaggerSystem(int argc, char *argv[]) {
    Options options;
    //Initializes label map and tag map.
    TermFrequencyMap label_map;
    label_map.Load(options.label_map_file_, 0, -1);
    TermFrequencyMap tag_map;
    tag_map.Load(options.tag_map_file_, 0, -1);

    // Initializes TaggerTransitionSystem.
    //TaggerTransitionSystem *transition_system_ = new TaggerTransitionSystem();
    //transition_system_->tag_map_ = &tag_map;

    ArcStandardTransitionSystem *transition_system_ = new ArcStandardTransitionSystem();

    // Read input file.
    CoNLLSyntaxFormat readParser;
    ifstream m_file(options.input_file_);
    if (!m_file.is_open()) {
        LOG(INFO) << "Open file [" << options.input_file_ << "] failed.";
        return -1;
    }

    string record;
    string doc_id = "conll";
    vector<Sentence *> sentences;
    while (readParser.ReadRecord(&m_file, &record)) {
        readParser.ConvertFromString(doc_id, record, &sentences);
    }

    LOG(INFO) << "sentence size: " << sentences.size() << endl;
    m_file.close();

    ParserState *state = new ParserState(sentences[0],
                                         transition_system_->NewTransitionState(true), &label_map);
    LOG(INFO) << "Initial parser state: " << state->ToString();

    while (!transition_system_->IsFinalState(*state)) {
        ParserAction action = transition_system_->GetNextGoldAction(*state);
        LOG(INFO) << "Performing action: "
                  << transition_system_->ActionAsString(action, *state);
        transition_system_->PerformActionWithoutHistory(action, state);
        LOG(INFO) << "Parser State: " << state->ToString();
    }

    delete state;
    return 0;
}*/

/*int TestLexiconBuilder(int argc, char *argv[]) {
    LeiconBuilder lex_builder_;
    Options options;
    lex_builder_.Compute(options);
    return 0;
}*/

/*int TestFMLParser(int argc, char *argv[]) {
    FMLParser parser;
    FeatureExtractorDescriptor *result = new FeatureExtractorDescriptor();
    parser.Parse("stack(3).tag", result);

    cout << "Runs to here" << endl;
    return 0;
}*/


int TestParserEmbeddingFeatureExtractor(int argc, char *argv[]) {
  // Init Parser Config.
  TaskContext *context = new TaskContext();
  TaskSpec *spec = context->mutable_spec();

  TaskInput *input = spec->add_input();
  input->set_name("training-corpus");
  TaskInput::Part *input_part = input->add_part();
  input_part->set_file_pattern("test/train.conll.utf8");

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
  feature_param->set_name("parser_features");
  feature_param->set_value(
    "input.word input(1).word input(2).word input(3).word stack.word stack(1).word stack(2).word stack(3).word stack.child(1).word stack.child(1).sibling(-1).word stack.child(-1).word stack.child(-1).sibling(1).word stack(1).child(1).word stack(1).child(1).sibling(-1).word stack(1).child(-1).word stack(1).child(-1).sibling(1).word stack.child(2).word stack.child(-2).word stack(1).child(2).word stack(1).child(-2).word;input.tag input(1).tag input(2).tag input(3).tag stack.tag stack(1).tag stack(2).tag stack(3).tag stack.child(1).tag stack.child(1).sibling(-1).tag stack.child(-1).tag stack.child(-1).sibling(1).tag stack(1).child(1).tag stack(1).child(1).sibling(-1).tag stack(1).child(-1).tag stack(1).child(-1).sibling(1).tag stack.child(2).tag stack.child(-2).tag stack(1).child(2).tag stack(1).child(-2).tag;stack.child(1).label stack.child(1).sibling(-1).label stack.child(-1).label stack.child(-1).sibling(1).label stack(1).child(1).label stack(1).child(1).sibling(-1).label stack(1).child(-1).label stack(1).child(-1).sibling(1).label stack.child(2).label stack.child(-2).label stack(1).child(2).label stack(1).child(-2).label");

  TaskSpec::Parameter *embedding_name = spec->add_parameter();
  embedding_name->set_name("parser_embedding_names");
  embedding_name->set_value("words;tags;labels");

  TaskSpec::Parameter *embedding_dims = spec->add_parameter();
  embedding_dims->set_name("parser_embedding_dims");
  embedding_dims->set_value("64;32;32");

  // Init ParserEmbeddingFeatureExtractor.
  ParserEmbeddingFeatureExtractor *features_ = new ParserEmbeddingFeatureExtractor("parser");
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

  /*for (int i = 0; i < num_embeddings; ++i) {
      LOG(INFO) << "feature size [" << i << "] " << feature_sizes_[i];
      LOG(INFO) << "domain size [" << i << "] " << domain_sizes_[i];
      LOG(INFO) << "embedding dim [" << i << "] " << embedding_dims_[i];
  }
  LOG(INFO) << "num actions " << num_actions;*/
  vector<int> hidden_layer_sizes_({200, 200});

  // Create GoldParseReader.
  GoldParseReader *gold_reader_ = new GoldParseReader(context);
  GreedyParser parser(num_actions, feature_sizes_,
                      domain_sizes_, embedding_dims_,
                      hidden_layer_sizes_);

  parser.BuildNetwork();
  parser.SetupModel();
  string symbol_path = "models/greedy-symbol-0010.json";
  string param_path = "models/greedy-param-0010.params";
  int epoch = 10;
  int iter = 1;
  while (true) {
    gold_reader_->Compute();
    parser.TrainOneBatch(gold_reader_->feature_outputs_,
                         gold_reader_->gold_actions_);
    if (gold_reader_->num_epochs() != iter) {
      LOG(INFO) << "Training Accuracy: " << parser.acc.Get();
      parser.acc.Reset();
      iter = gold_reader_->num_epochs();
    }
    if (gold_reader_->num_epochs() > epoch) break;
  }
  parser.SaveModel(symbol_path, param_path);

  delete gold_reader_;
}

int TestReaderOP(int argc, char *argv[]) {
  // Init Parser Config.
  TaskContext *context = new TaskContext();
  TaskSpec *spec = context->mutable_spec();

  TaskInput *input = spec->add_input();
  input->set_name("training-corpus");
  TaskInput::Part *input_part = input->add_part();
  input_part->set_file_pattern("test/dev");
  //input->set_record_format("conll-sentence");

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
  feature_param->set_name("parser_features");
  feature_param->set_value(
    "input.word input(1).word input(2).word input(3).word stack.word stack(1).word stack(2).word stack(3).word stack.child(1).word stack.child(1).sibling(-1).word stack.child(-1).word stack.child(-1).sibling(1).word stack(1).child(1).word stack(1).child(1).sibling(-1).word stack(1).child(-1).word stack(1).child(-1).sibling(1).word stack.child(2).word stack.child(-2).word stack(1).child(2).word stack(1).child(-2).word;input.tag input(1).tag input(2).tag input(3).tag stack.tag stack(1).tag stack(2).tag stack(3).tag stack.child(1).tag stack.child(1).sibling(-1).tag stack.child(-1).tag stack.child(-1).sibling(1).tag stack(1).child(1).tag stack(1).child(1).sibling(-1).tag stack(1).child(-1).tag stack(1).child(-1).sibling(1).tag stack.child(2).tag stack.child(-2).tag stack(1).child(2).tag stack(1).child(-2).tag;stack.child(1).label stack.child(1).sibling(-1).label stack.child(-1).label stack.child(-1).sibling(1).label stack(1).child(1).label stack(1).child(1).sibling(-1).label stack(1).child(-1).label stack(1).child(-1).sibling(1).label stack.child(2).label stack.child(-2).label stack(1).child(2).label stack(1).child(-2).label");

  TaskSpec::Parameter *embedding_name = spec->add_parameter();
  embedding_name->set_name("parser_embedding_names");
  embedding_name->set_value("words;tags;labels");

  TaskSpec::Parameter *embedding_dims = spec->add_parameter();
  embedding_dims->set_name("parser_embedding_dims");
  embedding_dims->set_value("64;32;32");

  // Init ParserEmbeddingFeatureExtractor.
  ParserEmbeddingFeatureExtractor *features_ = new ParserEmbeddingFeatureExtractor("parser");
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

  vector<int> hidden_layer_sizes_({50, 50});

  GreedyParser parser(num_actions, feature_sizes_,
                      domain_sizes_, embedding_dims_,
                      hidden_layer_sizes_);
  string symbol_path = "models/symbol-0001.json";
  string param_path = "models/param-0001.params";
  parser.LoadModel(symbol_path, param_path);
  DecodedParseReader *decoder = new DecodedParseReader(context);
  while (true) {
    decoder->Compute();
    decoder->DoPadding();
    decoder->scores_matrix_ = parser.Predict(decoder->feature_outputs_);
    if (decoder->num_epochs() > 1) {
      break;
    }
  }

  decoder->OutputCoNLLResult();
  delete decoder;
}

int main(int argc, char *argv[]) {
  // TestLexiconBuilder(argc, argv);
  // TestEmbeddingFeatureExtractor(argc, argv);
  // TestTaggerSystem(argc, argv);
  TestParserEmbeddingFeatureExtractor(argc, argv);
  // TestReaderOP(argc, argv);
  return 0;
}

#include <iostream>

//#include "text_formats.cc"
#include "lexicon_builder.cc"
#include "tagger_transitions.h"
#include "fml_parser.h"
#include "task_context.h"
#include "embedding_feature_extractor.h"

using namespace std;

int ReadCoNLL(int argc, char *argv[]){
  CoNLLSyntaxFormat readParser;
  ifstream m_file(argv[1]);
  if (!m_file.is_open()) {
    LOG(INFO) << "Open file [" << argv[1] << "] failed.";
    return -1;
  }

  // Read
  string record;
  string doc_id = "conll";
  vector<Sentence *> sentences;
  while (readParser.ReadRecord(m_file, &record)) {
    readParser.ConvertFromString(doc_id, record, &sentences);
  }

  LOG(INFO) << "sentence size: " << sentences.size() << endl;
  m_file.close();

  // Write
  for (size_t i = 0; i < sentences.size(); ++i) {
    string key, value;
    readParser.ConvertToString(*sentences[i], &key, &value);
    cout << value;
  }

  // tagger transition
  
  return 0;
}


int TestTaggerSystem(int argc, char *argv[]) {
  //Initializes label map and tag map.
  Options options;
  TermFrequencyMap label_map;
  label_map.Load(options.label_map_file_, 0, -1);
  TermFrequencyMap tag_map;
  tag_map.Load(options.tag_map_file_, 0, -1);

  // Initializes TaggerTransitionSystem.
  TaggerTransitionSystem *transition_system_ = new TaggerTransitionSystem();
  transition_system_->tag_map_ = &tag_map;

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
  while (readParser.ReadRecord(m_file, &record)) {
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
}

int TestLexiconBuilder(int argc, char *argv[]) {
  
  LeiconBuilder lex_builder_;
  Options options;
  lex_builder_.Compute(options);
  return 0;
}

int TestFMLParser(int argc, char *argv[]) {
    FMLParser parser;
    FeatureExtractorDescriptor *result = new FeatureExtractorDescriptor();
    parser.Parse("stack(3).tag", result);

    cout << "Runs to here" << endl;
    return 0;
}

int TestEmbeddingFeatureExtractor(int argc, char *argv[]) {
    // Init Parser Config.
    TaskContext *context = new TaskContext();
    TaskSpec *spec = context->mutable_spec();

    TaskInput *input = spec->add_input();
    input->set_name("test/dev.conll.utf8");

    TaskSpec::Parameter *feature_param = spec->add_parameter();
    feature_param->set_name("pos_features");
    feature_param->set_value("stack(3).word stack(2).word stack(1).word stack.word input.word");

    TaskSpec::Parameter *embedding_name = spec->add_parameter();
    embedding_name->set_name("pos_embedding_names");
    embedding_name->set_value("words");

    TaskSpec::Parameter *embedding_dims = spec->add_parameter();
    embedding_dims->set_name("pos_embedding_dims");
    embedding_dims->set_value("64");

    // Init ParserEmbeddingFeatureExtractor.
    ParserEmbeddingFeatureExtractor *features_ = new ParserEmbeddingFeatureExtractor("pos");
    features_->Setup(context);
}

int main(int argc, char *argv[]) {
    TestEmbeddingFeatureExtractor(argc, argv);
    return 0;
}

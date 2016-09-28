#include <iostream>
#include "options.h"
#include "beam_reader_ops.cc"

using namespace std;

int TestReaderOP(int argc, char *argv[]) {
    // Init Parser Config.
    TaskContext *context = new TaskContext();
    TaskSpec *spec = context->mutable_spec();

    TaskInput *input = spec->add_input();
    input->set_name("training-corpus");
    TaskInput::Part *input_part = input->add_part();
    input_part->set_file_pattern("test/dev.conll.utf8");

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

    BeamParser *parser = new BeamParser(context);
    int i = 0;
    while(i < 6) {
      parser->Compute(context);
      i++;
    }

    delete parser;
}


int main(int argc, char *argv[]) {
  TestReaderOP(argc, argv);
  return 0;
}

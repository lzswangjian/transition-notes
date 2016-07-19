cmake_minimum_required(VERSION 3.3)
project(SyntaxNet)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES src/text_formats.cc src/utils.h src/utils.cc
        src/sentence.h src/term_frequency_map.h src/term_frequency_map.cc
        src/lexicon_builder.cc
        src/parser_transitions.h src/parser_transitions.cc
        src/parser_state.h src/parser_state.cc
        src/tagger_transitions.h
        src/fml_parser.h src/fml_parser.cc
        src/feature.h src/feature.cc
        src/registry.h src/registry.cc
        src/feature.h src/feature.cc
        src/feature_types.h
        src/feature_extractor.h src/feature_extractor.cc
        src/parser_features.h src/parser_features.cc
        src/sentence_features.h src/sentence_features.cc
        src/embedding_feature_extractor.h src/embedding_feature_extractor.cc
        src/task_context.h src/task_context.cc
        src/task_spec.h src/task_spec.cc
        src/registry.h src/registry.cc
        src/work_space.h src/work_space.cc
        src/cli_main.cc)

add_executable(SyntaxNet ${SOURCE_FILES})

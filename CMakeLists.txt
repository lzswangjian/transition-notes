cmake_minimum_required(VERSION 3.3)
project(SyntaxNet)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories("include")

set(SOURCE_FILES src/io/text_formats.h src/utils/utils.h src/utils/utils.cc
        src/sentence.h src/lexicon/term_frequency_map.h src/lexicon/term_frequency_map.cc
        src/lexicon/lexicon_builder.cc
        src/parser/parser_transitions.h src/parser/parser_transitions.cc
        src/parser/parser_state.h src/parser/parser_state.cc
        src/parser/tagger_transitions.cc
        src/parser/arc_standard_transitions.cc
        src/fml/fml_parser.h src/fml/fml_parser.cc
        src/feature/feature.h src/feature/feature.cc
        src/utils/registry.h src/utils/registry.cc
        src/feature/feature.h src/feature/feature.cc
        src/feature/feature_types.h
        src/feature/feature_extractor.h src/feature/feature_extractor.cc
        src/feature/parser_features.h src/feature/parser_features.cc
        src/feature/sentence_features.h src/feature/sentence_features.cc
        src/feature/embedding_feature_extractor.h src/feature/embedding_feature_extractor.cc
        src/utils/task_context.h src/utils/task_context.cc
        src/utils/task_spec.h src/utils/task_spec.cc
        src/utils/registry.h src/utils/registry.cc
        src/utils/work_space.h src/utils/work_space.cc
        src/utils/shared_store.h src/utils/shared_store.cc
        src/io/text_reader.h src/io/text_reader.cc
        src/io/document_format.h src/io/document_format.cc
        src/model/model_predict.cc
        src/sentence_batch.h src/sentence_batch.cc
        src/reader_ops.cc
        src/cli_main.cc)

LINK_DIRECTORIES(lib)
add_executable(SyntaxNet ${SOURCE_FILES})

TARGET_LINK_LIBRARIES(SyntaxNet mxnet)

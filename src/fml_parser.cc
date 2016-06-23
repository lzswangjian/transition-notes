#include "fml_parser.h"

#include <ctype.h>

void FMLParser::Initialize(const string &source) {
  // Initialize parser state.
  source_ = source;
  current_ = source_.begin();
  item_start_ = line_start_ = current_;
  line_number_ = item_line_number_ = 1;

  // Read first input item.
  NextItem();
}

void FMLParser::Next() {
  // Move to the next input character. If we are at a line break
  // update line number and line start position.
  if (*current_ == '\n') {
    ++line_number_;
    ++current_;
    line_start_ = current_;
  } else {
    ++current_;
  }
}

void FMLParser::NextItem() {
  // Skip white space and comments.
  while (!eos()) {
    if (*current_ == '#') {
      // Skip comment.
      while (!eos() && *current_ != '\n') Next();
    } else if (isspace(*current_)) {
      // Skip whilespace
      while (!eos() && isspace(*current_)) Next();
    } else {
      break;
    }
  }

  // Record start position for next item.
  item_start_ = current_;
  item_line_number_ = line_number_;

  // Check for end of input.
  if (eos()) {
    item_type_ = END;
    return;
  }

  // Parse number.
  if (isdigit(*current_) || *current_ == '+' || *current_ == '-') {
    string::iterator start = current_;
    Next();
    while(isdigit(*current_) || *current_ == '.') Next();
    item_text_.assign(start, current_);
    item_type_ = NUMBER;
    return;
  }

  // Parse string.
  if (*current_ == '"') {
    Next();
    string::iterator start = current_;
    while(*current_ != '"') {
      if (eos()) Error("Unterminated string");
      Next();
    }
    item_text_.assign(start, current_);
    item_type_ = STRING;
    Next();
    return;
  }

  // Parse identifier name.
  if (isalpha(*current_) || *current_ == '_' || *current_ == '/') {
    string::iterator start = current_;
    while (isalnum(*current_) || *current_ == '_' || *current_ == '-' ||
        *current_ == '/') Next();
    item_text_.assign(start, current_);
    item_type_ = NAME;
    return;
  }

  // Single character item.
  item_type_ = *current_;
  Next();
}

void FMLParser::Parse(const string &source,
    FeatureExtractorDescriptor *result) {
  // Initialize parser.
  Initialize(source);

  while (item_type_ != END) {
    // Parse either a parameter name or a feature.
    if (item_type_ != NAME) Error("Feature type name expected");
    string name = item_text_;
    NextItem();

    if (item_type_ == '=') {
      Error("Invalid syntax: feature expected");
    } else {
      // Parse feature.
      FeatureExtractorDescriptor *descriptor = result->add_feature();
      descriptor->set_type(name);
      ParseFeature(descriptor);
    }
  }
}

void FMLParser::ParseFeature(FeatureExtractorDescriptor *result) {
  // Parse argument and paramters.
  if (item_type_ == '(') {
    NextItem();
    ParserParameter(result);
    while (item_type_ == ',') {
      NextItem();
      ParserParameter(result);
    }

    if (item_type_ != ')') Error(") expected");
    NextItem();
  }

  // Parse feature name.
  if (item_type_ == ':') {
    NextItem();
    if (item_type_ != NAME && item_type_ != STRING) {
      Error("Feature name expected");
    }
    string name = item_text_;
    NextItem();

    // Set feature name.
    result->set_name(name);
  }

  // Parse sub-features.
  if (item_type_ == '.') {
    // Parse dotted sub-feature.
    NextItem();
    if (item_type_ != NAME)  Error("feature type name expected");
    string type = item_text_;
    NextItem();

    // Parse sub-feature.
    FeatureExtractorDescriptor *subfeature = result->add_feature();
    subfeature->set_type(type);
    ParseFeature(subfeature);
  } else if (item_type_ == '{') {
    // Parse sub-feature block
    NextItem();
    while (item_type_ != '}') {
      if (item_type_ != NAME)  Error("feature type name expected");
      string type = item_text_;
      NextItem();

      // Parse sub-feature.
      FeatureExtractorDescriptor *subfeature = result->add_feature();
      subfeature->set_type(type);
      ParseFeature(subfeature);
    }
    NextItem();
  }
}

void FMLParser::ParserParameter(FeatureExtractorDescriptor *result) {
  if (item_type_ == NUMBER) {
    int argument = 
      utils::ParseUsing<int>(item_text_, tensorflow::strings::safe_strto32);
    NextItem();
    
    // Set default argument for feature.
    result->set_argument(argument);
  } else if (item_type_ == NAME) {
    string name = item_text_;
    NextItem();
    if (item_type_ != '=') Error("= expected");
    NextItem();
    if (item_type_ >= END) Error("Parameter value expected");
    string value = item_text_;
    NextItem();

    // Adds parameter to feature.
    Parameter *parameter;
    parameter = result->add_parameter();
    parameter->set_name(name);
    parameter->set_value(value);
  } else {
    Error("Syntax error in parameter list");
  }
}

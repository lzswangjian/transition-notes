/*!
 * \brief Feature modeling language (fml) parser.
 */

#ifndef FML_PARSER_H_
#define FML_PARSER_H_

#include <string>

class FMLParser {
  public:
    // Parses fml specification into feature extractor descriptor.
    void Parse(const string &source, FeatureExtractorDescriptor *result);

  private:
    // Source text.
    string source_;

    // Current input position.
    string::iterator current_;

    // Line number for current input position.
    int line_number_;
};
#endif

#ifndef STAN_LANG_GENERATOR_GENERATE_QUOTED_STRING_HPP
#define STAN_LANG_GENERATOR_GENERATE_QUOTED_STRING_HPP

#include <ostream>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Print the specified string to the specified output stream,
     * wrapping in double quotes (") and replacing all double quotes
     * in the input with apostrophes (').  For example, if the input
     * string is <tt>ab"cde"fg</tt> then the string
     * <tt>"ab'cde'fg"</tt> is streamed to the output stream.
     *
     * @param[in] s String to output
     * @param[in,out] o Output stream
     */
    void generate_quoted_string(const std::string& s, std::ostream& o) {
      o << '"';
      for (size_t i = 0; i < s.size(); ++i)
        o << ((s[i] == '"') ? '\'' : s[i]);
      o << '"';
    }



  }
}
#endif

#ifndef STAN_LANG_GENERATOR_GENERATE_TYPE_HPP
#define STAN_LANG_GENERATOR_GENERATE_TYPE_HPP

#include <stan/lang/ast.hpp>
#include <ostream>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Generate the base type for multi-dimensional arrays.
     *
     * @param[in] cpptype typename
     * @param[in] end number of standard vector embeddings
     * @param[in,out] o stream for generating
     */
    void generate_type(const std::string& cpptype,
                       const std::vector<expression>& /*dims*/,
                       size_t end, std::ostream& o) {
      for (size_t i = 0; i < end; ++i)
        o << "std::vector<";
      o << cpptype;
      for (size_t i = 0; i < end; ++i) {
        // nested types closed by "> >" (not input operator ">>")
        if (i > 0) o << ' ';
        o << '>';
      }
    }

  }
}
#endif

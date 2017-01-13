#ifndef STAN_LANG_GENERATOR_GENERATE_TYPE_HPP
#define STAN_LANG_GENERATOR_GENERATE_TYPE_HPP

#include <stan/lang/ast.hpp>
#include <ostream>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Generate the base type for multi-dimensional arrays.
     *
     * @param[in] base_type string representing bae type
     * @param[in] end number of standard vector embeddings
     * @param[in,out] o stream for generating
     */
    void generate_type(const std::string& base_type,
                       const std::vector<expression>& /*dims*/,
                       size_t end, std::ostream& o) {
      for (size_t i = 0; i < end; ++i)
        o << "std::vector<";
      o << base_type;
      for (size_t i = 0; i < end; ++i) {
        if (i > 0) o << ' ';
        o << '>';
      }
    }

  }
}
#endif

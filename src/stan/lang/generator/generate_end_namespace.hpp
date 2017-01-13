#ifndef STAN_LANG_GENERATOR_GENERATE_END_NAMESPACE_HPP
#define STAN_LANG_GENERATOR_GENERATE_END_NAMESPACE_HPP

#include <ostream>

namespace stan {
  namespace lang {

    /**
     * Generate the end of a namespace to the specified stream.
     *
     * @param[in, out] o stream for generating
     */
    void generate_end_namespace(std::ostream& o) {
      o << "}" << EOL2;
    }

  }
}
#endif

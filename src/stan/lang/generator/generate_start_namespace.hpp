#ifndef STAN_LANG_GENERATOR_GENERATE_START_NAMESPACE_HPP
#define STAN_LANG_GENERATOR_GENERATE_START_NAMESPACE_HPP

#include <ostream>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Generate the opening name and brace for a namespace, with two
     * end of lines.
     *
     * @param[in] name name of namespace
     * @param[in,out] o stream for generating
     */
    void generate_start_namespace(const std::string& name, std::ostream& o) {
      o << "namespace " << name << "_namespace {" << EOL2;
    }

  }
}
#endif

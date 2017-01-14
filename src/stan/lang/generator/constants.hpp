#ifndef STAN_LANG_GENERATOR_CONSTANTS_HPP
#define STAN_LANG_GENERATOR_CONSTANTS_HPP

#include <stan/lang/ast.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * End-of-line marker.
     */
    const std::string EOL("\n");

    /**
     * Sequence of two end-of-line markers.
     */
    const std::string EOL2("\n\n");

    /**
     * Single indentation.
     */
    const std::string INDENT("    ");

    /**
     * Double indentation.
     */
    const std::string INDENT2("        ");

    /**
     * Triple indentation.
     */
    const std::string INDENT3("            ");

    /**
     * Size zero vector of expressions.
     */
    const std::vector<expression> EMPTY_EXP_VECTOR(0);

  }
}
#endif

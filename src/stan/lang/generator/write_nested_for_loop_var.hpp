#ifndef STAN_LANG_GENERATOR_WRITE_NESTED_FOR_LOOP_VAR_ARG_HPP
#define STAN_LANG_GENERATOR_WRITE_NESTED_FOR_LOOP_VAR_ARG_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_expression.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <ostream>
#include <vector>

namespace stan {
  namespace lang {
    
    /**
     * Generate the loop variable and indexes for the specified
     * variable name and number of dimensions with the
     * specified indentation level writing to the specified stream.
     *
     * @param[in] name name of variable
     * @param[in] dims_size number of dimensions of variable
     * @param[in] indent indentation level
     * @param[in,out] o stream for generating
     */
    void write_nested_for_loop_var(const std::string& name, size_t dims_size,
                                   int indent, std::ostream& o) {
      o << name;
      for (size_t i = 0; i < dims_size; ++i)
        o << "[k" << i << "__]";
    }
  }
}
#endif

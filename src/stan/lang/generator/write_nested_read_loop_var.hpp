#ifndef STAN_LANG_GENERATOR_WRITE_NESTED_READ_LOOP_VAR_ARG_HPP
#define STAN_LANG_GENERATOR_WRITE_NESTED_READ_LOOP_VAR_ARG_HPP

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
     * @param[in] num_ar_dims number of array dimensions of variable
     * @param[in] num_args ternary indicator for matrix/vector/scalar types
     * @param[in] indent indentation level
     * @param[in,out] o stream for generating
     */
    void write_nested_read_loop_var(const std::string&name, size_t num_ar_dims,
                                    size_t num_args, int indent, std::ostream& o) {

      generate_indent(indent + num_ar_dims + num_args, o);
      o << name;

      for (size_t i = 0; i < num_ar_dims; ++i)
        o << "[k" << i << "__]";

      if (num_args == 1)
        o << "(j_1__)";
      else if (num_args == 2)
        o << "(j_1__, j_2__)";
    }

  }
}
#endif

#ifndef STAN_LANG_GENERATOR_WRITE_NESTED_FOR_LOOP_BEGIN_ARG_HPP
#define STAN_LANG_GENERATOR_WRITE_NESTED_FOR_LOOP_BEGIN_ARG_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_expression.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <ostream>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Generate the openings of a sequence of zero or more nested for loops
     * corresponding to the specified dimension sizes with the
     * specified indentation level writing to the specified stream.
     * Declare named size_t variable for each dimension size in order to avoid
     * re-evaluation of dimension size expression on each iteration.
     *
     * @param[in] dims dimension sizes
     * @param[in] indent indentation level
     * @param[in,out] o stream for generating
     */
    void write_nested_for_loop_begin(const std::vector<expression>& dims,
                                   int indent, std::ostream& o) {
      // declare size_t var k_<n>_max__
      for (size_t i = 0; i < dims.size(); ++i) {
        generate_indent(indent, o);
        o << "size_t k_" << i << "_max__ = ";
        generate_expression(dims[i], NOT_USER_FACING, o);
        o << ";" << EOL;
      }
      // nested for stmts open
      for (size_t i = 0; i < dims.size(); ++i) {
        generate_indent(indent + i, o);
        o << "for (int k"  << i << "__ = 0;"
          << " k" << i << "__ < k_" << i << "_max__;"
          << " ++k" << i << "__) {" << EOL;
      }
    }

  }
}
#endif

#ifndef STAN_LANG_GENERATOR_WRITE_NESTED_READ_LOOP_BEGIN_ARG_HPP
#define STAN_LANG_GENERATOR_WRITE_NESTED_READ_LOOP_BEGIN_ARG_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_expression.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <ostream>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Generate the openings of a sequence of zero or more for loops
     * corresponding to the specified dimension sizes with the
     * specified indentation level writing to the specified stream.
     * Declare named size_t variable for each dimension size in order to avoid
     * re-evaluation of dimension size expression on each iteration.
     * 
     * NOTE: Indexing order is column major, nesting is innermost to outermost
     * e.g., 3-d array of matrices indexing order:  col, row, d3, d2, d1
     *
     * @param[in] dims dimension sizes
     * @param[in] num_args ternary indicator for matrix/vector/scalar types
     * @param[in] indent indentation level
     * @param[in,out] o stream for generating
     */
    void write_nested_read_loop_begin(const std::vector<expression>& dims,
                                      size_t num_args, int indent, std::ostream& o)
    {
      // declare row, col dimension indexes size_t var j_<n>_max__
      for (size_t i = 0; i < num_args; ++i) {
        generate_indent(indent, o);
        o << "size_t j_" << (num_args - i) << "_max__ = ";
        generate_expression(dims[i], NOT_USER_FACING, o);
        o << ";" << EOL;
      }
      
      // declare array indexes size_t var k_<n>_max__
      for (size_t i = num_args; i < dims.size(); ++i) {
        generate_indent(indent, o);
        o << "size_t k_" << (dims.size() - i) << "_max__ = ";
        generate_expression(dims[i], NOT_USER_FACING, o);
        o << ";" << EOL;
      }

      // nested for stmts open, row, col indexes
      for (size_t i = 0; i < num_args; ++i) {
        generate_indent(indent + i, o);
        int idx = num_args - i;
        o << "for (int j"  << idx << "__ = 0;"
          << " j" << idx << "__ < j_" << idx << "_max__;"
          << " ++j" << idx << "__) {" << EOL;
      }
      // nested for stmts open, array indexes
      for (size_t i = num_args; i < dims.size(); ++i) {
        generate_indent(indent + i, o);
        int idx = dims.size() - i;
        o << "for (int k"  << idx << "__ = 0;"
          << " k" << idx << "__ < k_" << idx << "_max__;"
          << " ++k" << idx << "__) {" << EOL;
      }
    }

  }
}
#endif

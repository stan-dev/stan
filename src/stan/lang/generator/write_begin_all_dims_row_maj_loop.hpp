#ifndef STAN_LANG_GENERATOR_WRITE_BEGIN_ALL_DIMS_ROW_MAJ_LOOP_HPP
#define STAN_LANG_GENERATOR_WRITE_BEGIN_ALL_DIMS_ROW_MAJ_LOOP_HPP

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
     * @param[in] name variable name
     * @param[in] dims dimension sizes
     * @param[in] indent indentation level
     * @param[in,out] o stream for generating
     */
    void write_begin_all_dims_row_maj_loop(const std::string& name,
                                           const std::vector<expression>& dims,
                                           size_t num_args,
                                           int indent, std::ostream& o) {
      // declare size_t var k_<n>_max__
      for (size_t i = 0; i < dims.size() - num_args; ++i) {
        generate_indent(indent, o);
        o << "size_t " << name << "_k_" << i << "_max__ = ";
        generate_expression(dims[i], NOT_USER_FACING, o);
        o << ";" << EOL;
      }
      // declare size_t max for num rows/cols
      if (num_args > 0) {
        generate_indent(indent, o);
        o << "size_t " << name << "_j_1_max__ = ";
        generate_expression(dims[dims.size() - 2], NOT_USER_FACING, o);
        o << ";" << EOL;
      }
      if (num_args == 2) {
        generate_indent(indent, o);
        o << "size_t " << name << "_j_2_max__ = ";
        generate_expression(dims[dims.size() - 1], NOT_USER_FACING, o);
        o << ";" << EOL;
      }

      // nested for stmts open
      for (size_t i = 0; i < dims.size() - num_args; ++i) {
        generate_indent(indent + i, o);
        o << "for (int k"  << i << "__ = 0;"
          << " k" << i << "__ < " << name << "_k_" << i << "_max__;"
          << " ++k" << i << "__) {" << EOL;
      }
      // final dims are row/col indexes
      if (num_args > 0) {
        generate_indent(indent + dims.size() - num_args, o);
        o << "for (int j_1__ = 0; j_1__ < " << name << "_j_1_max__; "
          << " ++j_1__) {" << EOL;
      }
      if (num_args == 2) {
        generate_indent(indent + dims.size() - 1, o);
        o << "for (int j_2__ = 0; j_2__ < " << name << "_j_2_max__; "
          << " ++j_2__) {" << EOL;
      }
    }

  }
}
#endif

#ifndef STAN_LANG_GENERATOR_GENERATE_INITIALIZER_HPP
#define STAN_LANG_GENERATOR_GENERATE_INITIALIZER_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_eigen_index_expression.hpp>
#include <stan/lang/generator/generate_type.hpp>
#include <ostream>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    void generate_expression(const expression& e, std::ostream& o);

    /**
     * Generate an initializer for a variable of the specified base
     * type with the specified dimension sizes with an additional size
     * for vectors and row vectors and two additional sizes for
     * matrices.
     *
     * @param o stream for generating
     * @param base_type base type of variable
     * @param dims sizes of dimensions for variable
     * @param type_arg1 size of vector or row vector or size of rows
     * for matrix, not used otherwise
     * @param type_arg2 size of columns for matrix, not used otherwise
     */
    void generate_initializer(std::ostream& o,
                              const std::string& base_type,
                              const std::vector<expression>& dims,
                              const expression& type_arg1 = expression(),
                              const expression& type_arg2 = expression()) {
      for (size_t i = 0; i < dims.size(); ++i) {
        o << '(';
        generate_expression(dims[i].expr_, o);
        o << ',';
        generate_type(base_type, dims, dims.size() - i - 1, o);
      }

      o << '(';
      if (!is_nil(type_arg1)) {
        generate_eigen_index_expression(type_arg1, o);
        if (!is_nil(type_arg2)) {
          o << ',';
          generate_eigen_index_expression(type_arg2, o);
        }
      } else if (!is_nil(type_arg2.expr_)) {
        generate_eigen_index_expression(type_arg2, o);
      } else {
        o << '0';
      }
      o << ')';

      for (size_t i = 0; i < dims.size(); ++i)
        o << ')';
      o << ';' << EOL;
    }

  }
}
#endif

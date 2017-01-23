#ifndef STAN_LANG_GENERATOR_GENERATE_VALIDATE_CONTEXT_SIZE_HPP
#define STAN_LANG_GENERATOR_GENERATE_VALIDATE_CONTEXT_SIZE_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_validate_positive.hpp>
#include <ostream>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    void generate_expression(const expression& e, std::ostream& o);

    /*
     * Generates code to validate data variables to make sure they
     * only use positive dimension sizes and that the var_context
     * out of which they are read have matching dimension sizes.
     *
     * @param[in,out] o stream for generating
     * @param[in] stage step of processing
     * @param[in] var_name name of variable being validated
     * @param[in] base_type base type of variable
     * @param[in] dims dimension sizes
     * @param[in] type_arg1 optional size of vector, row vector or
     * matrix rows
     * @param[in] type_arg2 optional size of matrix columns
     */
    void generate_validate_context_size(std::ostream& o,
                                        const std::string& stage,
                                        const std::string& var_name,
                                        const std::string& base_type,
                                        const std::vector<expression>& dims,
                                        const expression& type_arg1
                                          = expression(),
                                        const expression& type_arg2
                                          = expression()) {
      // check array dimensions
      for (size_t i = 0; i < dims.size(); ++i)
        generate_validate_positive(var_name, dims[i], 2, o);
      // check vector, row_vector, and matrix rows
      if (!is_nil(type_arg1))
        generate_validate_positive(var_name, type_arg1, 2, o);
      // check matrix cols
      if (!is_nil(type_arg2))
        generate_validate_positive(var_name, type_arg2, 2, o);

      // calls var_context to make sure dimensions match
      o << INDENT2 << "context__.validate_dims("
        << '"' << stage << '"' << ", "
        << '"' << var_name << '"' << ", "
        << '"' << base_type << '"' << ", "
        << "context__.to_vec(";
      for (size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) o << ",";
        generate_expression(dims[i].expr_, o);
      }
      if (!is_nil(type_arg1)) {
        if (dims.size() > 0) o << ",";
        generate_expression(type_arg1.expr_, o);
        if (!is_nil(type_arg2)) {
          o << ",";
          generate_expression(type_arg2.expr_, o);
        }
      }
      o << "));" << EOL;
    }

  }
}
#endif

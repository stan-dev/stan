#ifndef STAN_LANG_GENERATOR_GENERATE_VALIDATE_VAR_DIMS_HPP
#define STAN_LANG_GENERATOR_GENERATE_VALIDATE_VAR_DIMS_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/generate_validate_positive.hpp>
#include <ostream>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Generate code to validate variable sizes for array dimensions
     * and vector/matrix number of rows, columns to the specified stream
     * at the specified level of indentation.
     *
     * @param[in] decl variable declaration
     * @param[in] indent indentation level
     * @param[in,out] o stream for generating
     */
    void generate_validate_var_dims(const block_var_decl& var_decl,
                                    int indent, std::ostream& o) {
      std::string name(var_decl.name());
      expression arg1 = var_decl.type().arg1();
      expression arg2 = var_decl.type().arg2();
      if (var_decl.type().is_array_type()) {
        arg1 = var_decl.type().array_contains().arg1();
        arg2 = var_decl.type().array_contains().arg2();
      }
      std::vector<expression> ar_var_dims = var_decl.type().array_lens();

      if (!is_nil(arg1))
        generate_validate_positive(name, arg1, indent, o);

      if (!is_nil(arg2))
        generate_validate_positive(name, arg2, indent, o);

      for (size_t i = 0; i < ar_var_dims.size(); ++i)
        generate_validate_positive(name, ar_var_dims[i], indent, o);
    }

  }
}
#endif

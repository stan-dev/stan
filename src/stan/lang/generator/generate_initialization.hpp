#ifndef STAN_LANG_GENERATOR_GENERATE_INITIALIZATION_HPP
#define STAN_LANG_GENERATOR_GENERATE_INITIALIZATION_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/generate_initializer.hpp>
#include <stan/lang/generator/generate_type.hpp>
#include <stan/lang/generator/generate_validate_positive.hpp>
#include <ostream>
#include <string>
#include <vector>

namespace stan {
  namespace lang {



    /**
     * Generate varaible initialization, checking dimension sizes are
     * all positive, gnerating to the specified stream for a variable
     * with the specified name, type, dimension sizes, and optional
     * matrix/vector size declarations.
     *
     * @param[in,out] o stream for generating
     * @param[in] var_name name of variable being initialized
     * @param[in] base_type base type of variable
     * @param[in] dims dimension sizes
     * @param[in] type_arg1 optional vector/row-vector size or matrix
     * rows
     * @param[in] type_arg2 optional size of matrix columns
     */
    void generate_initialization(std::ostream& o, const std::string& var_name,
                                 const std::string& base_type,
                                 const std::vector<expression>& dims,
                                 const expression& type_arg1 = expression(),
                                 const expression& type_arg2 = expression()) {
      // check array dimensions
      for (size_t i = 0; i < dims.size(); ++i)
        generate_validate_positive(var_name, dims[i], 2, o);
      // check vector, row_vector, and matrix rows
      if (!is_nil(type_arg1))
        generate_validate_positive(var_name, type_arg1, 2, o);
      // check matrix cols
      if (!is_nil(type_arg2))
        generate_validate_positive(var_name, type_arg2, 2, o);

      // initialize variable
      o << INDENT2
        << var_name << " = ";
      generate_type(base_type, dims, dims.size(), o);
      generate_initializer(o, base_type, dims, type_arg1, type_arg2);
    }

  }
}
#endif

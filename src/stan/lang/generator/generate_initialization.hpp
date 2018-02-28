#ifndef STAN_LANG_GENERATOR_GENERATE_INITIALIZATION_HPP
#define STAN_LANG_GENERATOR_GENERATE_INITIALIZATION_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <stan/lang/generator/generate_initializer.hpp>
#include <stan/lang/generator/generate_type.hpp>
#include <stan/lang/generator/generate_validate_positive.hpp>
#include <stan/lang/generator/get_typedef_var_type.hpp>
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
     * @param[in] indent indentation level
     * @param[in] var_name name of variable being initialized
     * @param[in] bare type of variable (after array unfolding)
     * @param[in] arg1 expr for vector/matrix num rows
     * @param[in] arg2 expr for matrix num cols
     * @param[in] dims array dimension sizes
     */
    void generate_initialization(std::ostream& o, size_t indent,
                                 const std::string& var_name,
                                 const bare_expr_type& bare_type,
                                 const expression& arg1,
                                 const expression& arg2,
                                 const std::vector<expression>& dims) {

      if (!is_nil(arg1))
        generate_validate_positive(var_name, arg1, indent, o);
      if (!is_nil(arg2))
        generate_validate_positive(var_name, arg2, indent, o);
      for (size_t i = 0; i < dims.size(); ++i) 
        generate_validate_positive(var_name, dims[i], indent, o);
      
      generate_indent(indent, o);
      o << var_name << " = ";
      if (bare_type.is_double_type() && dims.size() == 0)
        o << "DUMMY_VAR__;" << EOL;
      else {
        std::string cpptype = get_typedef_var_type(bare_type);
        generate_type(cpptype, dims, dims.size(), o);
        generate_initializer(o, cpptype, dims, arg1, arg2);
      }
    }

  }
}
#endif

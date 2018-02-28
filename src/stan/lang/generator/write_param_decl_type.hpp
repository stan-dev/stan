#ifndef STAN_LANG_GENERATOR_WRITE_PARAM_DECL_TYPE_HPP
#define STAN_LANG_GENERATOR_WRITE_PARAM_DECL_TYPE_HPP

#include <stan/lang/generator/get_typedef_var_type.hpp>
#include <string>
#include <ostream>

namespace stan {
  namespace lang {
    /**
     * Write the parameter variable type declaration to the specified stream.
     *
     * Note:  this is called after array type has been unfolded,
     * so bare_type shouldn't be bare_array_type (or ill_formed_type).
     *
     * @param[in] bare_type parameter variable type
     * @param[in] number of array dimensions
     * @param[in] indent indentation level
     * @param[in,out] o stream for generating
     */
    void
    write_param_decl_type(const bare_expr_type& bare_type,
                        int ar_dims,
                        int indent,
                        std::ostream& o){
      std::string cpp_typename = get_typedef_var_type(bare_type);
      for (int i = 0; i < indent; ++i)
        o << INDENT;
      for (int i = 0; i < ar_dims; ++i)
        o << "vector<";
      o << cpp_typename;
      for (int i = 0; i < ar_dims; ++i) {
        if (ar_dims > 0)
          o << " ";  // maybe not needed for c++11
        o << " >";
      }
    }
  }
}
#endif

#ifndef STAN_LANG_GENERATOR_GENERATE_VALIDATE_TPARAM_INITS_HPP
#define STAN_LANG_GENERATOR_GENERATE_VALIDATE_TPARAM_INITS_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_expression.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <stan/lang/generator/write_begin_all_dims_row_maj_loop.hpp>
#include <stan/lang/generator/write_end_loop.hpp>
#include <stan/lang/generator/write_var_idx_all_dims.hpp>
#include <stan/lang/generator/write_var_idx_all_dims_msg.hpp>
#include <ostream>
#include <vector>

namespace stan {
  namespace lang {

    // check initialization
    
    /**
     * Generate code to validate the specified variable declaration
     * using the specified indentation level and stream.
     * Checks any defined bounds or constraints on specialized types.
     * NOTE:  bounded / specialized types are mutually exclusive
     *
     * @param[in] decl variable declaration
     * @param[in] indent indentation level
     * @param[in,out] o stream for generating
     */
    void generate_validate_tparam_inits(const block_var_decl decl,
                                        int indent, std::ostream& o) {

      std::string var_name(decl.name());
      // unfold array type to get array element info
      block_var_type btype = (decl.type());
      if (btype.is_array_type())
        btype = btype.array_contains();
      std::vector<expression> ar_lens(decl.type().array_lens());
      expression arg1 = btype.arg1();
      expression arg2 = btype.arg2();

      // nested loops in row-major order for all dimensions
      // check stan::math::is_uninitialized
      std::vector<expression> dims;
      size_t num_args = (!is_nil(arg2)) ? 2 : ((!is_nil(arg1)) ? 1 : 0);
      for (size_t i = 0; i < ar_lens.size(); ++i)
        dims.push_back(ar_lens[i]);
      if (num_args > 0)
        dims.push_back(arg1);
      if (num_args == 2)
        dims.push_back(arg2);

      write_begin_all_dims_row_maj_loop(var_name, dims, num_args, indent, o);
      
      // innermost loop stmt: do check, throw exception
      // TODO:mitzi - generate located exception?
      generate_indent(indent + dims.size(), o);
      o << "if (stan::math::is_uninitialized(" << var_name;
      write_var_idx_all_dims(ar_lens.size(), num_args, o);
      o << ")) {" << EOL;
      
      generate_indent(indent + dims.size() + 1, o);
      o << "std::stringstream msg__;" << EOL;

      generate_indent(indent + dims.size() + 1, o);
      o << "msg__ << \"Undefined transformed parameter: " << var_name << "\"";
      write_var_idx_all_dims_msg(ar_lens.size(), num_args, o);

      generate_indent(indent + dims.size() + 1, o);
      o << "throw std::runtime_error(msg__.str());" << EOL;
      
      generate_indent(indent + dims.size(), o);
      o << "}" << EOL;
      
      write_end_loop(dims.size(), indent, o);
    }
  }
}
#endif

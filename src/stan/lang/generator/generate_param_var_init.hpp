#ifndef STAN_LANG_GENERATOR_GENERATE_PARAM_VAR_INIT_HPP
#define STAN_LANG_GENERATOR_GENERATE_PARAM_VAR_INIT_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_expression.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <stan/lang/generator/write_nested_for_loop_end.hpp>
#include <stan/lang/generator/write_nested_for_loop_var.hpp>
#include <stan/lang/generator/write_nested_resize_loop_begin.hpp>
#include <iostream>
#include <ostream>
#include <vector>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Generate dynamic initializations for container parameter variables
     * or void statement for scalar parameters.
     *
     * @param[in] var_decl parameter block variable
     * @param[in] gen_decl_stmt if true, generate variable declaration
     * @param[in] indent indentation level
     * @param[in,out] o stream for generating
     */
    void generate_param_var_init(const block_var_decl& var_decl,
                                 bool gen_decl_stmt,
                                 int indent, std::ostream& o) {

      // setup - name, type, and var shape
      std::string var_name(var_decl.name());
      block_var_type btype = (var_decl.type());
      if (btype.is_array_type())
        btype = btype.array_contains();

      std::string constrain_str = get_constrain_fn_prefix(btype);
      std::vector<expression> dims(var_decl.type().array_lens());
      // add comma before lp__ arg to constrain function?
      std::string lp_arg("lp__)");
      if (btype.has_def_bounds() || !btype.bare_type().is_double_type())
        lp_arg= ", lp__)";
      
      // declare
      if (gen_decl_stmt) {
        generate_indent(indent, o);
        generate_bare_type(var_decl.type().bare_type(), "local_scalar_t__", o);
        o << " " << var_name << ";" << EOL;
      }

      // init
      write_nested_resize_loop_begin(var_name, dims, indent, o);
      
      // innermost loop stmt: read in param, apply jacobian
      generate_indent(indent + dims.size(), o);
      o << "if (jacobian__)" << EOL;

      generate_indent(indent + dims.size() + 1, o);
      if (dims.size() > 0) {
        write_nested_for_loop_var(var_name, dims.size() - 1, indent, o);
        o << ".push_back(in__." << constrain_str << lp_arg << ");" << EOL;
      }
      else {
        o << var_name << " = in__." << constrain_str << lp_arg << ";" << EOL;
      }

      generate_indent(indent + dims.size(), o);
      o << "else" << EOL;

      generate_indent(indent + dims.size() + 1, o);
      if (dims.size() > 0) {
        write_nested_for_loop_var(var_name, dims.size() - 1, indent, o);
        o << ".push_back(in__." << constrain_str << "));" << EOL;
      }
      else {
        o << var_name << " = in__." << constrain_str << ");" << EOL;
      }
      
      write_nested_for_loop_end(dims.size(), indent, o);
    }        

  }
}
#endif

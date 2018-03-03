#ifndef STAN_LANG_GENERATOR_GENERATE_LOG_PROB_VAR_INITS_HPP
#define STAN_LANG_GENERATOR_GENERATE_LOG_PROB_VAR_INITS_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <stan/lang/generator/get_constrain_fn_prefix.hpp>
#include <stan/lang/generator/get_verbose_var_type.hpp>
#include <stan/lang/generator/write_var_decl_type.hpp>
#include <ostream>
#include <vector>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Generate initializations for the specified log_prob variables,
     * with flags indicating whether the generation is in a variable
     * context and whether variables need to be declared, writing to
     * the specified stream.
     *
     * @param[in] vs variable declarations
     * @param[in] declare_vars true if variables should be declared
     * @param[in] indent indentation level
     * @param[in,out] o stream for generating
     */
    void generate_log_prob_var_inits(std::vector<block_var_decl> vs,
                                     bool declare_vars,
                                     int indent, std::ostream& o) {
      generate_indent(indent, o);
      o << "stan::io::reader<local_scalar_t__> in__(params_r__);"
        << EOL2;

      // per-var init
      for (size_t i = 0; i < vs.size(); ++i) {
        // TODO:mitzi : generate runtime error?
        // parser should prevent this from happening
        // avoid generating code that won't compile - flag/ignore int params
        if (vs[i].bare_type().is_int_type()) {
          std::stringstream ss;
          ss << "Found int-valued param: " << vs[i].name()
             << "; illegal - params must be real-valued" << EOL;
          generate_comment(ss.str(), indent, o);
          continue;
        }

        // setup - name, type, and var shape, loop var names
        std::string var_name(vs[i].name());
        // unfold array type to get array element info
        block_var_type btype = (vs[i].type());
        if (btype.is_array_type())
          btype = btype.array_contains();

        std::string cpp_type_str = get_verbose_var_type(btype.bare_type());
        std::string constrain_str = get_constrain_fn_prefix(btype);
        // dimension sizes and type - array or matrix/vec rows, columns
        std::vector<expression> ar_lens(vs[i].type().array_lens());
        // use parallel arrays of index names and limits for constraint loop
        std::vector<std::string> dim_size_var_names;
        std::vector<std::string> idx_names;
        for (size_t k = 0; k < ar_lens.size(); ++k) {
          std::stringstream ss;
          ss << "dim" << var_name << "_" << k << "__";
          dim_size_var_names.push_back(ss.str());
          ss.str(std::string());
          ss.clear();
          ss << "k_" << k << "__";
          idx_names.push_back(ss.str());
        }

        // generate declaration
        if (declare_vars) {
          generate_indent(indent, o);
          generate_bare_type(btype.bare_type(), cpp_type_str, o);
          o << " " << var_name << ";" << EOL;
        }

        // void stmt or for loop for array types
        if (ar_lens.size() == 0)
          generate_void_statement(var_name, indent, o);
        else {
          // dynamic initialization for array types
           for (size_t k = 0; k < ar_lens.size(); ++k) {
            // declare dim_size_var
            generate_indent(indent + k, o);
            o << "size_t " << dim_size_var_names[k] << " = ";
            generate_expression(ar_lens[k], NOT_USER_FACING, o);
            o << EOL;
            // resize
            generate_indent(indent + k, o);
            o << var_name;
            if (k > 0)
              o << "[" << idx_names[k-1] << "]";
            o << ".resize(" << dim_size_var_names[k] << ");" << EOL;
            // open for loop
            generate_indent(indent + k, o);
            o << "for (size_t " << idx_names[k] << "; "
              << idx_names[k] << " < " << dim_size_var_names[k]
              << "; ++" << idx_names[k] << ") {" << EOL;
          }
        }

        // constrain element
        generate_indent(indent + ar_lens.size(), o);
        o << "if (jacobian__)" << EOL;

        generate_indent(indent + ar_lens.size() + 1, o);
        o << var_name;
        for (size_t k = 1; k < ar_lens.size(); ++k)
          o << "[" << idx_names[k-1] << "]";
        o << ".push_back(in__." << constrain_str << ", lp__));" << EOL;

        generate_indent(indent + ar_lens.size(), o);
        o << "else" << EOL;

        generate_indent(indent + ar_lens.size() + 1, o);
        o << var_name;
        for (size_t k = 1; k < ar_lens.size(); ++k)
          o << "[" << idx_names[k-1] << "]";
        o << ".push_back(in__." << constrain_str << "));" << EOL;

        // close for loop for array types
        for (size_t k = ar_lens.size(); k > 0; --k) {
          generate_indent(indent + k - 1, o);
          o << "}" << EOL;
        }
        o << EOL;
      }
      o << EOL;
    }
    
  }
}
#endif

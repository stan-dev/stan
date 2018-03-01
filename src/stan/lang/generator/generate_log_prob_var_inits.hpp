#ifndef STAN_LANG_GENERATOR_GENERATE_LOG_PROB_VAR_INITS_HPP
#define STAN_LANG_GENERATOR_GENERATE_LOG_PROB_VAR_INITS_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <stan/lang/generator/get_verbose_var_type.hpp>
#include <stan/lang/generator/write_var_decl_type.hpp>
#include <ostream>
#include <vector>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Generate the name of the read function together
     * with expressions for the bounds parameters, if any
     *
     * NOTE: expecting that parser disallows integer params.
     *
     * @param[in,out] o stream for generating
     */
    std::string constrain_fn(const block_var_type& btype) {
      std::stringstream ss;
      if (btype.bare_type().is_double_type())
        ss << "scalar";
      else
        ss << btype.name();
      if (btype.has_def_bounds()) {
        if (btype.bounds().has_low() && btype.bounds().has_high()) {
          ss << "_lub_constrain(";
          generate_expression(btype.bounds().low_.expr_, NOT_USER_FACING, ss);
          ss << ", ";
          generate_expression(btype.bounds().high_.expr_, NOT_USER_FACING, ss);
          if (!is_nil(btype.arg1()))
            ss << ", ";
        } else if (btype.bounds().has_low()) {
          ss << "_lb_constrain(";
          generate_expression(btype.bounds().low_.expr_, NOT_USER_FACING, ss);
          if (!is_nil(btype.arg1()))
            ss << ", ";
        } else {
          ss << "_ub_constrain(";
          generate_expression(btype.bounds().high_.expr_, NOT_USER_FACING, ss);
          if (!is_nil(btype.arg1()))
            ss << ", ";
        }
      } else {
        ss << "_constrain(";
      }
      if (!is_nil(btype.arg1())) {
        generate_expression(btype.arg1(), NOT_USER_FACING, ss);
      }
      if (!is_nil(btype.arg2())) {
        ss << ", ";
        generate_expression(btype.arg2(), NOT_USER_FACING, ss);
      }
      return ss.str();
    }

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
        std::string constrain_str = constrain_fn(btype);
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
          write_var_decl_type(btype.bare_type(), cpp_type_str,
                              ar_lens.size(), indent, o);
          o << " " << var_name << ";" << EOL;
        }

        if (!btype.bare_type().is_array_type())
          generate_void_statement(var_name, indent, o);
        else {
          // dynamic initialization for array types
           for (size_t k = 0; k < ar_lens.size(); ++k) {
            // declare dim_size_var
            generate_indent(indent + k, o);
            o << "size_t " << dim_size_var_names[i] << " = ";
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
            o << "for(size_t " << idx_names[k] << "; "
              << idx_names[k] << " < " << dim_size_var_names[k]
              << "; ++" << idx_names[k] << ") {" << EOL;
          }
        }

        // innermost dimension - apply constraint
        generate_indent(indent + ar_lens.size(), o);
        o << "if (jacobian__)" << EOL;
        generate_indent(indent + ar_lens.size() + 1, o);
        o << var_name;
        for (size_t k = 1; k < ar_lens.size(); ++k)
          o << "[" << idx_names[k-1] << "]";
        o << ".push_back(in__." << constrain_str << ", lp__));" << EOL;
        generate_indent(indent + ar_lens.size(), o);
        o << "else" << EOL;
        // w/o Jacobian
        generate_indent(indent + ar_lens.size() + 1, o);
        o << var_name;
        for (size_t k = 1; k < ar_lens.size(); ++k)
          o << "[" << idx_names[k-1] << "]";
        o << ".push_back(in__." << constrain_str << "));" << EOL;

        // close brackets
        for (size_t k = ar_lens.size(); k > 0; --k) {
          generate_indent(indent + k - 1, o);
          o << "}" << EOL << EOL;
        }
        o << EOL; 
      }
    }
    
  }
}
#endif

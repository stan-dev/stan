#ifndef STAN_LANG_GENERATOR_GENERATE_LOG_PROB_VAR_INITS_HPP
#define STAN_LANG_GENERATOR_GENERATE_LOG_PROB_VAR_INITS_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <ostream>
#include <vector>

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
    std::string read_fn(const block_var_type& btype) {
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
            ss << ", ";
          } else if (btype.bounds().has_low()) {
            ss << "_lb_constrain(";
            generate_expression(btype.bounds().low_.expr_, NOT_USER_FACING, ss);
            ss << ", ";
          } else {
            ss << "_ub_constrain(";
            generate_expression(btype.bounds().high_.expr_, NOT_USER_FACING, ss);
            ss << ", ";
          }
        } else {
          ss << "_constrain(";
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
      for (size_t i = 0; i < vs.size(); ++i)
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

        // setup - name, type, and var shape
        std::string var_name(vs[i].name());
        // unfold array type to get array element info
        block_var_type btype = (vs[i].type());
        if (btype.is_array_type())
          btype = btype.array_contains();
        // dimension sizes and type - array or matrix/vec rows, columns
        std::vector<expression> ar_lens(vs[i].type().array_lens());
        expression arg1 = btype.arg1();
        expression arg2 = btype.arg2();
        // use parallel arrays of index names and limits for fill vals_r__ loop
        std::vector<std::string> idx_names;
        std::vector<expression> limits;
        int start_ar_idx = (!is_nil(arg2)) ? 2 : ((!is_nil(arg1)) ? 1 : 0);
        if (start_ar_idx == 2) {
          idx_names.push_back("j2__");
          limits.push_back(arg2);
        }
        if (start_ar_idx > 0) {
          idx_names.push_back("j1__");
          limits.push_back(arg1);
        }
        for (size_t i = ar_lens.size(); i > 0; --i) {
          std::stringstream ss_name;
          ss_name << "i" << i - 1 << "__";
          idx_names.push_back(ss_name.str());
          limits.push_back(ar_lens[i-1]);
        }
        // generate declaration
        if (declare_vars) {
          generate_indent(indent, o);
          write_var_decl_type(btype.bare_type(), ar_lens.size(), indent, o);
          o << " " << var_name << ";" << EOL;
        }

        if (btype.bare_type.is_double_type())
          // scalar param
          generate_void_statement(var_name, indent, o);
        else {
          // for each dimension - 
          // decl size_t
          // reserve
          // index
        }
        // innermost dimension
        generate_indent(indent, o);
        o << "if (jacobian__)" << EOL;
        // w Jacobian
        generate_indent(indent + 1, o);
        // generate var name
        // generate index
        // o << ".push_back(in__." << read_fn << ",lp__); << EOL
        generate_indent(indent, o);
        o << "else" << EOL;
        // w/o Jacobian
        generate_indent(indent + 1, o);
        // generate var name
        // generate index
        // o << ".push_back(in__." << read_fn << "); << EOL

      }
    }

  }
}
#endif

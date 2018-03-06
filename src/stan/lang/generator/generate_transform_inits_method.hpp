#ifndef STAN_LANG_GENERATOR_GENERATE_TRANSFORM_INITS_METHOD_HPP
#define STAN_LANG_GENERATOR_GENERATE_TRANSFORM_INITS_METHOD_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_comment.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <stan/lang/generator/generate_validate_context_size.hpp>
#include <stan/lang/generator/get_typedef_var_type.hpp>
#include <stan/lang/generator/get_verbose_var_type.hpp>
#include <stan/lang/generator/write_var_decl_type.hpp>
#include <stan/lang/generator/write_var_decl_arg.hpp>
#include <iostream>
#include <ostream>
#include <vector>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Generate the name of the unconstrain function together
     * with expressions for the bounds parameters, if any
     *
     * NOTE: expecting that parser disallows integer params.
     *
     * @param[in,out] o stream for generating
     */
    std::string unconstrain_fn(const block_var_type& btype) {
        std::stringstream ss;
        if (btype.bare_type().is_double_type())
          ss << "scalar";
        else
          ss << btype.name();
        if (btype.has_def_bounds()) {
          if (btype.bounds().has_low() && btype.bounds().has_high()) {
            ss << "_lub_unconstrain(";
            generate_expression(btype.bounds().low_.expr_, NOT_USER_FACING, ss);
            ss << ", ";
            generate_expression(btype.bounds().high_.expr_, NOT_USER_FACING, ss);
            ss << ", ";
          } else if (btype.bounds().has_low()) {
            ss << "_lb_unconstrain(";
            generate_expression(btype.bounds().low_.expr_, NOT_USER_FACING, ss);
            ss << ", ";
          } else {
            ss << "_ub_unconstrain(";
            generate_expression(btype.bounds().high_.expr_, NOT_USER_FACING, ss);
            ss << ", ";
          }
        } else {
          ss << "_unconstrain(";
        }
        return ss.str();
    }

    /**
     * Generate the <code>transform_inits</code> method declaration
     * and variable decls.
     *
     * @param[in,out] o stream for generating
     */
    void generate_method_begin(std::ostream& o) {
      o << EOL;
      o << INDENT
        << "void transform_inits(const stan::io::var_context& context__,"
        << EOL;
      o << INDENT << "                     std::vector<double>& params_r__,"
        << EOL;
      o << INDENT << "                     std::ostream* pstream__) const {"
        << EOL;
      o << INDENT2 << "stan::io::writer<double> "
        << "writer__(params_r__);"        
        << EOL;

      o << INDENT2 << "size_t pos__;" << EOL;
      o << INDENT2 << "(void) pos__; // dummy call to supress warning" << EOL;
      o << INDENT2 << "std::vector<double> vals_r__;" << EOL;
    }

    /**
     * Generate the <code>transform_inits</code> method declaration final statements
     * and close.
     *
     * @param[in,out] o stream for generating
     */
    void generate_method_end(std::ostream& o) {
      o << INDENT2 << "params_r__ = writer__.data_r();" << EOL;
      o << INDENT << "}" << EOL2;

      o << INDENT
        << "void transform_inits(const stan::io::var_context& context," << EOL;
      o << INDENT
        << "                     "
        << "Eigen::Matrix<double, Eigen::Dynamic, 1>& params_r," << EOL;
      o << INDENT
        << "                     std::ostream* pstream__) const {" << EOL;
      o << INDENT << "  std::vector<double> params_r_vec;" << EOL;
      o << INDENT
        << "  transform_inits(context, params_r_vec, pstream__);"
        << EOL;
      o << INDENT << "  params_r.resize(params_r_vec.size());" << EOL;
      o << INDENT << "  for (int i = 0; i < params_r.size(); ++i)" << EOL;
      o << INDENT << "    params_r(i) = params_r_vec[i];" << EOL;
      o << INDENT << "}" << EOL2;
    }
    
    /**
     * Generate the <code>transform_inits</code> method for the
     * specified parameter variable declarations to the specified stream.
     *
     * @param[in] vs variable declarations
     * @param[in,out] o stream for generating
     */
    void generate_transform_inits_method(const std::vector<block_var_decl>& vs,
                              std::ostream& o) {
      int indent = 2;

      generate_method_begin(o);
      o << EOL;
      
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

        // setup - name, type, and var shape
        std::string var_name(vs[i].name());
        // unfold array type to get array element info
        block_var_type btype = (vs[i].type());
        if (btype.is_array_type())
          btype = btype.array_contains();
        std::string cpp_type_str = get_typedef_var_type(btype.bare_type());
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

        generate_indent(indent, o);
        o << "current_statement_begin__ = " <<  vs[i].begin_line_ << ";"
          << EOL;


        // check context
        generate_indent(indent, o);
        o << "if (!(context__.contains_r(\""
          << var_name << "\")))" << EOL;
        generate_indent(indent + 1, o);
        o << "stan::lang::rethrow_located("
          << "std::runtime_error(std::string(\"Variable "
          << var_name
          << "missing\")), current_statement_begin__, prog_reader__());"
          << EOL;
        // init context position
        generate_indent(indent, o);
        o << "vals_r__ = context__.vals_r(\""
          << var_name << "\");" << EOL;
        generate_indent(indent, o);
        o << "pos__ = 0U;" << EOL;
        
        // validate dimensions before they get used in declaration
        generate_validate_context_size(vs[i], "parameter initialization",
                                       indent, o);

        // declaration
        write_var_decl_type(btype.bare_type(), cpp_type_str,
                            ar_lens.size(), indent, o);
        o << " " << var_name;
        write_var_decl_arg(btype.bare_type(), cpp_type_str,
                             ar_lens, btype.arg1(), btype.arg2(), o);
        o << ";" << EOL;

        // fill vals_r__ buffer loop
        // indexes innermost scalar element of (array of) matrix/vec types
        for (size_t i = 0; i < idx_names.size(); ++i) {
          generate_indent(indent + i, o);
          o << "for (int " << idx_names[i] << " = 0U; "
            << idx_names[i] << " < ";
          generate_expression(limits[i], NOT_USER_FACING, o);
          o << "; ++" << idx_names[i] << ")" << EOL;
        }
        // fill vals_r__ stmt
        generate_indent(indent + idx_names.size(), o);
        o << var_name;
        for (size_t i = start_ar_idx; i < idx_names.size(); ++i)
          o << "[" << idx_names[i] << "]";
        if (start_ar_idx == 2)
          o << "(j1__,j2__)";
        else if (start_ar_idx == 1)
          o << "(j1__)";
        o << " = vals_r__[pos__++];" << EOL;
          
        // write loop for each array element (if array var)
        for (size_t i = 0; i < ar_lens.size(); ++i) {
          generate_indent(indent + i, o);
          o << "for (int i" << i << "__ = 0U; i" << i << "__ < ";
          generate_expression(ar_lens[i], NOT_USER_FACING, o);
          o << "; __" << i << "__)" << EOL;
        }
        // generate call to unconstrain fn in try/catch block
        generate_indent(indent, o);
        o << "try {" << EOL;
        generate_indent(indent + 1, o);
        o << "writer__." << unconstrain_fn(btype) << var_name;
        for (size_t i = 0; i < ar_lens.size(); ++i) 
          o << "[i" << i << "__]";
        o << ");" << EOL;
        generate_indent(indent, o);
        o << "} catch (const std::exception& e) {" << EOL;
        generate_indent(indent + 1, o);
        o << "stan::lang::rethrow_located("
          << "std::runtime_error(std::string(\"Error transforming variable "
          << var_name
          << ": \") + e.what()), current_statement_begin__, prog_reader__());"
          << EOL;
        generate_indent(indent + 1, o);
        o << "// Next line prevents compiler griping about no return" << EOL;
        generate_indent(indent + 1, o);
        o << "throw std::runtime_error"
          << "(\"*** IF YOU SEE THIS, PLEASE REPORT A BUG ***\");"
          << EOL;
        generate_indent(indent, o);
        o << "}" << EOL << EOL;
      }

      generate_method_end(o);
    }

  }
}
#endif

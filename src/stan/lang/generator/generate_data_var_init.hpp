#ifndef STAN_LANG_GENERATOR_GENERATE_DATA_VAR_INIT_HPP
#define STAN_LANG_GENERATOR_GENERATE_DATA_VAR_INIT_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_expression.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <iostream>
#include <ostream>
#include <vector>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Generate initializations for data block variables by reading
     * dump format data from constructor variable context.
     * In dump format data, arrays are indexed in last-index major fashion,
     * which corresponds to column-major order for matrices
     * represented as two-dimensional arrays.  As a result, the first
     * indices change fastest.  Therefore loops must be constructed:
     * (col) (row) (array-dim-N) ... (array-dim-1)
     *
     * @param[in] vs data block variable declarations
     * @param[in] indent indentation level
     * @param[in,out] o stream for generating
     */
    void generate_data_var_init(const block_var_decl& var_decl,
                                int indent, std::ostream& o) {

      // setup - name, type, and var shape
      std::string var_name(var_decl.name());
      // unfold array type to get array element info
      block_var_type btype = (var_decl.type());
      if (btype.is_array_type())
        btype = btype.array_contains();
      // dimension sizes and type - array or matrix/vec rows, columns
      std::vector<expression> ar_lens(var_decl.type().array_lens());
      expression arg1 = btype.arg1();
      expression arg2 = btype.arg2();
      // use parallel arrays of index names and limits for read loop
      std::vector<std::string> idx_names;
      std::vector<expression> limits;
      size_t start_ar_idx = (!is_nil(arg2)) ? 2 : ((!is_nil(arg1)) ? 1 : 0);
      if (start_ar_idx == 2) {
        idx_names.push_back("j_2__");
        limits.push_back(arg2);
      }
      if (start_ar_idx > 0) {
        idx_names.push_back("j_1__");
        limits.push_back(arg1);
      }
      for (size_t i = ar_lens.size(); i > 0; --i) {
        std::stringstream ss_name;
        ss_name << "k_" << i - 1 << "__";
        idx_names.push_back(ss_name.str());
        limits.push_back(ar_lens[i - 1]);
      }
      std::string vals("vals_r");
      if (btype.bare_type().is_int_type())
        vals = "vals_i";

      generate_indent(indent, o);
      o << vals << "__ = context__." << vals << "(\"" << var_name << "\");" << EOL;
      generate_indent(indent, o);
      o << "pos__ = 0;" << EOL;

      // nested for stmts open
      int indentation = indent;
      for (size_t i = 0; i < limits.size(); ++i) {
        generate_indent(indentation, o);
        o << "for (size_t " << idx_names[i] << " = 0; "
          << idx_names[i] << " < ";
        generate_expression(limits[i], NOT_USER_FACING, o);
        o << "; ++" << idx_names[i] << ") {" << EOL;
        indentation++;                                    
      }               
      // innermost assign - update pos__
      generate_indent(indentation, o);
      o << var_name;
      // array idxs
      for (size_t i = limits.size(); i > start_ar_idx; --i)
        o << "[" << idx_names[i-1] << "]";
      if (start_ar_idx == 2) {
        o << "(" << idx_names[1] << ", " << idx_names[0] << ")";
      } else if (start_ar_idx == 1) {
        o << "(" << idx_names[0] << ")";
      }
      o << " = " << vals << "__[pos__++]; " << EOL;

      // nested close
      for (size_t i = 0; i < limits.size(); ++i) {
        indentation--;                                    
        generate_indent(indentation, o);
        o << "}" << EOL;
      }        
    }

  }
}
#endif

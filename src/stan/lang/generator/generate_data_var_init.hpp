#ifndef STAN_LANG_GENERATOR_GENERATE_DATA_VAR_INIT_HPP
#define STAN_LANG_GENERATOR_GENERATE_DATA_VAR_INIT_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_expression.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <stan/lang/generator/write_end_loop.hpp>
#include <stan/lang/generator/write_nested_read_loop_begin.hpp>
#include <stan/lang/generator/write_var_idx_all_dims.hpp>
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
      block_var_type btype = (var_decl.type());
      if (btype.is_array_type())
        btype = btype.array_contains();
      std::vector<expression> ar_lens(var_decl.type().array_lens());
      expression arg1 = btype.arg1();
      expression arg2 = btype.arg2();

      // combine all dimension sizes in column major order
      std::vector<expression> dims;
      size_t num_args = (!is_nil(arg2)) ? 2 : ((!is_nil(arg1)) ? 1 : 0);
      if (num_args == 2) 
        dims.push_back(arg2);
      if (num_args > 0)
        dims.push_back(arg1);
      for (size_t i = ar_lens.size(); i > 0; --i)
        dims.push_back(ar_lens[i - 1]);

      std::string vals("vals_r");
      if (btype.bare_type().is_int_type())
        vals = "vals_i";

      generate_indent(indent, o);
      o << vals << "__ = context__." << vals << "(\"" << var_name << "\");" << EOL;
      generate_indent(indent, o);
      o << "pos__ = 0;" << EOL;
      
      write_nested_read_loop_begin(var_name, dims, num_args, indent, o);

      // innermost loop stmt: update pos__
      generate_indent(indent + dims.size(), o);
      o << var_name;
      write_var_idx_all_dims(ar_lens.size(), num_args, o);
      o << " = " << vals << "__[pos__++]; " << EOL;

      write_end_loop(dims.size(), indent, o);
    }        

  }
}
#endif

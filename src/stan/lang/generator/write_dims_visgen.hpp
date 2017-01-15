#ifndef STAN_LANG_GENERATOR_WRITE_DIMS_VISGEN_HPP
#define STAN_LANG_GENERATOR_WRITE_DIMS_VISGEN_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/visgen.hpp>
#include <ostream>
#include <vector>

namespace stan {
  namespace lang {


    void generate_expression(const expression& e, std::ostream& o);

    /**
     * Visitor for writing the dimensions of variables.
     */
    struct write_dims_visgen : public visgen {
      /**
       * Construct a dimension writer visitor for the specified
       * stream.
       *
       * @param[in,out] o stream for writing
       */
      explicit write_dims_visgen(std::ostream& o) : visgen(o) {  }

      /**
       * Generate the array of dimensions for the specified sequences
       * of matrix dimension sizes and array dimension sizes.
       *
       * @param[in] matrix_dims matrix dimension sizes
       * @param[in] array_dims array dimension sizes
       */
      void generate_dims_array(const std::vector<expression>& matrix_dims,
                               const std::vector<expression>& array_dims)
        const {
        o_ << INDENT2 << "dims__.resize(0);" << EOL;
        for (size_t i = 0; i < array_dims.size(); ++i) {
          o_ << INDENT2 << "dims__.push_back(";
          generate_expression(array_dims[i].expr_, o_);
          o_ << ");" << EOL;
        }
        for (size_t i = 0; i < matrix_dims.size(); ++i) {
          o_ << INDENT2 << "dims__.push_back(";
          generate_expression(matrix_dims[i].expr_, o_);
          o_ << ");" << EOL;
        }
        o_ << INDENT2 << "dimss__.push_back(dims__);" << EOL;
      }

      void operator()(const nil& /*x*/) const  { }

      void operator()(const int_var_decl& x) const {
        generate_dims_array(EMPTY_EXP_VECTOR, x.dims_);
      }

      void operator()(const double_var_decl& x) const {
        generate_dims_array(EMPTY_EXP_VECTOR, x.dims_);
      }

      void operator()(const vector_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.M_);
        generate_dims_array(matrix_args, x.dims_);
      }

      void operator()(const row_vector_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.N_);
        generate_dims_array(matrix_args, x.dims_);
      }

      void operator()(const matrix_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.M_);
        matrix_args.push_back(x.N_);
        generate_dims_array(matrix_args, x.dims_);
      }

      void operator()(const unit_vector_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        generate_dims_array(matrix_args, x.dims_);
      }

      void operator()(const simplex_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        generate_dims_array(matrix_args, x.dims_);
      }

      void operator()(const ordered_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        generate_dims_array(matrix_args, x.dims_);
      }

      void operator()(const positive_ordered_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        generate_dims_array(matrix_args, x.dims_);
      }

      void operator()(const cholesky_factor_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.M_);
        matrix_args.push_back(x.N_);
        generate_dims_array(matrix_args, x.dims_);
      }

      void operator()(const cholesky_corr_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        matrix_args.push_back(x.K_);
        generate_dims_array(matrix_args, x.dims_);
      }

      void operator()(const cov_matrix_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        matrix_args.push_back(x.K_);
        generate_dims_array(matrix_args, x.dims_);
      }

      void operator()(const corr_matrix_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        matrix_args.push_back(x.K_);
        generate_dims_array(matrix_args, x.dims_);
      }
    };

  }
}
#endif

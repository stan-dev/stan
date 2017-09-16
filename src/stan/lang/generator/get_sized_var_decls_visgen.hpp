#ifndef STAN_LANG_GENERATOR_GET_SIZED_VAR_DECLS_VISGEN_HPP
#define STAN_LANG_GENERATOR_GET_SIZED_VAR_DECLS_VISGEN_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/generate_array_builder_adds.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/visgen.hpp>
#include <ostream>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Visitor for generating code to push static variable
     * declarations onto an accumulator.  return static variable
     */
    struct get_sized_var_decls_visgen : public visgen {
      /**
       * Construct a get variable declarations visitor for the
       * specified stream.  This one is for the runtime method that
       * returns sizes based on expressions in the declaration.
       *
       * @param[in,out] o stream for generating
       */
      get_sized_var_decls_visgen(std::ostream& o)
        : visgen(3, o) { }

      void push_back_var_decl(const std::string& type_name,
                              const std::string& name,
                              const std::vector<expression>& dims) const {
        o_ << INDENT3 << "decls__.push_back(sized_var_decl("
           << "\"" << name << "\", "
           << "\"" << type_name << "\", "
           << dims.size();

        if (dims.size() == 0) {
          o_ << "));" << std::endl;
          return;
        }
        o_ << ", " << std::endl
           << INDENT3 << INDENT << "stan::math::array_builder<int>()";
        if (dims.size() > 0) o_ << std::endl;
        o_ << INDENT3 << INDENT;

        bool is_user_facing = false;
        bool is_var_context = false;
        generate_array_builder_adds(dims, is_user_facing, is_var_context, o_);
        o_ << std::endl << INDENT3 << INDENT << ".array()));" << std::endl;
      }

      void push_back_var_decl(const std::string& type_name,
                              const std::string& name,
                              const std::vector<expression>& dims,
                              const expression& dim) const {
        std::vector<expression> dims_plus(dims);
        dims_plus.push_back(dim);
        push_back_var_decl(type_name, name, dims_plus);
      }

      void push_back_var_decl(const std::string& type_name,
                              const std::string& name,
                              const std::vector<expression>& dims,
                              const expression& dim1,
                              const expression& dim2) const {
        std::vector<expression> dims_plus(dims);
        dims_plus.push_back(dim1);
        dims_plus.push_back(dim2);
        push_back_var_decl(type_name, name, dims_plus);
      }

      void operator()(const nil& /*x*/) const { }

      void operator()(const int_var_decl& x) const {
        push_back_var_decl("int", x.name_, x.dims_);
      }

      void operator()(const double_var_decl& x) const {
        push_back_var_decl("real", x.name_, x.dims_);
      }

      void operator()(const unit_vector_var_decl& x) const {
        push_back_var_decl("unit_vector", x.name_, x.dims_, x.K_);
      }

      void operator()(const simplex_var_decl& x) const {
        push_back_var_decl("simplex", x.name_, x.dims_, x.K_);
      }

      void operator()(const ordered_var_decl& x) const {
        push_back_var_decl("ordered", x.name_, x.dims_, x.K_);
      }

      void operator()(const positive_ordered_var_decl& x) const {
        push_back_var_decl("positive_ordered", x.name_, x.dims_, x.K_);
      }

      void operator()(const vector_var_decl& x) const {
        push_back_var_decl("vector", x.name_, x.dims_, x.M_);
      }

      void operator()(const row_vector_var_decl& x) const {
        push_back_var_decl("row_vector", x.name_, x.dims_, x.N_);
      }

      void operator()(const matrix_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        push_back_var_decl("matrix", x.name_, x.dims_, x.M_, x.N_);
      }

      void operator()(const cholesky_factor_var_decl& x) const {
        push_back_var_decl("cholesky_factor_cov", x.name_, x.dims_,
                           x.M_, x.N_);
      }

      void operator()(const cholesky_corr_var_decl& x) const {
        push_back_var_decl("cholesky_factor_corr", x.name_, x.dims_,
                           x.K_, x.K_);
      }

      void operator()(const cov_matrix_var_decl& x) const {
        push_back_var_decl("cov_matrix", x.name_, x.dims_, x.K_, x.K_);
      }

      void operator()(const corr_matrix_var_decl& x) const {
        push_back_var_decl("corr_matrix", x.name_, x.dims_, x.K_, x.K_);
      }
    };

  }
}
#endif

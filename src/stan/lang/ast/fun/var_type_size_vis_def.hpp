#ifndef STAN_LANG_AST_FUN_VAR_TYPE_SIZE_VIS_DEF_HPP
#define STAN_LANG_AST_FUN_VAR_TYPE_SIZE_VIS_DEF_HPP

#include <stan/lang/ast.hpp>
#include <boost/variant/apply_visitor.hpp>

namespace stan {
  namespace lang {
    var_type_size_vis::var_type_size_vis() { }

    std::vector<expression>
    var_type_size_vis::operator()(const block_array_type& x)
      const {
      std::vector<expression> sizes(x.array_lens());
      std::vector<expression> base_sizes = x.contains().size();
      for (size_t i=0; i < base_sizes.size(); ++i) {
        sizes.push_back(base_sizes[i]);
      }
      return sizes;
    }

    std::vector<expression>
    var_type_size_vis::operator()(const local_array_type& x)
      const {
      std::vector<expression> sizes;
      sizes.push_back(x.array_len_);
      local_var_type cur_type(x.element_type_);
      while (cur_type.is_array_type()) {
        sizes.push_back(cur_type.array_len());
        cur_type = cur_type.array_element_type();
      }
      std::vector<expression> base_sizes = cur_type.size();
      for (size_t i=0; i < base_sizes.size(); ++i) {
        sizes.push_back(base_sizes[i]);
      }
      return sizes;
    }

    std::vector<expression>
    var_type_size_vis::operator()(const cholesky_factor_corr_block_type& x)
      const {
      std::vector<expression> sizes;
      sizes.push_back(x.K_);
      sizes.push_back(x.K_);
      return sizes;
    }

    std::vector<expression>
    var_type_size_vis::operator()(const cholesky_factor_cov_block_type& x)
      const {
      std::vector<expression> sizes;
      sizes.push_back(x.M_);
      sizes.push_back(x.N_);
      return sizes;
    }

    std::vector<expression>
    var_type_size_vis::operator()(const corr_matrix_block_type& x)
      const {
      std::vector<expression> sizes;
      sizes.push_back(x.K_);
      sizes.push_back(x.K_);
      return sizes;
    }

    std::vector<expression>
    var_type_size_vis::operator()(const cov_matrix_block_type& x)
      const {
      std::vector<expression> sizes;
      sizes.push_back(x.K_);
      sizes.push_back(x.K_);
      return sizes;
    }

    std::vector<expression>
    var_type_size_vis::operator()(const double_block_type& x)
      const {
      return std::vector<expression>();
    }

    std::vector<expression>
    var_type_size_vis::operator()(const double_type& x)
      const {
      return std::vector<expression>();
    }

    std::vector<expression>
    var_type_size_vis::operator()(const ill_formed_type& x)
      const {
      return std::vector<expression>();
    }

    std::vector<expression>
    var_type_size_vis::operator()(const int_block_type& x)
      const {
      return std::vector<expression>();
    }

    std::vector<expression>
    var_type_size_vis::operator()(const int_type& x)
      const {
      return std::vector<expression>();
    }

    std::vector<expression>
    var_type_size_vis::operator()(const matrix_block_type& x)
      const {
      std::vector<expression> sizes;
      sizes.push_back(x.M_);
      sizes.push_back(x.N_);
      return sizes;
    }

    std::vector<expression>
    var_type_size_vis::operator()(const matrix_local_type& x)
      const {
      std::vector<expression> sizes;
      sizes.push_back(x.M_);
      sizes.push_back(x.N_);
      return sizes;
    }

    std::vector<expression>
    var_type_size_vis::operator()(const ordered_block_type& x)
      const {
      std::vector<expression> sizes;
      sizes.push_back(x.K_);
      return sizes;
    }

    std::vector<expression>
    var_type_size_vis::operator()(const positive_ordered_block_type& x)
      const {
      std::vector<expression> sizes;
      sizes.push_back(x.K_);
      return sizes;
    }

    std::vector<expression>
    var_type_size_vis::operator()(const row_vector_block_type& x)
      const {
      std::vector<expression> sizes;
      sizes.push_back(x.N_);
      return sizes;
    }

    std::vector<expression>
    var_type_size_vis::operator()(const row_vector_local_type& x)
      const {
      std::vector<expression> sizes;
      sizes.push_back(x.N_);
      return sizes;
    }

    std::vector<expression>
    var_type_size_vis::operator()(const simplex_block_type& x)
      const {
      std::vector<expression> sizes;
      sizes.push_back(x.K_);
      return sizes;
    }

    std::vector<expression>
    var_type_size_vis::operator()(const unit_vector_block_type& x)
      const {
      std::vector<expression> sizes;
      sizes.push_back(x.K_);
      return sizes;
    }

    std::vector<expression>
    var_type_size_vis::operator()(const vector_block_type& x)
      const {
      std::vector<expression> sizes;
      sizes.push_back(x.N_);
      return sizes;
    }

    std::vector<expression>
    var_type_size_vis::operator()(const vector_local_type& x)
      const {
      std::vector<expression> sizes;
      sizes.push_back(x.N_);
      return sizes;
    }
  }
}
#endif

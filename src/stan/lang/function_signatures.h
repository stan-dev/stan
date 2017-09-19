// included from constructor for function_signatures() in src/stan/lang/ast.hpp

std::vector<base_expr_type> base_types;
base_types.push_back(int_type());
base_types.push_back(double_type());
base_types.push_back(vector_type());
base_types.push_back(row_vector_type());
base_types.push_back(matrix_type());

std::vector<expr_type> vector_types;
vector_types.push_back(expr_type(double_type()));  // scalar
vector_types.push_back(expr_type(double_type(), 1U));  // std vector
vector_types.push_back(expr_type(vector_type()));  // Eigen vector
vector_types.push_back(expr_type(row_vector_type()));  // Eigen row vector

std::vector<expr_type> int_vector_types;
int_vector_types.push_back(expr_type(int_type()));  // scalar
int_vector_types.push_back(expr_type(int_type(), 1U));  // std vector

std::vector<expr_type> primitive_types;
primitive_types.push_back(expr_type(int_type()));
primitive_types.push_back(expr_type(double_type()));

add("abs", expr_type(int_type()), expr_type(int_type()));
add("abs", expr_type(double_type()), expr_type(double_type()));
add_unary_vectorized("acos");
add_unary_vectorized("acosh");
for (size_t i = 0; i < base_types.size(); ++i) {
  add("add", base_types[i], base_types[i], base_types[i]);
}
add("add", expr_type(vector_type()), expr_type(vector_type()), expr_type(double_type()));
add("add", expr_type(row_vector_type()), expr_type(row_vector_type()), expr_type(double_type()));
add("add", expr_type(matrix_type()), expr_type(matrix_type()), expr_type(double_type()));
add("add", expr_type(vector_type()), expr_type(double_type()), expr_type(vector_type()));
add("add", expr_type(row_vector_type()), expr_type(double_type()), expr_type(row_vector_type()));
add("add", expr_type(matrix_type()), expr_type(double_type()), expr_type(matrix_type()));
for (size_t i = 0; i < base_types.size(); ++i) {
  add("add", base_types[i], base_types[i]);
}
for (size_t i = 1; i < 8; ++i) {
  add("append_array", expr_type(INT_T, i), expr_type(INT_T, i), expr_type(INT_T, i));
  add("append_array", expr_type(DOUBLE_T, i), expr_type(DOUBLE_T, i), expr_type(DOUBLE_T, i));
  add("append_array", expr_type(VECTOR_T, i), expr_type(VECTOR_T, i), expr_type(VECTOR_T, i));
  add("append_array", expr_type(ROW_VECTOR_T, i), expr_type(ROW_VECTOR_T, i), expr_type(ROW_VECTOR_T, i));
  add("append_array", expr_type(MATRIX_T, i), expr_type(MATRIX_T, i), expr_type(MATRIX_T, i));
}
add_unary_vectorized("asin");
add_unary_vectorized("asinh");
add_unary_vectorized("atan");
add_binary("atan2");
add_unary_vectorized("atanh");
for (size_t i = 0; i < int_vector_types.size(); ++i)
  for (size_t j = 0; j < vector_types.size(); ++j) {
    add("bernoulli_ccdf_log", expr_type(double_type()), int_vector_types[i], 
	vector_types[j]);
    add("bernoulli_cdf", expr_type(double_type()), int_vector_types[i], 
	vector_types[j]);
    add("bernoulli_cdf_log", expr_type(double_type()), int_vector_types[i], 
	vector_types[j]);
    add("bernoulli_log", expr_type(double_type()), int_vector_types[i], 
	vector_types[j]);
    add("bernoulli_lccdf", expr_type(double_type()), int_vector_types[i], 
	vector_types[j]);
    add("bernoulli_lcdf", expr_type(double_type()), int_vector_types[i], 
	vector_types[j]);
    add("bernoulli_lpmf", expr_type(double_type()), int_vector_types[i], 
	vector_types[j]);
  }
add("bernoulli_rng", expr_type(int_type()), expr_type(double_type()));
add("bernoulli_logit_rng", expr_type(int_type()), expr_type(double_type()));
for (size_t i = 0; i < int_vector_types.size(); ++i)
  for (size_t j = 0; j < vector_types.size(); ++j) {
    add("bernoulli_logit_log", expr_type(double_type()), int_vector_types[i], 
	vector_types[j]);
    add("bernoulli_logit_lpmf", expr_type(double_type()), int_vector_types[i], 
	vector_types[j]);
  }
add("bessel_first_kind", expr_type(double_type()), expr_type(int_type()), expr_type(double_type()));
add("bessel_second_kind", expr_type(double_type()), expr_type(int_type()), expr_type(double_type()));
for (size_t i = 0; i < int_vector_types.size(); i++)
  for (size_t j = 0; j < int_vector_types.size(); j++)
    for (size_t k = 0; k < vector_types.size(); k++)
      for (size_t l = 0; l < vector_types.size(); l++) {
        add("beta_binomial_ccdf_log", expr_type(double_type()),
            int_vector_types[i], int_vector_types[j],
	    vector_types[k], vector_types[l]);
        add("beta_binomial_cdf", expr_type(double_type()),
            int_vector_types[i], int_vector_types[j],
	    vector_types[k], vector_types[l]);
        add("beta_binomial_cdf_log", expr_type(double_type()),
            int_vector_types[i], int_vector_types[j],
	    vector_types[k], vector_types[l]);
        add("beta_binomial_log", expr_type(double_type()),
            int_vector_types[i], int_vector_types[j],
	    vector_types[k], vector_types[l]);
        add("beta_binomial_lccdf", expr_type(double_type()), int_vector_types[i],
	    int_vector_types[j], vector_types[k], vector_types[l]);
        add("beta_binomial_lcdf", expr_type(double_type()),
            int_vector_types[i], int_vector_types[j],
	    vector_types[k], vector_types[l]);
        add("beta_binomial_lpmf", expr_type(double_type()),
            int_vector_types[i], int_vector_types[j],
	    vector_types[k], vector_types[l]);
      }
add("beta_binomial_rng", expr_type(int_type()), expr_type(int_type()), expr_type(double_type()), expr_type(double_type()));
for (size_t i = 0; i < vector_types.size(); ++i) {
  for (size_t j = 0; j < vector_types.size(); ++j) {
    for (size_t k = 0; k < vector_types.size(); ++k) {
      add("beta_ccdf_log", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("beta_cdf", expr_type(double_type()), vector_types[i], vector_types[j],
	  vector_types[k]);
      add("beta_cdf_log", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("beta_log", expr_type(double_type()), vector_types[i], vector_types[j],
	  vector_types[k]);
      add("beta_lccdf", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("beta_lcdf", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("beta_lpdf", expr_type(double_type()), vector_types[i], vector_types[j],
	  vector_types[k]);
    }
  }
}
add_binary("beta_rng");
add("binary_log_loss", expr_type(double_type()), expr_type(int_type()), expr_type(double_type()));
for (size_t i = 0; i < int_vector_types.size(); ++i) {
  for (size_t j = 0; j < int_vector_types.size(); ++j) {
    for (size_t k = 0; k < vector_types.size(); ++k) {
      add("binomial_ccdf_log", expr_type(double_type()), 
          int_vector_types[i], int_vector_types[j], vector_types[k]);
      add("binomial_cdf", expr_type(double_type()), 
          int_vector_types[i], int_vector_types[j], vector_types[k]);
      add("binomial_cdf_log", expr_type(double_type()), 
          int_vector_types[i], int_vector_types[j], vector_types[k]);
      add("binomial_log", expr_type(double_type()), 
          int_vector_types[i], int_vector_types[j], vector_types[k]);
      add("binomial_lccdf", expr_type(double_type()), 
          int_vector_types[i], int_vector_types[j], vector_types[k]);
      add("binomial_lcdf", expr_type(double_type()), 
          int_vector_types[i], int_vector_types[j], vector_types[k]);
      add("binomial_lpmf", expr_type(double_type()), 
          int_vector_types[i], int_vector_types[j], vector_types[k]);
    }
  }
}
add("binomial_rng", expr_type(int_type()), expr_type(int_type()), expr_type(double_type()));
add_binary("binomial_coefficient_log");
for (size_t i = 0; i < int_vector_types.size(); ++i) {
  for (size_t j = 0; j < int_vector_types.size(); ++j) {
    for (size_t k = 0; k < vector_types.size(); ++k) {
      add("binomial_logit_log", expr_type(double_type()), 
          int_vector_types[i], int_vector_types[j], vector_types[k]);
      add("binomial_logit_lpmf", expr_type(double_type()), 
          int_vector_types[i], int_vector_types[j], vector_types[k]);
    }
  }
}
add("block", expr_type(matrix_type()), expr_type(matrix_type()), expr_type(int_type()), expr_type(int_type()), expr_type(int_type()), expr_type(int_type()));
for (size_t i = 0; i < int_vector_types.size(); ++i) {
  add("categorical_log", expr_type(double_type()), int_vector_types[i], expr_type(vector_type()));
  add("categorical_logit_log", expr_type(double_type()), int_vector_types[i],
      expr_type(vector_type()));
  add("categorical_lpmf", expr_type(double_type()), int_vector_types[i], expr_type(vector_type()));
  add("categorical_logit_lpmf", expr_type(double_type()), int_vector_types[i],
      expr_type(vector_type()));
}
add("categorical_rng", expr_type(int_type()), expr_type(vector_type()));
add("categorical_logit_rng", expr_type(int_type()), expr_type(vector_type()));
for (size_t i = 0; i < vector_types.size(); ++i) {
  for (size_t j = 0; j < vector_types.size(); ++j) {
    for (size_t k = 0; k < vector_types.size(); ++k) {
      add("cauchy_ccdf_log", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("cauchy_cdf", expr_type(double_type()), vector_types[i], vector_types[j],
	  vector_types[k]);
      add("cauchy_cdf_log", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("cauchy_log", expr_type(double_type()), vector_types[i], vector_types[j],
	  vector_types[k]);
      add("cauchy_lccdf", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("cauchy_lcdf", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("cauchy_lpdf", expr_type(double_type()), vector_types[i], vector_types[j],
	  vector_types[k]);
    }
  }
 }
add_binary("cauchy_rng");
add("append_col", expr_type(matrix_type()), expr_type(matrix_type()), expr_type(matrix_type()));
add("append_col", expr_type(matrix_type()), expr_type(vector_type()), expr_type(matrix_type()));
add("append_col", expr_type(matrix_type()), expr_type(matrix_type()), expr_type(vector_type()));
add("append_col", expr_type(matrix_type()), expr_type(vector_type()), expr_type(vector_type()));
add("append_col", expr_type(row_vector_type()), expr_type(row_vector_type()), expr_type(row_vector_type()));
add("append_col", expr_type(row_vector_type()), expr_type(double_type()), expr_type(row_vector_type()));
add("append_col", expr_type(row_vector_type()), expr_type(row_vector_type()), expr_type(double_type()));
add_unary_vectorized("cbrt");
add_unary_vectorized("ceil");
for (size_t i = 0; i < vector_types.size(); ++i) {
  for (size_t j = 0; j < vector_types.size(); ++j) {
      add("chi_square_ccdf_log", expr_type(double_type()), vector_types[i],
	  vector_types[j]);
      add("chi_square_cdf", expr_type(double_type()), vector_types[i],
	  vector_types[j]);
      add("chi_square_cdf_log", expr_type(double_type()), vector_types[i],
	  vector_types[j]);
      add("chi_square_log", expr_type(double_type()), vector_types[i],
	  vector_types[j]);
      add("chi_square_lccdf", expr_type(double_type()), vector_types[i],
	  vector_types[j]);
      add("chi_square_lcdf", expr_type(double_type()), vector_types[i],
	  vector_types[j]);
      add("chi_square_lpdf", expr_type(double_type()), vector_types[i],
	  vector_types[j]);
  }
}
add_unary("chi_square_rng");
add("cholesky_decompose", expr_type(matrix_type()), expr_type(matrix_type()));
add("choose", expr_type(int_type()), expr_type(int_type()), expr_type(int_type()));
add("col", expr_type(vector_type()), expr_type(matrix_type()), expr_type(int_type()));
add("cols", expr_type(int_type()), expr_type(vector_type()));
add("cols", expr_type(int_type()), expr_type(row_vector_type()));
add("cols", expr_type(int_type()), expr_type(matrix_type()));
add("columns_dot_product", expr_type(row_vector_type()), expr_type(vector_type()), expr_type(vector_type()));
add("columns_dot_product", expr_type(row_vector_type()), expr_type(row_vector_type()), expr_type(row_vector_type()));
add("columns_dot_product", expr_type(row_vector_type()), expr_type(matrix_type()), expr_type(matrix_type()));
add("columns_dot_self", expr_type(row_vector_type()), expr_type(vector_type()));
add("columns_dot_self", expr_type(row_vector_type()), expr_type(row_vector_type()));
add("columns_dot_self", expr_type(row_vector_type()), expr_type(matrix_type()));
add_unary_vectorized("cos");
add_unary_vectorized("cosh");
add("cov_exp_quad", expr_type(matrix_type()), expr_type(double_type(), 1U), expr_type(double_type()), expr_type(double_type()));
add("cov_exp_quad", expr_type(matrix_type()), expr_type(vector_type(), 1U), expr_type(double_type()), expr_type(double_type()));
add("cov_exp_quad", expr_type(matrix_type()), expr_type(row_vector_type(), 1U), expr_type(double_type()), expr_type(double_type()));
add("cov_exp_quad", expr_type(matrix_type()), expr_type(double_type(), 1U), expr_type(double_type(), 1U), expr_type(double_type()), expr_type(double_type()));
add("cov_exp_quad", expr_type(matrix_type()), expr_type(vector_type(), 1U), expr_type(vector_type(), 1U), expr_type(double_type()), expr_type(double_type()));
add("cov_exp_quad", expr_type(matrix_type()), expr_type(row_vector_type(), 1U), expr_type(row_vector_type(), 1U), expr_type(double_type()), expr_type(double_type()));
add("crossprod", expr_type(matrix_type()), expr_type(matrix_type()));
add("csr_matrix_times_vector",expr_type(vector_type()), expr_type(int_type()), expr_type(int_type()),
          expr_type(vector_type()), expr_type(int_type(), 1U), expr_type(int_type(), 1U), expr_type(vector_type()));
add("csr_to_dense_matrix", expr_type(matrix_type()),expr_type(int_type()), expr_type(int_type()),
          expr_type(vector_type()), expr_type(int_type(), 1U), expr_type(int_type(), 1U));
add("csr_extract_w", expr_type(vector_type()), expr_type(matrix_type()));
add("csr_extract_v", expr_type(int_type(), 1U), expr_type(matrix_type()));
add("csr_extract_u", expr_type(int_type(), 1U), expr_type(matrix_type()));
add("cumulative_sum", expr_type(double_type(), 1U),
    expr_type(double_type(), 1U));
add("cumulative_sum", expr_type(vector_type()), expr_type(vector_type()));
add("cumulative_sum", expr_type(row_vector_type()), expr_type(row_vector_type()));
add("determinant", expr_type(double_type()), expr_type(matrix_type()));
add("diag_matrix", expr_type(matrix_type()), expr_type(vector_type()));
add("diag_post_multiply", expr_type(matrix_type()), expr_type(matrix_type()), expr_type(vector_type()));
add("diag_post_multiply", expr_type(matrix_type()), expr_type(matrix_type()), expr_type(row_vector_type()));
add("diag_pre_multiply", expr_type(matrix_type()), expr_type(vector_type()), expr_type(matrix_type()));
add("diag_pre_multiply", expr_type(matrix_type()), expr_type(row_vector_type()), expr_type(matrix_type()));
add("diagonal", expr_type(vector_type()), expr_type(matrix_type()));
add_unary_vectorized("digamma");
for (size_t i = 0; i < 8; ++i) {
  add("dims", expr_type(int_type(), 1), expr_type(int_type(), i));
  add("dims", expr_type(int_type(), 1), expr_type(double_type(), i));
  add("dims", expr_type(int_type(), 1), expr_type(vector_type(), i));
  add("dims", expr_type(int_type(), 1), expr_type(row_vector_type(), i));
  add("dims", expr_type(int_type(), 1), expr_type(matrix_type(), i));
}
add("dirichlet_log", expr_type(double_type()), expr_type(vector_type()), expr_type(vector_type()));
add("dirichlet_lpdf", expr_type(double_type()), expr_type(vector_type()), expr_type(vector_type()));
add("dirichlet_rng", expr_type(vector_type()), expr_type(vector_type()));
add("distance", expr_type(double_type()), expr_type(vector_type()), expr_type(vector_type()));
add("distance", expr_type(double_type()), expr_type(row_vector_type()), expr_type(row_vector_type()));
add("distance", expr_type(double_type()), expr_type(vector_type()), expr_type(row_vector_type()));
add("distance", expr_type(double_type()), expr_type(row_vector_type()), expr_type(vector_type()));
add("divide", expr_type(int_type()), expr_type(int_type()), expr_type(int_type()));
add("divide", expr_type(double_type()), expr_type(double_type()), expr_type(double_type()));
add("divide", expr_type(vector_type()), expr_type(vector_type()), expr_type(double_type()));
add("divide", expr_type(row_vector_type()), expr_type(row_vector_type()), expr_type(double_type()));
add("divide", expr_type(matrix_type()), expr_type(matrix_type()), expr_type(double_type()));
add("dot_product", expr_type(double_type()), expr_type(vector_type()), expr_type(vector_type()));
add("dot_product", expr_type(double_type()), expr_type(row_vector_type()), expr_type(row_vector_type()));
add("dot_product", expr_type(double_type()), expr_type(vector_type()), expr_type(row_vector_type()));
add("dot_product", expr_type(double_type()), expr_type(row_vector_type()), expr_type(vector_type()));
add("dot_product", expr_type(double_type()), expr_type(double_type(), 1U),
    expr_type(double_type(), 1U));
add("dot_self", expr_type(double_type()), expr_type(vector_type()));
add("dot_self", expr_type(double_type()), expr_type(row_vector_type()));
for (size_t i = 0; i < vector_types.size(); ++i) {
  for (size_t j = 0; j < vector_types.size(); ++j) {
    for (size_t k = 0; k < vector_types.size(); ++k) {
      add("double_exponential_ccdf_log", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("double_exponential_cdf", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("double_exponential_cdf_log", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("double_exponential_log", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("double_exponential_lccdf", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("double_exponential_lcdf", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("double_exponential_lpdf", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
    }
  }
}
add_binary("double_exponential_rng");
add_nullary("e");
add("eigenvalues_sym", expr_type(vector_type()), expr_type(matrix_type()));
add("eigenvectors_sym", expr_type(matrix_type()), expr_type(matrix_type()));
add("qr_Q", expr_type(matrix_type()), expr_type(matrix_type()));
add("qr_R", expr_type(matrix_type()), expr_type(matrix_type()));
add("elt_divide", expr_type(vector_type()), expr_type(vector_type()), expr_type(vector_type()));
add("elt_divide", expr_type(row_vector_type()), expr_type(row_vector_type()), expr_type(row_vector_type()));
add("elt_divide", expr_type(matrix_type()), expr_type(matrix_type()), expr_type(matrix_type()));
add("elt_divide", expr_type(vector_type()), expr_type(vector_type()), expr_type(double_type()));
add("elt_divide", expr_type(row_vector_type()), expr_type(row_vector_type()), expr_type(double_type()));
add("elt_divide", expr_type(matrix_type()), expr_type(matrix_type()), expr_type(double_type()));
add("elt_divide", expr_type(vector_type()), expr_type(double_type()), expr_type(vector_type()));
add("elt_divide", expr_type(row_vector_type()), expr_type(double_type()), expr_type(row_vector_type()));
add("elt_divide", expr_type(matrix_type()), expr_type(double_type()), expr_type(matrix_type()));
add("elt_multiply", expr_type(vector_type()), expr_type(vector_type()), expr_type(vector_type()));
add("elt_multiply", expr_type(row_vector_type()), expr_type(row_vector_type()), expr_type(row_vector_type()));
add("elt_multiply", expr_type(matrix_type()), expr_type(matrix_type()), expr_type(matrix_type()));
add_unary_vectorized("erf");
add_unary_vectorized("erfc");
add_unary_vectorized("exp");
add_unary_vectorized("exp2");
for (size_t i = 0; i < vector_types.size(); ++i) {
  for (size_t j = 0; j < vector_types.size(); ++j) {
    for (size_t k = 0; k < vector_types.size(); ++k) {
      for (size_t l = 0; l < vector_types.size(); ++l) {
        add("exp_mod_normal_ccdf_log", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k], vector_types[l]);
        add("exp_mod_normal_cdf", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k], vector_types[l]);
        add("exp_mod_normal_cdf_log", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k], vector_types[l]);
        add("exp_mod_normal_log", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k], vector_types[l]);
        add("exp_mod_normal_lccdf", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k], vector_types[l]);
        add("exp_mod_normal_lcdf", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k], vector_types[l]);
        add("exp_mod_normal_lpdf", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k], vector_types[l]);
      }
    }
  }
}
add_ternary("exp_mod_normal_rng");
add_unary_vectorized("expm1");
for (size_t i = 0; i < vector_types.size(); ++i) {
  for (size_t j = 0; j < vector_types.size(); ++j) {
      add("exponential_ccdf_log", expr_type(double_type()), vector_types[i], vector_types[j]);
      add("exponential_cdf", expr_type(double_type()), vector_types[i], vector_types[j]);
      add("exponential_cdf_log", expr_type(double_type()), vector_types[i], vector_types[j]);
      add("exponential_log", expr_type(double_type()), vector_types[i], vector_types[j]);
      add("exponential_lccdf", expr_type(double_type()), vector_types[i], vector_types[j]);
      add("exponential_lcdf", expr_type(double_type()), vector_types[i], vector_types[j]);
      add("exponential_lpdf", expr_type(double_type()), vector_types[i], vector_types[j]);
  }
}
add_unary("exponential_rng");
add_unary_vectorized("fabs");
add_binary("falling_factorial");
add_binary("fdim");
add_unary_vectorized("floor");
add_ternary("fma");
add_binary("fmax");
add_binary("fmin");
add_binary("fmod");
for (size_t i = 0; i < vector_types.size(); ++i) {
  for (size_t j = 0; j < vector_types.size(); ++j) {
    for (size_t k = 0; k < vector_types.size(); ++k) {
        add("frechet_ccdf_log", expr_type(double_type()), vector_types[i], 
	    vector_types[j], vector_types[k]);
        add("frechet_cdf", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k]);
        add("frechet_cdf_log", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k]);
        add("frechet_log", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k]);
        add("frechet_lccdf", expr_type(double_type()), vector_types[i], 
	    vector_types[j], vector_types[k]);
        add("frechet_lcdf", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k]);
        add("frechet_lpdf", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k]);
    }
  }
}
add_binary("frechet_rng");
for (size_t i = 0; i < vector_types.size(); ++i) {
  for (size_t j = 0; j < vector_types.size(); ++j) {
    for (size_t k = 0; k < vector_types.size(); ++k) {
      add("gamma_ccdf_log", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("gamma_cdf", expr_type(double_type()), vector_types[i], vector_types[j],
	  vector_types[k]);
      add("gamma_cdf_log", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("gamma_log", expr_type(double_type()), vector_types[i], vector_types[j],
	  vector_types[k]);
      add("gamma_lccdf", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("gamma_lcdf", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("gamma_lpdf", expr_type(double_type()), vector_types[i], vector_types[j],
	  vector_types[k]);
    }
  }
}
add_binary("gamma_p");
add_binary("gamma_q");
add_binary("gamma_rng");
add("gaussian_dlm_obs_log", expr_type(double_type()), expr_type(matrix_type()), expr_type(matrix_type()), expr_type(matrix_type()),
    expr_type(matrix_type()), expr_type(matrix_type()), expr_type(vector_type()), expr_type(matrix_type()));
add("gaussian_dlm_obs_log", expr_type(double_type()), expr_type(matrix_type()), expr_type(matrix_type()), expr_type(matrix_type()),
    expr_type(vector_type()), expr_type(matrix_type()), expr_type(vector_type()), expr_type(matrix_type()));
add("gaussian_dlm_obs_lpdf", expr_type(double_type()), expr_type(matrix_type()), expr_type(matrix_type()), expr_type(matrix_type()),
    expr_type(matrix_type()), expr_type(matrix_type()), expr_type(vector_type()), expr_type(matrix_type()));
add("gaussian_dlm_obs_lpdf", expr_type(double_type()), expr_type(matrix_type()), expr_type(matrix_type()), expr_type(matrix_type()),
    expr_type(vector_type()), expr_type(matrix_type()), expr_type(vector_type()), expr_type(matrix_type()));
add_nullary("get_lp");  // special handling in term_grammar_def
for (size_t i = 0; i < vector_types.size(); ++i) {
  for (size_t j = 0; j < vector_types.size(); ++j) {
    for (size_t k = 0; k < vector_types.size(); ++k) {
      add("gumbel_ccdf_log", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("gumbel_cdf", expr_type(double_type()), vector_types[i], vector_types[j],
	  vector_types[k]);
      add("gumbel_cdf_log", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("gumbel_log", expr_type(double_type()), vector_types[i], vector_types[j],
	  vector_types[k]);
      add("gumbel_lccdf", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("gumbel_lcdf", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("gumbel_lpdf", expr_type(double_type()), vector_types[i], vector_types[j],
	  vector_types[k]);
    }
  }
}
add_binary("gumbel_rng");
add("head", expr_type(row_vector_type()), expr_type(row_vector_type()), expr_type(int_type()));
add("head", expr_type(vector_type()), expr_type(vector_type()), expr_type(int_type()));
for (size_t i = 0; i < base_types.size(); ++i) {
  add("head", expr_type(base_types[i], 1U),
      expr_type(base_types[i], 1U), expr_type(int_type()));
  add("head", expr_type(base_types[i], 2U),
      expr_type(base_types[i], 2U), expr_type(int_type()));
  add("head", expr_type(base_types[i], 3U),
      expr_type(base_types[i], 3U), expr_type(int_type()));
}
add("hypergeometric_log", expr_type(double_type()), expr_type(int_type()), expr_type(int_type()), expr_type(int_type()), expr_type(int_type()));
add("hypergeometric_lpmf", expr_type(double_type()), expr_type(int_type()), expr_type(int_type()), expr_type(int_type()), expr_type(int_type()));
add("hypergeometric_rng", expr_type(int_type()), expr_type(int_type()), expr_type(int_type()), expr_type(int_type()));
add_binary("hypot");
add("if_else", expr_type(double_type()), expr_type(int_type()), expr_type(double_type()), expr_type(double_type()));
add("inc_beta", expr_type(double_type()), expr_type(double_type()), expr_type(double_type()), expr_type(double_type()));
add("int_step", expr_type(int_type()), expr_type(double_type()));
add("int_step", expr_type(int_type()), expr_type(int_type()));
add_unary_vectorized("inv");
for (size_t i = 0; i < vector_types.size(); ++i) {
  for (size_t j = 0; j < vector_types.size(); ++j) {
    add("inv_chi_square_ccdf_log", expr_type(double_type()), vector_types[i], vector_types[j]);
    add("inv_chi_square_cdf", expr_type(double_type()), vector_types[i], vector_types[j]);
    add("inv_chi_square_cdf_log", expr_type(double_type()), vector_types[i], vector_types[j]);
    add("inv_chi_square_log", expr_type(double_type()), vector_types[i], vector_types[j]);
    add("inv_chi_square_lccdf", expr_type(double_type()), vector_types[i], vector_types[j]);
    add("inv_chi_square_lcdf", expr_type(double_type()), vector_types[i], vector_types[j]);
    add("inv_chi_square_lpdf", expr_type(double_type()), vector_types[i], vector_types[j]);
  }
}
add_unary("inv_chi_square_rng");
add_unary_vectorized("inv_cloglog");
for (size_t i = 0; i < vector_types.size(); ++i) {
  for (size_t j = 0; j < vector_types.size(); ++j) {
    for (size_t k = 0; k < vector_types.size(); ++k) {
      add("inv_gamma_ccdf_log", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("inv_gamma_cdf", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("inv_gamma_cdf_log", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("inv_gamma_log", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("inv_gamma_lccdf", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("inv_gamma_lcdf", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("inv_gamma_lpdf", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
    }
  }
}
add_binary("inv_gamma_rng");
add_unary_vectorized("inv_logit");
add_unary_vectorized("inv_Phi");
add_unary_vectorized("inv_sqrt");
add_unary_vectorized("inv_square");
add("inv_wishart_log", expr_type(double_type()), expr_type(matrix_type()), expr_type(double_type()), expr_type(matrix_type()));
add("inv_wishart_lpdf", expr_type(double_type()), expr_type(matrix_type()), expr_type(double_type()), expr_type(matrix_type()));
add("inv_wishart_rng", expr_type(matrix_type()), expr_type(double_type()), expr_type(matrix_type()));
add("inverse", expr_type(matrix_type()), expr_type(matrix_type()));
add("inverse_spd", expr_type(matrix_type()), expr_type(matrix_type()));
add("is_inf", expr_type(int_type()), expr_type(double_type()));
add("is_nan", expr_type(int_type()), expr_type(double_type()));
add_binary("lbeta");
add_binary("lchoose");
add_unary_vectorized("lgamma");
add("lkj_corr_cholesky_log", expr_type(double_type()), expr_type(matrix_type()), expr_type(double_type()));
add("lkj_corr_cholesky_lpdf", expr_type(double_type()), expr_type(matrix_type()), expr_type(double_type()));
add("lkj_corr_cholesky_rng", expr_type(matrix_type()), expr_type(int_type()), expr_type(double_type()));
add("lkj_corr_log", expr_type(double_type()), expr_type(matrix_type()), expr_type(double_type()));
add("lkj_corr_lpdf", expr_type(double_type()), expr_type(matrix_type()), expr_type(double_type()));
add("lkj_corr_rng", expr_type(matrix_type()), expr_type(int_type()), expr_type(double_type()));
add("lkj_cov_log", expr_type(double_type()), expr_type(matrix_type()), expr_type(vector_type()), expr_type(vector_type()), expr_type(double_type()));
add("lmgamma", expr_type(double_type()), expr_type(int_type()), expr_type(double_type()));
add_binary("lmultiply");
add_unary_vectorized("log");
add_nullary("log10");
add_unary_vectorized("log10");
add_unary_vectorized("log1m");
add_unary_vectorized("log1m_exp");
add_unary_vectorized("log1m_inv_logit");
add_unary_vectorized("log1p");
add_unary_vectorized("log1p_exp");
add_nullary("log2");
add_unary_vectorized("log2");
add("log_determinant", expr_type(double_type()), expr_type(matrix_type()));
add_binary("log_diff_exp");
add_binary("log_falling_factorial");
add_ternary("log_mix");
add_binary("log_rising_factorial");
add_unary_vectorized("log_inv_logit");
add("log_softmax", expr_type(vector_type()), expr_type(vector_type()));
add("log_sum_exp", expr_type(double_type()), expr_type(double_type(), 1U));
add("log_sum_exp", expr_type(double_type()), expr_type(vector_type()));
add("log_sum_exp", expr_type(double_type()), expr_type(row_vector_type()));
add("log_sum_exp", expr_type(double_type()), expr_type(matrix_type()));
add_binary("log_sum_exp");
for (size_t i = 0; i < primitive_types.size(); ++i) {
  add("logical_negation", expr_type(int_type()), primitive_types[i]);
  for (size_t j = 0; j < primitive_types.size(); ++j) {
    add("logical_or", expr_type(int_type()), primitive_types[i],
	primitive_types[j]);
    add("logical_and", expr_type(int_type()), primitive_types[i],
	primitive_types[j]);
    add("logical_eq", expr_type(int_type()), primitive_types[i],
	primitive_types[j]);
    add("logical_neq", expr_type(int_type()), primitive_types[i],
	primitive_types[j]);
    add("logical_lt", expr_type(int_type()), primitive_types[i],
	primitive_types[j]);
    add("logical_lte", expr_type(int_type()), primitive_types[i],
	primitive_types[j]);
    add("logical_gt", expr_type(int_type()), primitive_types[i],
	primitive_types[j]);
    add("logical_gte", expr_type(int_type()), primitive_types[i],
	primitive_types[j]);
  }
}
for (size_t i = 0; i < vector_types.size(); ++i) {
  for (size_t j = 0; j < vector_types.size(); ++j) {
    for (size_t k = 0; k < vector_types.size(); ++k) {
      add("logistic_ccdf_log", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("logistic_cdf", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("logistic_cdf_log", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("logistic_log", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("logistic_lccdf", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("logistic_lcdf", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("logistic_lpdf", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
    }
  }
}
add_binary("logistic_rng");
add_unary_vectorized("logit");
for (size_t i = 0; i < vector_types.size(); ++i) {
  for (size_t j = 0; j < vector_types.size(); ++j) {
    for (size_t k = 0; k < vector_types.size(); ++k) {
      add("lognormal_ccdf_log", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("lognormal_cdf", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("lognormal_cdf_log", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("lognormal_log", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("lognormal_lccdf", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("lognormal_lcdf", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
      add("lognormal_lpdf", expr_type(double_type()), vector_types[i],
	  vector_types[j], vector_types[k]);
    }
  }
}
add_binary("lognormal_rng");
add_nullary("machine_precision");
add("matrix_exp", expr_type(matrix_type()), expr_type(matrix_type()));
add("max", expr_type(int_type()), expr_type(int_type(), 1));
add("max", expr_type(double_type()), expr_type(double_type(), 1));
add("max", expr_type(double_type()), expr_type(vector_type()));
add("max", expr_type(double_type()), expr_type(row_vector_type()));
add("max", expr_type(double_type()), expr_type(matrix_type()));
add("max", expr_type(int_type()), expr_type(int_type()), expr_type(int_type()));
add("mdivide_left", expr_type(vector_type()), expr_type(matrix_type()), expr_type(vector_type()));
add("mdivide_left", expr_type(matrix_type()), expr_type(matrix_type()), expr_type(matrix_type()));
add("mdivide_left_spd", expr_type(vector_type()), expr_type(matrix_type()), expr_type(vector_type()));
add("mdivide_left_spd", expr_type(matrix_type()), expr_type(matrix_type()), expr_type(matrix_type()));
add("mdivide_left_tri_low", expr_type(matrix_type()), expr_type(matrix_type()), expr_type(matrix_type()));
add("mdivide_left_tri_low", expr_type(vector_type()), expr_type(matrix_type()), expr_type(vector_type()));
add("mdivide_right", expr_type(row_vector_type()), expr_type(row_vector_type()), expr_type(matrix_type()));
add("mdivide_right_spd", expr_type(matrix_type()), expr_type(matrix_type()), expr_type(matrix_type()));
add("mdivide_right_spd", expr_type(row_vector_type()), expr_type(row_vector_type()), expr_type(matrix_type()));
add("mdivide_right", expr_type(matrix_type()), expr_type(matrix_type()), expr_type(matrix_type()));
add("mdivide_right_tri_low", expr_type(row_vector_type()), expr_type(row_vector_type()), expr_type(matrix_type()));
add("mdivide_right_tri_low", expr_type(matrix_type()), expr_type(matrix_type()), expr_type(matrix_type()));
add("mean", expr_type(double_type()), expr_type(double_type(), 1));
add("mean", expr_type(double_type()), expr_type(vector_type()));
add("mean", expr_type(double_type()), expr_type(row_vector_type()));
add("mean", expr_type(double_type()), expr_type(matrix_type()));
add("min", expr_type(int_type()), expr_type(int_type(), 1));
add("min", expr_type(double_type()), expr_type(double_type(), 1));
add("min", expr_type(double_type()), expr_type(vector_type()));
add("min", expr_type(double_type()), expr_type(row_vector_type()));
add("min", expr_type(double_type()), expr_type(matrix_type()));
add("min", expr_type(int_type()), expr_type(int_type()), expr_type(int_type()));
add("minus", expr_type(double_type()), expr_type(double_type()));
add("minus", expr_type(vector_type()), expr_type(vector_type()));
add("minus", expr_type(row_vector_type()), expr_type(row_vector_type()));
add("minus", expr_type(matrix_type()), expr_type(matrix_type()));
add("modified_bessel_first_kind", expr_type(double_type()), expr_type(int_type()), expr_type(double_type()));
add("modified_bessel_second_kind", expr_type(double_type()), expr_type(int_type()), expr_type(double_type()));
add("modulus", expr_type(int_type()), expr_type(int_type()), expr_type(int_type()));
add("multi_gp_log", expr_type(double_type()), expr_type(matrix_type()), expr_type(matrix_type()), expr_type(vector_type()));
add("multi_gp_lpdf", expr_type(double_type()), expr_type(matrix_type()), expr_type(matrix_type()), expr_type(vector_type()));
add("multi_gp_cholesky_log", expr_type(double_type()), expr_type(matrix_type()), expr_type(matrix_type()), expr_type(vector_type()));
add("multi_gp_cholesky_lpdf", expr_type(double_type()), expr_type(matrix_type()), expr_type(matrix_type()), expr_type(vector_type()));
{
  std::vector<base_expr_type> eigen_vector_types;
  eigen_vector_types.push_back(vector_type());
  eigen_vector_types.push_back(row_vector_type());
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      for (size_t k = 0; k < 2; ++k) {
        for (size_t l = 0; l < 2; ++l) {
          add("multi_normal_cholesky_log", expr_type(double_type()),
              expr_type(eigen_vector_types[k], i),
              expr_type(eigen_vector_types[l], j), expr_type(matrix_type()));
          add("multi_normal_cholesky_lpdf", expr_type(double_type()),
              expr_type(eigen_vector_types[k], i),
              expr_type(eigen_vector_types[l], j), expr_type(matrix_type()));

          add("multi_normal_log", expr_type(double_type()),
              expr_type(eigen_vector_types[k], i),
              expr_type(eigen_vector_types[l], j), expr_type(matrix_type()));
          add("multi_normal_lpdf", expr_type(double_type()),
              expr_type(eigen_vector_types[k], i),
              expr_type(eigen_vector_types[l], j), expr_type(matrix_type()));

          add("multi_normal_prec_log", expr_type(double_type()),
              expr_type(eigen_vector_types[k], i),
              expr_type(eigen_vector_types[l], j), expr_type(matrix_type()));
          add("multi_normal_prec_lpdf", expr_type(double_type()),
              expr_type(eigen_vector_types[k], i),
              expr_type(eigen_vector_types[l], j), expr_type(matrix_type()));

          add("multi_student_t_log", expr_type(double_type()),
              expr_type(eigen_vector_types[k], i), expr_type(double_type()),
              expr_type(eigen_vector_types[l], j), expr_type(matrix_type()));
          add("multi_student_t_lpdf", expr_type(double_type()),
              expr_type(eigen_vector_types[k], i), expr_type(double_type()),
              expr_type(eigen_vector_types[l], j), expr_type(matrix_type()));
        }
      }
    }
  }
}
add("multi_normal_rng", expr_type(vector_type()), expr_type(vector_type()), expr_type(matrix_type()));
add("multi_normal_cholesky_rng", expr_type(vector_type()), expr_type(vector_type()), expr_type(matrix_type()));
add("multi_student_t_rng", expr_type(vector_type()), expr_type(double_type()), expr_type(vector_type()), expr_type(matrix_type()));
add("multinomial_log", expr_type(double_type()), expr_type(int_type(), 1U), expr_type(vector_type()));
add("multinomial_lpmf", expr_type(double_type()), expr_type(int_type(), 1U), expr_type(vector_type()));
add("multinomial_rng", expr_type(int_type(), 1U), expr_type(vector_type()), expr_type(int_type()));
add("multiply", expr_type(double_type()), expr_type(double_type()), expr_type(double_type()));
add("multiply", expr_type(vector_type()), expr_type(vector_type()), expr_type(double_type()));
add("multiply", expr_type(row_vector_type()), expr_type(row_vector_type()), expr_type(double_type()));
add("multiply", expr_type(matrix_type()), expr_type(matrix_type()), expr_type(double_type()));
add("multiply", expr_type(double_type()), expr_type(row_vector_type()), expr_type(vector_type()));
add("multiply", expr_type(matrix_type()), expr_type(vector_type()), expr_type(row_vector_type()));
add("multiply", expr_type(vector_type()), expr_type(matrix_type()), expr_type(vector_type()));
add("multiply", expr_type(row_vector_type()), expr_type(row_vector_type()), expr_type(matrix_type()));
add("multiply", expr_type(matrix_type()), expr_type(matrix_type()), expr_type(matrix_type()));
add("multiply", expr_type(vector_type()), expr_type(double_type()), expr_type(vector_type()));
add("multiply", expr_type(row_vector_type()), expr_type(double_type()), expr_type(row_vector_type()));
add("multiply", expr_type(matrix_type()), expr_type(double_type()), expr_type(matrix_type()));
add_binary("multiply_log");
add("multiply_lower_tri_self_transpose", expr_type(matrix_type()), expr_type(matrix_type()));
for (size_t i = 0; i < int_vector_types.size(); ++i) {
  for (size_t j = 0; j < vector_types.size(); ++j) {
    for (size_t k = 0; k < vector_types.size(); ++k) {
      add("neg_binomial_ccdf_log", expr_type(double_type()), 
          int_vector_types[i], vector_types[j], vector_types[k]);
      add("neg_binomial_cdf", expr_type(double_type()), 
          int_vector_types[i], vector_types[j], vector_types[k]);
      add("neg_binomial_cdf_log", expr_type(double_type()), 
          int_vector_types[i], vector_types[j], vector_types[k]);
      add("neg_binomial_log", expr_type(double_type()), 
          int_vector_types[i], vector_types[j], vector_types[k]);
      add("neg_binomial_lccdf", expr_type(double_type()), 
          int_vector_types[i], vector_types[j], vector_types[k]);
      add("neg_binomial_lcdf", expr_type(double_type()), 
          int_vector_types[i], vector_types[j], vector_types[k]);
      add("neg_binomial_lpmf", expr_type(double_type()), 
          int_vector_types[i], vector_types[j], vector_types[k]);

      add("neg_binomial_2_ccdf_log", expr_type(double_type()), 
          int_vector_types[i], vector_types[j], vector_types[k]);
      add("neg_binomial_2_cdf", expr_type(double_type()), 
          int_vector_types[i], vector_types[j], vector_types[k]);
      add("neg_binomial_2_cdf_log", expr_type(double_type()), 
          int_vector_types[i], vector_types[j], vector_types[k]);
      add("neg_binomial_2_log", expr_type(double_type()), 
          int_vector_types[i], vector_types[j], vector_types[k]);
      add("neg_binomial_2_lccdf", expr_type(double_type()), 
          int_vector_types[i], vector_types[j], vector_types[k]);
      add("neg_binomial_2_lcdf", expr_type(double_type()), 
          int_vector_types[i], vector_types[j], vector_types[k]);
      add("neg_binomial_2_lpmf", expr_type(double_type()), 
          int_vector_types[i], vector_types[j], vector_types[k]);

      add("neg_binomial_2_log_log", expr_type(double_type()), 
          int_vector_types[i], vector_types[j], vector_types[k]);
      add("neg_binomial_2_log_lpmf", expr_type(double_type()), 
          int_vector_types[i], vector_types[j], vector_types[k]);
    }
  }
}
add("neg_binomial_rng", expr_type(int_type()), expr_type(double_type()), expr_type(double_type()));
add("neg_binomial_2_rng", expr_type(int_type()), expr_type(double_type()), expr_type(double_type()));
add("neg_binomial_2_log_rng", expr_type(int_type()), expr_type(double_type()), expr_type(double_type()));
add_nullary("negative_infinity");
for (size_t i = 0; i < vector_types.size(); ++i) {
  for (size_t j = 0; j < vector_types.size(); ++j) {
    for (size_t k = 0; k < vector_types.size(); ++k) {
      add("normal_ccdf_log", expr_type(double_type()),
          vector_types[i], vector_types[j], vector_types[k]);
      add("normal_cdf", expr_type(double_type()),
          vector_types[i], vector_types[j], vector_types[k]);
      add("normal_cdf_log", expr_type(double_type()),
          vector_types[i], vector_types[j], vector_types[k]);
      add("normal_log", expr_type(double_type()),
          vector_types[i], vector_types[j], vector_types[k]);
      add("normal_lccdf", expr_type(double_type()),
          vector_types[i], vector_types[j], vector_types[k]);
      add("normal_lcdf", expr_type(double_type()),
          vector_types[i], vector_types[j], vector_types[k]);
      add("normal_lpdf", expr_type(double_type()),
          vector_types[i], vector_types[j], vector_types[k]);
    }
  }
}
add_binary("normal_rng");
add_nullary("not_a_number");
add("num_elements", expr_type(int_type()), expr_type(matrix_type()));
add("num_elements", expr_type(int_type()), expr_type(vector_type()));
add("num_elements", expr_type(int_type()), expr_type(row_vector_type()));
for (size_t i=1; i < 10; i++) {
  add("num_elements", expr_type(int_type()), expr_type(int_type(), i));
  add("num_elements", expr_type(int_type()), expr_type(double_type(), i));
  add("num_elements", expr_type(int_type()), expr_type(matrix_type(), i));
  add("num_elements", expr_type(int_type()), expr_type(row_vector_type(), i));
  add("num_elements", expr_type(int_type()), expr_type(vector_type(), i));
}
add("ordered_logistic_log", expr_type(double_type()), expr_type(int_type()), expr_type(double_type()), expr_type(vector_type()));
add("ordered_logistic_lpmf", expr_type(double_type()), expr_type(int_type()), expr_type(double_type()), expr_type(vector_type()));
add("ordered_logistic_rng", expr_type(int_type()), expr_type(double_type()), expr_type(vector_type()));
add_binary("owens_t");
for (size_t i = 0; i < vector_types.size(); ++i) {
  for (size_t j = 0; j < vector_types.size(); ++j) {
    for (size_t k = 0; k < vector_types.size(); ++k) {
      add("pareto_ccdf_log", expr_type(double_type()),
          vector_types[i], vector_types[j], vector_types[k]);
      add("pareto_cdf", expr_type(double_type()),
          vector_types[i], vector_types[j], vector_types[k]);
      add("pareto_cdf_log", expr_type(double_type()),
          vector_types[i], vector_types[j], vector_types[k]);
      add("pareto_log", expr_type(double_type()),
          vector_types[i], vector_types[j], vector_types[k]);
      add("pareto_lccdf", expr_type(double_type()),
          vector_types[i], vector_types[j], vector_types[k]);
      add("pareto_lcdf", expr_type(double_type()),
          vector_types[i], vector_types[j], vector_types[k]);
      add("pareto_lpdf", expr_type(double_type()),
          vector_types[i], vector_types[j], vector_types[k]);
    }
  }
}
add_binary("pareto_rng");
for (size_t i = 0; i < vector_types.size(); ++i) {
  for (size_t j = 0; j < vector_types.size(); ++j) {
    for (size_t k = 0; k < vector_types.size(); ++k) {
      for (size_t l = 0; l < vector_types.size(); ++l) {
        add("pareto_type_2_ccdf_log", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k], vector_types[l]);
        add("pareto_type_2_cdf", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k], vector_types[l]);
        add("pareto_type_2_cdf_log", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k], vector_types[l]);
        add("pareto_type_2_log", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k], vector_types[l]);
        add("pareto_type_2_lccdf", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k], vector_types[l]);
        add("pareto_type_2_lcdf", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k], vector_types[l]);
        add("pareto_type_2_lpdf", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k], vector_types[l]);
      }
    }
  }
}
add_ternary("pareto_type_2_rng");
add_unary_vectorized("Phi");
add_unary_vectorized("Phi_approx");
add_nullary("pi");
for (size_t i = 0; i < int_vector_types.size(); ++i) {
  for (size_t j = 0; j < vector_types.size(); ++j) {
    add("poisson_ccdf_log", expr_type(double_type()), int_vector_types[i],
	vector_types[j]);
    add("poisson_cdf", expr_type(double_type()), int_vector_types[i],
	vector_types[j]);
    add("poisson_cdf_log", expr_type(double_type()), int_vector_types[i],
	vector_types[j]);
    add("poisson_log", expr_type(double_type()), int_vector_types[i],
	vector_types[j]);
    add("poisson_lccdf", expr_type(double_type()), int_vector_types[i],
    	vector_types[j]);
    add("poisson_lcdf", expr_type(double_type()), int_vector_types[i],
    	vector_types[j]);
    add("poisson_lpmf", expr_type(double_type()), int_vector_types[i],
    	vector_types[j]);
  }
}
add("poisson_rng", expr_type(int_type()), expr_type(double_type()));
for (size_t i = 0; i < int_vector_types.size(); ++i) {
  for (size_t j = 0; j < vector_types.size(); ++j) {
    add("poisson_log_log", expr_type(double_type()), int_vector_types[i],
	vector_types[j]);
    add("poisson_log_lpmf", expr_type(double_type()), int_vector_types[i],
	vector_types[j]);
  }
}
add("poisson_log_rng", expr_type(int_type()), expr_type(double_type()));
add_nullary("positive_infinity");
add_binary("pow");
add("prod", expr_type(int_type()), expr_type(int_type(), 1));
add("prod", expr_type(double_type()), expr_type(double_type(), 1));
add("prod", expr_type(double_type()), expr_type(vector_type()));
add("prod", expr_type(double_type()), expr_type(row_vector_type()));
add("prod", expr_type(double_type()), expr_type(matrix_type()));
add("quad_form", expr_type(double_type()), expr_type(matrix_type()), expr_type(vector_type()));
add("quad_form", expr_type(matrix_type()), expr_type(matrix_type()), expr_type(matrix_type()));
add("quad_form_sym", expr_type(double_type()), expr_type(matrix_type()), expr_type(vector_type()));
add("quad_form_sym", expr_type(matrix_type()), expr_type(matrix_type()), expr_type(matrix_type()));
add("quad_form_diag", expr_type(matrix_type()), expr_type(matrix_type()), expr_type(vector_type()));
add("quad_form_diag", expr_type(matrix_type()), expr_type(matrix_type()), expr_type(row_vector_type()));
add("rank", expr_type(int_type()), expr_type(int_type(), 1), expr_type(int_type()));
add("rank", expr_type(int_type()), expr_type(double_type(), 1), expr_type(int_type()));
add("rank", expr_type(int_type()), expr_type(vector_type()), expr_type(int_type()));
add("rank", expr_type(int_type()), expr_type(row_vector_type()), expr_type(int_type()));
for (size_t i = 0; i < vector_types.size(); ++i) {
  for (size_t j = 0; j < vector_types.size(); ++j) {
    add("rayleigh_ccdf_log", expr_type(double_type()), vector_types[i], vector_types[j]);
    add("rayleigh_cdf", expr_type(double_type()), vector_types[i], vector_types[j]);
    add("rayleigh_cdf_log", expr_type(double_type()), vector_types[i], vector_types[j]);
    add("rayleigh_log", expr_type(double_type()), vector_types[i], vector_types[j]);
    add("rayleigh_lccdf", expr_type(double_type()), vector_types[i], vector_types[j]);
    add("rayleigh_lcdf", expr_type(double_type()), vector_types[i], vector_types[j]);
    add("rayleigh_lpdf", expr_type(double_type()), vector_types[i], vector_types[j]);
  }
}
add_unary("rayleigh_rng");
add("append_row", expr_type(matrix_type()), expr_type(matrix_type()), expr_type(matrix_type()));
add("append_row", expr_type(matrix_type()), expr_type(row_vector_type()), expr_type(matrix_type()));
add("append_row", expr_type(matrix_type()), expr_type(matrix_type()), expr_type(row_vector_type()));
add("append_row", expr_type(matrix_type()), expr_type(row_vector_type()), expr_type(row_vector_type()));
add("append_row", expr_type(vector_type()), expr_type(vector_type()), expr_type(vector_type()));
add("append_row", expr_type(vector_type()), expr_type(double_type()), expr_type(vector_type()));
add("append_row", expr_type(vector_type()), expr_type(vector_type()), expr_type(double_type()));
for (size_t i = 0; i < base_types.size(); ++i) {
  add("rep_array", expr_type(base_types[i], 1), base_types[i], expr_type(int_type()));
  add("rep_array", expr_type(base_types[i], 2), base_types[i], expr_type(int_type()), expr_type(int_type()));
  add("rep_array", expr_type(base_types[i], 3), base_types[i],
      expr_type(int_type()), expr_type(int_type()), expr_type(int_type()));
  for (size_t j = 1; j <= 3; ++j) {
    add("rep_array", expr_type(base_types[i], j + 1),
	expr_type(base_types[i], j),  expr_type(int_type()));
    add("rep_array", expr_type(base_types[i], j + 2),
	expr_type(base_types[i], j),  expr_type(int_type()), expr_type(int_type()));
    add("rep_array", expr_type(base_types[i], j + 3),
	expr_type(base_types[i], j),  expr_type(int_type()), expr_type(int_type()), expr_type(int_type()));
  }
}
add("rep_matrix", expr_type(matrix_type()), expr_type(double_type()), expr_type(int_type()), expr_type(int_type()));
add("rep_matrix", expr_type(matrix_type()), expr_type(vector_type()), expr_type(int_type()));
add("rep_matrix", expr_type(matrix_type()), expr_type(row_vector_type()), expr_type(int_type()));
add("rep_row_vector", expr_type(row_vector_type()), expr_type(double_type()), expr_type(int_type()));
add("rep_vector", expr_type(vector_type()), expr_type(double_type()), expr_type(int_type()));
add_binary("rising_factorial");
add_unary_vectorized("round");
add("row", expr_type(row_vector_type()), expr_type(matrix_type()), expr_type(int_type()));
add("rows", expr_type(int_type()), expr_type(vector_type()));
add("rows", expr_type(int_type()), expr_type(row_vector_type()));
add("rows", expr_type(int_type()), expr_type(matrix_type()));
add("rows_dot_product", expr_type(vector_type()), expr_type(vector_type()), expr_type(vector_type()));
add("rows_dot_product", expr_type(vector_type()), expr_type(row_vector_type()), expr_type(row_vector_type()));
add("rows_dot_product", expr_type(vector_type()), expr_type(matrix_type()), expr_type(matrix_type()));
add("rows_dot_self", expr_type(vector_type()), expr_type(vector_type()));
add("rows_dot_self", expr_type(vector_type()), expr_type(row_vector_type()));
add("rows_dot_self", expr_type(vector_type()), expr_type(matrix_type()));
for (size_t i = 0; i < vector_types.size(); ++i) {
  for (size_t j = 0; j < vector_types.size(); ++j) {
    for (size_t k = 0; k < vector_types.size(); ++k) {
      add("scaled_inv_chi_square_ccdf_log", expr_type(double_type()),
          vector_types[i], vector_types[j], vector_types[k]);
      add("scaled_inv_chi_square_cdf", expr_type(double_type()),
          vector_types[i], vector_types[j], vector_types[k]);
      add("scaled_inv_chi_square_cdf_log", expr_type(double_type()),
          vector_types[i], vector_types[j], vector_types[k]);
      add("scaled_inv_chi_square_log", expr_type(double_type()),
          vector_types[i], vector_types[j], vector_types[k]);
      add("scaled_inv_chi_square_lccdf", expr_type(double_type()),
          vector_types[i], vector_types[j], vector_types[k]);
      add("scaled_inv_chi_square_lcdf", expr_type(double_type()),
          vector_types[i], vector_types[j], vector_types[k]);
      add("scaled_inv_chi_square_lpdf", expr_type(double_type()),
          vector_types[i], vector_types[j], vector_types[k]);
    }
  }
}
add_binary("scaled_inv_chi_square_rng");
add("sd", expr_type(double_type()), expr_type(double_type(), 1));
add("sd", expr_type(double_type()), expr_type(vector_type()));
add("sd", expr_type(double_type()), expr_type(row_vector_type()));
add("sd", expr_type(double_type()), expr_type(matrix_type()));
add("segment", expr_type(row_vector_type()), expr_type(row_vector_type()), expr_type(int_type()), expr_type(int_type()));
add("segment", expr_type(vector_type()), expr_type(vector_type()), expr_type(int_type()), expr_type(int_type()));
for (size_t i = 0; i < base_types.size(); ++i) {
  add("segment", expr_type(base_types[i], 1U),
      expr_type(base_types[i], 1U), expr_type(int_type()), expr_type(int_type()));
  add("segment", expr_type(base_types[i], 2U),
      expr_type(base_types[i], 2U), expr_type(int_type()), expr_type(int_type()));
  add("segment", expr_type(base_types[i], 3U),
      expr_type(base_types[i], 3U), expr_type(int_type()), expr_type(int_type()));
}
add_unary_vectorized("sin");
add("singular_values", expr_type(vector_type()), expr_type(matrix_type()));
add_unary_vectorized("sinh");
// size() is polymorphic over arrays, so start i at 1
for (size_t i = 1; i < 8; ++i) {
  add("size", expr_type(int_type()), expr_type(int_type(), i));
  add("size", expr_type(int_type()), expr_type(double_type(), i));
  add("size", expr_type(int_type()), expr_type(vector_type(), i));
  add("size", expr_type(int_type()), expr_type(row_vector_type(), i));
  add("size", expr_type(int_type()), expr_type(matrix_type(), i));
}
for (size_t i = 0; i < vector_types.size(); ++i) {
  for (size_t j = 0; j < vector_types.size(); ++j) {
    for (size_t k = 0; k < vector_types.size(); ++k) {
      for (size_t l = 0; l < vector_types.size(); ++l) {
        add("skew_normal_ccdf_log", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k], vector_types[l]);
        add("skew_normal_cdf", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k], vector_types[l]);
        add("skew_normal_cdf_log", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k], vector_types[l]);
        add("skew_normal_log", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k], vector_types[l]);
        add("skew_normal_lccdf", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k], vector_types[l]);
        add("skew_normal_lcdf", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k], vector_types[l]);
        add("skew_normal_lpdf", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k], vector_types[l]);
      }
    }
  }
}
add_ternary("skew_normal_rng");
add("softmax", expr_type(vector_type()), expr_type(vector_type()));
add("sort_asc", expr_type(int_type(), 1), expr_type(int_type(), 1));
add("sort_asc", expr_type(double_type(), 1), expr_type(double_type(), 1));
add("sort_asc", expr_type(vector_type()), expr_type(vector_type()));
add("sort_asc", expr_type(row_vector_type()), expr_type(row_vector_type()));
add("sort_desc", expr_type(int_type(), 1), expr_type(int_type(), 1));
add("sort_desc", expr_type(double_type(), 1), expr_type(double_type(), 1));
add("sort_desc", expr_type(vector_type()), expr_type(vector_type()));
add("sort_desc", expr_type(row_vector_type()), expr_type(row_vector_type()));
add("sort_indices_asc", expr_type(int_type(), 1), expr_type(int_type(), 1));
add("sort_indices_asc", expr_type(int_type(), 1), expr_type(double_type(), 1));
add("sort_indices_asc", expr_type(int_type(), 1), expr_type(vector_type()));
add("sort_indices_asc", expr_type(int_type(), 1), expr_type(row_vector_type()));
add("sort_indices_desc", expr_type(int_type(), 1), expr_type(int_type(), 1));
add("sort_indices_desc", expr_type(int_type(), 1),
    expr_type(double_type(), 1));
add("sort_indices_desc", expr_type(int_type(), 1), expr_type(vector_type()));
add("sort_indices_desc", expr_type(int_type(), 1), expr_type(row_vector_type()));
add("squared_distance", expr_type(double_type()), expr_type(double_type()), expr_type(double_type()));
add("squared_distance", expr_type(double_type()), expr_type(vector_type()), expr_type(vector_type()));
add("squared_distance", expr_type(double_type()), expr_type(row_vector_type()), expr_type(row_vector_type()));
add("squared_distance", expr_type(double_type()), expr_type(vector_type()), expr_type(row_vector_type()));
add("squared_distance", expr_type(double_type()), expr_type(row_vector_type()), expr_type(vector_type()));
add_unary_vectorized("sqrt");
add_nullary("sqrt2");
add_unary_vectorized("square");
add_unary("step");
for (size_t i = 0; i < vector_types.size(); ++i) {
  for (size_t j = 0; j < vector_types.size(); ++j) {
    for (size_t k = 0; k < vector_types.size(); ++k) {
      for (size_t l = 0; l < vector_types.size(); ++l) {
        add("student_t_ccdf_log", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k], vector_types[l]);
        add("student_t_cdf", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k], vector_types[l]);
        add("student_t_cdf_log", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k], vector_types[l]);
        add("student_t_log", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k], vector_types[l]);
        add("student_t_lccdf", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k], vector_types[l]);
        add("student_t_lcdf", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k], vector_types[l]);
        add("student_t_lpdf", expr_type(double_type()), vector_types[i],
	    vector_types[j], vector_types[k], vector_types[l]);
      }
    }
  }
}
add_ternary("student_t_rng");
add("sub_col", expr_type(vector_type()), expr_type(matrix_type()), expr_type(int_type()), expr_type(int_type()), expr_type(int_type()));
add("sub_row", expr_type(row_vector_type()), expr_type(matrix_type()), expr_type(int_type()), expr_type(int_type()), expr_type(int_type()));
add("subtract", expr_type(vector_type()), expr_type(vector_type()), expr_type(vector_type()));
add("subtract", expr_type(row_vector_type()), expr_type(row_vector_type()), expr_type(row_vector_type()));
add("subtract", expr_type(matrix_type()), expr_type(matrix_type()), expr_type(matrix_type()));
add("subtract", expr_type(vector_type()), expr_type(vector_type()), expr_type(double_type()));
add("subtract", expr_type(row_vector_type()), expr_type(row_vector_type()), expr_type(double_type()));
add("subtract", expr_type(matrix_type()), expr_type(matrix_type()), expr_type(double_type()));
add("subtract", expr_type(vector_type()), expr_type(double_type()), expr_type(vector_type()));
add("subtract", expr_type(row_vector_type()), expr_type(double_type()), expr_type(row_vector_type()));
add("subtract", expr_type(matrix_type()), expr_type(double_type()), expr_type(matrix_type()));
add("sum", expr_type(int_type()), expr_type(int_type(), 1));
add("sum", expr_type(double_type()), expr_type(double_type(), 1));
add("sum", expr_type(double_type()), expr_type(vector_type()));
add("sum", expr_type(double_type()), expr_type(row_vector_type()));
add("sum", expr_type(double_type()), expr_type(matrix_type()));
add("tail", expr_type(row_vector_type()), expr_type(row_vector_type()), expr_type(int_type()));
add("tail", expr_type(vector_type()), expr_type(vector_type()), expr_type(int_type()));
for (size_t i = 0; i < base_types.size(); ++i) {
  add("tail", expr_type(base_types[i], 1U),
      expr_type(base_types[i], 1U), expr_type(int_type()));
  add("tail", expr_type(base_types[i], 2U),
      expr_type(base_types[i], 2U), expr_type(int_type()));
  add("tail", expr_type(base_types[i], 3U),
      expr_type(base_types[i], 3U), expr_type(int_type()));
}
add_unary_vectorized("tan");
add_unary_vectorized("tanh");
add_nullary("target");  // converted to "get_lp" in term_grammar semantics
add("tcrossprod", expr_type(matrix_type()), expr_type(matrix_type()));
add_unary_vectorized("tgamma");
add("to_array_1d", expr_type(double_type(), 1), expr_type(matrix_type()));
add("to_array_1d", expr_type(double_type(), 1), expr_type(vector_type()));
add("to_array_1d", expr_type(double_type(), 1), expr_type(row_vector_type()));
for (size_t i=1; i < 10; i++) {
  add("to_array_1d", expr_type(double_type(), 1),
      expr_type(double_type(), i));
  add("to_array_1d", expr_type(int_type(), 1), expr_type(int_type(), i));
}
add("to_array_2d", expr_type(double_type(), 2), expr_type(matrix_type()));
add("to_matrix", expr_type(matrix_type()), expr_type(matrix_type()));
add("to_matrix", expr_type(matrix_type()), expr_type(matrix_type()), expr_type(int_type()), expr_type(int_type()));
add("to_matrix", expr_type(matrix_type()), expr_type(matrix_type()), expr_type(int_type()), expr_type(int_type()), expr_type(int_type()));
add("to_matrix", expr_type(matrix_type()), expr_type(vector_type()));
add("to_matrix", expr_type(matrix_type()), expr_type(vector_type()), expr_type(int_type()), expr_type(int_type()));
add("to_matrix", expr_type(matrix_type()), expr_type(vector_type()), expr_type(int_type()), expr_type(int_type()), expr_type(int_type()));
add("to_matrix", expr_type(matrix_type()), expr_type(row_vector_type()));
add("to_matrix", expr_type(matrix_type()), expr_type(row_vector_type()), expr_type(int_type()), expr_type(int_type()));
add("to_matrix", expr_type(matrix_type()), expr_type(row_vector_type()), expr_type(int_type()), expr_type(int_type()), expr_type(int_type()));
add("to_matrix", expr_type(matrix_type()), expr_type(double_type(), 1), expr_type(int_type()), expr_type(int_type()));
add("to_matrix", expr_type(matrix_type()), expr_type(double_type(), 1), expr_type(int_type()), expr_type(int_type()), expr_type(int_type()));
add("to_matrix", expr_type(matrix_type()), expr_type(int_type(), 1), expr_type(int_type()), expr_type(int_type()));
add("to_matrix", expr_type(matrix_type()), expr_type(int_type(), 1), expr_type(int_type()), expr_type(int_type()), expr_type(int_type()));
add("to_matrix", expr_type(matrix_type()), expr_type(double_type(), 2));
add("to_matrix", expr_type(matrix_type()), expr_type(int_type(), 2));
add("to_row_vector", expr_type(row_vector_type()), expr_type(matrix_type()));
add("to_row_vector", expr_type(row_vector_type()), expr_type(vector_type()));
add("to_row_vector", expr_type(row_vector_type()), expr_type(row_vector_type()));
add("to_row_vector", expr_type(row_vector_type()), expr_type(double_type(), 1));
add("to_row_vector", expr_type(row_vector_type()), expr_type(int_type(), 1));
add("to_vector", expr_type(vector_type()), expr_type(matrix_type()));
add("to_vector", expr_type(vector_type()), expr_type(vector_type()));
add("to_vector", expr_type(vector_type()), expr_type(row_vector_type()));
add("to_vector", expr_type(vector_type()), expr_type(double_type(), 1));
add("to_vector", expr_type(vector_type()), expr_type(int_type(), 1));
add("trace", expr_type(double_type()), expr_type(matrix_type()));
add("trace_gen_quad_form", expr_type(double_type()), expr_type(matrix_type()), expr_type(matrix_type()), expr_type(matrix_type()));
add("trace_quad_form", expr_type(double_type()), expr_type(matrix_type()), expr_type(vector_type()));
add("trace_quad_form", expr_type(double_type()), expr_type(matrix_type()), expr_type(matrix_type()));
add("transpose", expr_type(row_vector_type()), expr_type(vector_type()));
add("transpose", expr_type(vector_type()), expr_type(row_vector_type()));
add("transpose", expr_type(matrix_type()), expr_type(matrix_type()));
add_unary_vectorized("trunc");
add_unary_vectorized("trigamma");
for (size_t i = 0; i < vector_types.size(); ++i) {
  for (size_t j = 0; j < vector_types.size(); ++j) {
    for (size_t k = 0; k < vector_types.size(); ++k) {
        add("uniform_ccdf_log", expr_type(double_type()),
            vector_types[i], vector_types[j], vector_types[k]);
        add("uniform_cdf", expr_type(double_type()),
            vector_types[i], vector_types[j], vector_types[k]);
        add("uniform_cdf_log", expr_type(double_type()),
            vector_types[i], vector_types[j], vector_types[k]);
        add("uniform_log", expr_type(double_type()),
            vector_types[i], vector_types[j], vector_types[k]);
        add("uniform_lccdf", expr_type(double_type()),
            vector_types[i], vector_types[j], vector_types[k]);
        add("uniform_lcdf", expr_type(double_type()),
            vector_types[i], vector_types[j], vector_types[k]);
        add("uniform_lpdf", expr_type(double_type()),
            vector_types[i], vector_types[j], vector_types[k]);
    }
  }
}
add_binary("uniform_rng");
add("variance", expr_type(double_type()), expr_type(double_type(), 1));
add("variance", expr_type(double_type()), expr_type(vector_type()));
add("variance", expr_type(double_type()), expr_type(row_vector_type()));
add("variance", expr_type(double_type()), expr_type(matrix_type()));
for (size_t i = 0; i < vector_types.size(); ++i) {
  for (size_t j = 0; j < vector_types.size(); ++j) {
    for (size_t k = 0; k < vector_types.size(); ++k) {
      add("von_mises_log", expr_type(double_type()),
          vector_types[i], vector_types[j], vector_types[k]);
      add("von_mises_lpdf", expr_type(double_type()),
          vector_types[i], vector_types[j], vector_types[k]);
    }
  }
}
add_binary("von_mises_rng");
for (size_t i = 0; i < vector_types.size(); ++i) {
  for (size_t j = 0; j < vector_types.size(); ++j) {
    for (size_t k = 0; k < vector_types.size(); ++k) {
        add("weibull_ccdf_log", expr_type(double_type()),
            vector_types[i], vector_types[j], vector_types[k]);
        add("weibull_cdf", expr_type(double_type()),
            vector_types[i], vector_types[j], vector_types[k]);
        add("weibull_cdf_log", expr_type(double_type()),
            vector_types[i], vector_types[j], vector_types[k]);
        add("weibull_log", expr_type(double_type()),
            vector_types[i], vector_types[j], vector_types[k]);
        add("weibull_lccdf", expr_type(double_type()),
            vector_types[i], vector_types[j], vector_types[k]);
        add("weibull_lcdf", expr_type(double_type()),
            vector_types[i], vector_types[j], vector_types[k]);
        add("weibull_lpdf", expr_type(double_type()),
            vector_types[i], vector_types[j], vector_types[k]);
    }
  }
}
add_binary("weibull_rng");
for (size_t i = 0; i < vector_types.size(); ++i) {
  for (size_t j = 0; j < vector_types.size(); ++j) {
    for (size_t k = 0; k < vector_types.size(); ++k) {
      for (size_t l = 0; l < vector_types.size(); ++l) {
	for (size_t m = 0; m < vector_types.size(); ++m) {
          add("wiener_log", expr_type(double_type()), vector_types[i],
	      vector_types[j],vector_types[k], vector_types[l],
	      vector_types[m]);
	  add("wiener_lpdf", expr_type(double_type()), vector_types[i],
	      vector_types[j],vector_types[k], vector_types[l],
	      vector_types[m]);
	}
      }
    }
  }
}
add("wishart_log", expr_type(double_type()), expr_type(matrix_type()), expr_type(double_type()), expr_type(matrix_type()));
add("wishart_lpdf", expr_type(double_type()), expr_type(matrix_type()), expr_type(double_type()), expr_type(matrix_type()));
add("wishart_rng", expr_type(matrix_type()), expr_type(double_type()), expr_type(matrix_type()));

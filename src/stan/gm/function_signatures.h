// included from constructor for function_signatures() in
// src/stan/gm/ast.hpp

std::vector<base_expr_type> base_types;
base_types.push_back(INT_T);
base_types.push_back(DOUBLE_T);
base_types.push_back(VECTOR_T);
base_types.push_back(ROW_VECTOR_T);
base_types.push_back(MATRIX_T);

std::vector<expr_type> vector_types;
vector_types.push_back(DOUBLE_T);                  // scalar
vector_types.push_back(expr_type(DOUBLE_T,1U));    // std vector
vector_types.push_back(VECTOR_T);                  // Eigen vector
vector_types.push_back(ROW_VECTOR_T);              // Eigen row vector

std::vector<expr_type> int_vector_types;
int_vector_types.push_back(INT_T);                  // scalar
int_vector_types.push_back(expr_type(INT_T,1U));    // std vector

std::vector<expr_type> primitive_types;
primitive_types.push_back(INT_T);
primitive_types.push_back(DOUBLE_T);

add_unary("abs");
add("abs",INT_T,INT_T);
add_unary("acos");
add_unary("acosh");
add("add",VECTOR_T,VECTOR_T,VECTOR_T);
add("add",ROW_VECTOR_T,ROW_VECTOR_T,ROW_VECTOR_T);
add("add",MATRIX_T,MATRIX_T,MATRIX_T);
add("add",VECTOR_T,VECTOR_T,DOUBLE_T);
add("add",ROW_VECTOR_T,ROW_VECTOR_T,DOUBLE_T);
add("add",MATRIX_T,MATRIX_T,DOUBLE_T);
add("add",VECTOR_T,DOUBLE_T,VECTOR_T);
add("add",ROW_VECTOR_T,DOUBLE_T,ROW_VECTOR_T);
add("add",MATRIX_T,DOUBLE_T,MATRIX_T);
add_unary("asin");
add_unary("asinh");
add_unary("atan");
add_binary("atan2");
add_unary("atanh");
for (size_t i = 0; i < int_vector_types.size(); ++i) 
  for (size_t j = 0; j < vector_types.size(); ++j) {
    add("bernoulli_cdf",DOUBLE_T,int_vector_types[i],vector_types[j]);
    add("bernoulli_ccdf_log",DOUBLE_T,int_vector_types[i],vector_types[j]);
    add("bernoulli_cdf_log",DOUBLE_T,int_vector_types[i],vector_types[j]);
    add("bernoulli_log",DOUBLE_T,int_vector_types[i],vector_types[j]);
    add("bernoulli_logit_log",DOUBLE_T,int_vector_types[i],vector_types[j]);
  }
add("bernoulli_rng",INT_T,DOUBLE_T);
add("bessel_first_kind",DOUBLE_T,INT_T,DOUBLE_T);
add("bessel_second_kind",DOUBLE_T,INT_T,DOUBLE_T);
for (size_t i = 0; i < int_vector_types.size(); i++)
  for (size_t j = 0; j < int_vector_types.size(); j++)
    for (size_t k = 0; k < vector_types.size(); k++)
      for (size_t l = 0; l < vector_types.size(); l++) {
        add("beta_binomial_cdf",DOUBLE_T, // result
            int_vector_types[i],int_vector_types[j],vector_types[k],
            vector_types[l]);
        add("beta_binomial_ccdf_log",DOUBLE_T, // result
            int_vector_types[i],int_vector_types[j],vector_types[k],
            vector_types[l]);
        add("beta_binomial_cdf_log",DOUBLE_T, // result
            int_vector_types[i],int_vector_types[j],vector_types[k],
            vector_types[l]);
        add("beta_binomial_log",DOUBLE_T, // result
            int_vector_types[i],int_vector_types[j],vector_types[k],
            vector_types[l]);
      }
add("beta_binomial_rng",INT_T,INT_T,DOUBLE_T,DOUBLE_T);
for (size_t i = 0; i < vector_types.size(); ++i)
  for (size_t j = 0; j < vector_types.size(); ++j)
    for (size_t k = 0; k < vector_types.size(); ++k) {
      add("beta_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("beta_ccdf_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("beta_cdf",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("beta_cdf_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
    }
add_binary("beta_rng");
add("binary_log_loss",DOUBLE_T,INT_T,DOUBLE_T);
add_binary("binomial_coefficient_log");
for (size_t i = 0; i < int_vector_types.size(); ++i) {
  for (size_t j = 0; j < int_vector_types.size(); ++j) {
    for (size_t k = 0; k < vector_types.size(); ++k) {
      add("binomial_cdf",DOUBLE_T,
          int_vector_types[i],int_vector_types[j],vector_types[k]);
      add("binomial_ccdf_log",DOUBLE_T,
          int_vector_types[i],int_vector_types[j],vector_types[k]);
      add("binomial_cdf_log",DOUBLE_T,
          int_vector_types[i],int_vector_types[j],vector_types[k]);
      add("binomial_log",DOUBLE_T,
          int_vector_types[i],int_vector_types[j],vector_types[k]);
      add("binomial_logit_log",DOUBLE_T,
          int_vector_types[i],int_vector_types[j],vector_types[k]);
    }
  }
}
add("binomial_rng",INT_T,INT_T,DOUBLE_T);
for (size_t i = 0; i < int_vector_types.size(); ++i) {
  add("categorical_log",DOUBLE_T, int_vector_types[i],VECTOR_T);
  add("categorical_logit_log",DOUBLE_T, int_vector_types[i],VECTOR_T);
}
add("categorical_rng",INT_T,VECTOR_T);
for (size_t i = 0; i < vector_types.size(); ++i)
  for (size_t j = 0; j < vector_types.size(); ++j)
    for (size_t k = 0; k < vector_types.size(); ++k) {
      add("cauchy_cdf",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("cauchy_ccdf_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("cauchy_cdf_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("cauchy_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
    }
add_binary("cauchy_rng");
add_unary("cbrt");
add_unary("ceil");
add("cholesky_decompose",MATRIX_T,MATRIX_T);
for (size_t i = 0; i < vector_types.size(); ++i)
  for (size_t j = 0; j < vector_types.size(); ++j) {
      add("chi_square_cdf",
          DOUBLE_T, // result
          vector_types[i], vector_types[j]); // args
      add("chi_square_ccdf_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j]); // args
      add("chi_square_cdf_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j]); // args
      add("chi_square_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j]); // args
  }
add_unary("chi_square_rng");
add("block",MATRIX_T,MATRIX_T,INT_T,INT_T,INT_T,INT_T);
add("col",VECTOR_T,MATRIX_T,INT_T);
add("cols",INT_T,VECTOR_T);
add("cols",INT_T,ROW_VECTOR_T);
add("cols",INT_T,MATRIX_T);
add("columns_dot_product",ROW_VECTOR_T,VECTOR_T,VECTOR_T);
add("columns_dot_product",ROW_VECTOR_T,ROW_VECTOR_T,ROW_VECTOR_T);
add("columns_dot_product",ROW_VECTOR_T,MATRIX_T,MATRIX_T);
add("columns_dot_self",ROW_VECTOR_T,VECTOR_T);
add("columns_dot_self",ROW_VECTOR_T,ROW_VECTOR_T);
add("columns_dot_self",ROW_VECTOR_T,MATRIX_T);
add_unary("cos");
add_unary("cosh");
add("cumulative_sum", expr_type(DOUBLE_T,1U), expr_type(DOUBLE_T,1U));
add("cumulative_sum", VECTOR_T, VECTOR_T);
add("cumulative_sum", ROW_VECTOR_T, ROW_VECTOR_T);
add("determinant",DOUBLE_T,MATRIX_T);
add("diagonal",VECTOR_T,MATRIX_T);
add("diag_matrix",MATRIX_T,VECTOR_T);
add("dirichlet_log",DOUBLE_T,VECTOR_T,VECTOR_T);
add("dirichlet_rng",VECTOR_T,VECTOR_T);
add("divide",DOUBLE_T,DOUBLE_T,DOUBLE_T);
add("divide",VECTOR_T,VECTOR_T,DOUBLE_T);
add("divide",ROW_VECTOR_T,ROW_VECTOR_T,DOUBLE_T);
add("divide",MATRIX_T,MATRIX_T,DOUBLE_T);
add("dist",DOUBLE_T,VECTOR_T,VECTOR_T);
add("dist",DOUBLE_T,ROW_VECTOR_T,ROW_VECTOR_T);
add("dist",DOUBLE_T,VECTOR_T,ROW_VECTOR_T);
add("dist",DOUBLE_T,ROW_VECTOR_T,VECTOR_T);
add("squared_dist",DOUBLE_T,VECTOR_T,VECTOR_T);
add("squared_dist",DOUBLE_T,ROW_VECTOR_T,ROW_VECTOR_T);
add("squared_dist",DOUBLE_T,VECTOR_T,ROW_VECTOR_T);
add("squared_dist",DOUBLE_T,ROW_VECTOR_T,VECTOR_T);
add("squared_dist",DOUBLE_T,MATRIX_T,MATRIX_T);
add("quad_form",DOUBLE_T,MATRIX_T,VECTOR_T);
add("quad_form",MATRIX_T,MATRIX_T,MATRIX_T);
add("quad_form_sym",DOUBLE_T,MATRIX_T,VECTOR_T);
add("quad_form_sym",MATRIX_T,MATRIX_T,MATRIX_T);
add("trace_quad_form",DOUBLE_T,MATRIX_T,VECTOR_T);
add("trace_quad_form",DOUBLE_T,MATRIX_T,MATRIX_T);
add("trace_gen_quad_form",DOUBLE_T,MATRIX_T,MATRIX_T,MATRIX_T);
add("dot_product",DOUBLE_T,VECTOR_T,VECTOR_T);
add("dot_product",DOUBLE_T,ROW_VECTOR_T,ROW_VECTOR_T);
add("dot_product",DOUBLE_T,VECTOR_T,ROW_VECTOR_T);
add("dot_product",DOUBLE_T,ROW_VECTOR_T,VECTOR_T);
add("dot_product",DOUBLE_T,expr_type(DOUBLE_T,1U),expr_type(DOUBLE_T,1U)); // vectorized
add("dot_self",DOUBLE_T,VECTOR_T);
add("dot_self",DOUBLE_T,ROW_VECTOR_T);
for (size_t i = 0; i < vector_types.size(); ++i)
  for (size_t j = 0; j < vector_types.size(); ++j)
    for (size_t k = 0; k < vector_types.size(); ++k) {
      add("double_exponential_cdf",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("double_exponential_ccdf_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("double_exponential_cdf_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("double_exponential_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
    }
add_binary("double_exponential_rng");
add_nullary("e");
add("eigenvalues_sym",VECTOR_T,MATRIX_T);
add("eigenvectors_sym",MATRIX_T,MATRIX_T);
add("elt_divide",VECTOR_T,VECTOR_T,VECTOR_T);
add("elt_divide",ROW_VECTOR_T,ROW_VECTOR_T,ROW_VECTOR_T);
add("elt_divide",MATRIX_T,MATRIX_T,MATRIX_T);
add("elt_multiply",VECTOR_T,VECTOR_T,VECTOR_T);
add("elt_multiply",ROW_VECTOR_T,ROW_VECTOR_T,ROW_VECTOR_T);
add("elt_multiply",MATRIX_T,MATRIX_T,MATRIX_T);
add("diag_pre_multiply",MATRIX_T,MATRIX_T,MATRIX_T);
add("diag_pre_multiply",MATRIX_T,VECTOR_T,MATRIX_T);
add("diag_pre_multiply",MATRIX_T,ROW_VECTOR_T,MATRIX_T);
add("diag_post_multiply",MATRIX_T,MATRIX_T,MATRIX_T);
add("diag_post_multiply",MATRIX_T,MATRIX_T,VECTOR_T);
add("diag_post_multiply",MATRIX_T,MATRIX_T,ROW_VECTOR_T);
add_nullary("epsilon");
add_unary("erf");
add_unary("erfc");
add_unary("exp");
add("exp",VECTOR_T,VECTOR_T);
add("exp",ROW_VECTOR_T,ROW_VECTOR_T);
add("exp",MATRIX_T,MATRIX_T);
add_unary("exp2");
add_unary("expm1");
for (size_t i = 0; i < vector_types.size(); ++i)
  for (size_t j = 0; j < vector_types.size(); ++j) {
      add("exponential_cdf",
          DOUBLE_T, // result
          vector_types[i], vector_types[j]); // args
      add("exponential_ccdf_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j]); // args
      add("exponential_cdf_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j]); // args
      add("exponential_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j]); // args
  }
add_unary("exponential_rng");
for (size_t i = 0; i < vector_types.size(); ++i)
  for (size_t j = 0; j < vector_types.size(); ++j) 
    for (size_t k = 0; k < vector_types.size(); ++k)
      for (size_t l = 0; l < vector_types.size(); ++l) {
        add("exp_mod_normal_cdf",
            DOUBLE_T, // result
            vector_types[i], vector_types[j], vector_types[k], 
            vector_types[l]); // args
        add("exp_mod_normal_ccdf_log",
            DOUBLE_T, // result
            vector_types[i], vector_types[j], vector_types[k], 
            vector_types[l]); // args
        add("exp_mod_normal_cdf_log",
            DOUBLE_T, // result
            vector_types[i], vector_types[j], vector_types[k], 
            vector_types[l]); // args
        add("exp_mod_normal_log",
            DOUBLE_T, // result
            vector_types[i], vector_types[j], vector_types[k], 
            vector_types[l]); // args
      }
add_ternary("exp_mod_normal_rng");
add_unary("fabs");
add_binary("falling_factorial");
add_binary("fdim");
add_unary("floor");
add_ternary("fma");
add_binary("fmax");
add_binary("fmin");
add_binary("fmod");
for (size_t i = 0; i < vector_types.size(); ++i) 
  for (size_t j = 0; j < vector_types.size(); ++j) 
    for (size_t k = 0; k < vector_types.size(); ++k) {
      add("gamma_cdf",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("gamma_ccdf_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("gamma_cdf_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("gamma_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
    }
add_binary("gamma_p");
add_binary("gamma_q");
add_binary("gamma_rng");
for (size_t i = 0; i < vector_types.size(); ++i) 
  for (size_t j = 0; j < vector_types.size(); ++j) 
    for (size_t k = 0; k < vector_types.size(); ++k) {
      add("gumbel_cdf",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("gumbel_ccdf_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("gumbel_cdf_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("gumbel_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
    }
add("gaussian_dlm_obs_log",DOUBLE_T,MATRIX_T,MATRIX_T,MATRIX_T,MATRIX_T,MATRIX_T,VECTOR_T,MATRIX_T);
add("gaussian_dlm_obs_log",DOUBLE_T,MATRIX_T,MATRIX_T,MATRIX_T,VECTOR_T,MATRIX_T,VECTOR_T,MATRIX_T);
add_binary("gumbel_rng");
add("head",ROW_VECTOR_T, ROW_VECTOR_T,INT_T);
add("head",VECTOR_T, VECTOR_T,INT_T);
for (size_t i = 0; i < base_types.size(); ++i) {
  add("head",expr_type(base_types[i],1U), expr_type(base_types[i],1U),INT_T);
  add("head",expr_type(base_types[i],2U), expr_type(base_types[i],2U),INT_T);
  add("head",expr_type(base_types[i],3U), expr_type(base_types[i],3U),INT_T);
 }
add("hypergeometric_log",DOUBLE_T, INT_T,INT_T,INT_T,INT_T);
add("hypergeometric_rng",INT_T, INT_T,INT_T,INT_T);
add_binary("hypot");
add("if_else",DOUBLE_T,INT_T,DOUBLE_T,DOUBLE_T);
add("int_step",INT_T,DOUBLE_T);
add("int_step",INT_T,INT_T);
for (size_t i = 0; i < vector_types.size(); ++i)
  for (size_t j = 0; j < vector_types.size(); ++j) {
      add("inv_chi_square_cdf",
          DOUBLE_T, // result
          vector_types[i], vector_types[j]); // args
      add("inv_chi_square_ccdf_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j]); // args
      add("inv_chi_square_cdf_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j]); // args
      add("inv_chi_square_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j]); // args
  }
add_unary("inv_chi_square_rng");
add_unary("inv_cloglog");
for (size_t i = 0; i < vector_types.size(); ++i) 
  for (size_t j = 0; j < vector_types.size(); ++j) 
    for (size_t k = 0; k < vector_types.size(); ++k) {
      add("inv_gamma_cdf",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("inv_gamma_ccdf_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("inv_gamma_cdf_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("inv_gamma_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args 
   }
add_binary("inv_gamma_rng");
add_unary("inv_logit");
add_unary("inv_square");
add_unary("inv_sqrt");
add("inv_wishart_log",DOUBLE_T, MATRIX_T,DOUBLE_T,MATRIX_T);
add("inv_wishart_rng",MATRIX_T,DOUBLE_T,MATRIX_T);
add_unary("inv");
add("inverse",MATRIX_T,MATRIX_T);
add("inverse_spd",MATRIX_T,MATRIX_T);
add_binary("lbeta");
add_unary("lgamma");
add("lkj_corr_cholesky_log",DOUBLE_T, MATRIX_T,DOUBLE_T);
add("lkj_corr_cholesky_rng",MATRIX_T,INT_T,DOUBLE_T);
add("lkj_corr_log",DOUBLE_T, MATRIX_T,DOUBLE_T);
add("lkj_corr_rng",MATRIX_T,INT_T,DOUBLE_T);
add("lkj_cov_log",DOUBLE_T, MATRIX_T,VECTOR_T,VECTOR_T,DOUBLE_T);
add("lmgamma",DOUBLE_T,INT_T,DOUBLE_T);
for (size_t i = 0; i < primitive_types.size(); ++i) {
  add("logical_negation",INT_T,primitive_types[i]);
  for (size_t j = 0; j < primitive_types.size(); ++j) {
    add("logical_or",INT_T,primitive_types[i], primitive_types[j]);
    add("logical_and",INT_T,primitive_types[i], primitive_types[j]);
    add("logical_eq",INT_T,primitive_types[i], primitive_types[j]);
    add("logical_neq",INT_T,primitive_types[i], primitive_types[j]);
    add("logical_lt",INT_T,primitive_types[i], primitive_types[j]);
    add("logical_lte",INT_T,primitive_types[i], primitive_types[j]);
    add("logical_gt",INT_T,primitive_types[i], primitive_types[j]);
    add("logical_gte",INT_T,primitive_types[i], primitive_types[j]);
  }
}
add_unary("log");
add("log",VECTOR_T,VECTOR_T);
add("log",ROW_VECTOR_T,ROW_VECTOR_T);
add("log",MATRIX_T,MATRIX_T);
add("log_determinant",DOUBLE_T,MATRIX_T);
add_binary("log_falling_factorial");
add_binary("log_rising_factorial");
add("log_softmax",VECTOR_T,VECTOR_T);
add("log_sum_exp",DOUBLE_T, expr_type(DOUBLE_T,1U));
add("log_sum_exp",DOUBLE_T, VECTOR_T);
add("log_sum_exp",DOUBLE_T, ROW_VECTOR_T);
add("log_sum_exp",DOUBLE_T, MATRIX_T);
add_binary("log_sum_exp");
add_nullary("log10");
add_unary("log10");
add_unary("log1m");
add_unary("log1m_exp");
add_unary("log1p");
add_unary("log1p_exp");
add_binary("log_diff_exp");
add_unary("log_inv_logit");
add_unary("log1m_inv_logit");
add_nullary("log2");
add_unary("log2");
for (size_t i = 0; i < vector_types.size(); ++i) 
  for (size_t j = 0; j < vector_types.size(); ++j) 
    for (size_t k = 0; k < vector_types.size(); ++k) {
      add("logistic_cdf",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("logistic_ccdf_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("logistic_cdf_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("logistic_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args    
    }
add_binary("logistic_rng");
add_unary("logit");
for (size_t i = 0; i < vector_types.size(); ++i) 
  for (size_t j = 0; j < vector_types.size(); ++j) 
    for (size_t k = 0; k < vector_types.size(); ++k) {
      add("lognormal_cdf",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("lognormal_ccdf_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("lognormal_cdf_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("lognormal_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
    }
add_binary("lognormal_rng");
add("max",INT_T,expr_type(INT_T,1));
add("max",DOUBLE_T,expr_type(DOUBLE_T,1));
add("max",DOUBLE_T,VECTOR_T);
add("max",DOUBLE_T,ROW_VECTOR_T);
add("max",DOUBLE_T,MATRIX_T);
add("max",INT_T,INT_T,INT_T);
add("mdivide_left",VECTOR_T,MATRIX_T,VECTOR_T);
add("mdivide_left",MATRIX_T,MATRIX_T,MATRIX_T);
add("mdivide_right",ROW_VECTOR_T,ROW_VECTOR_T,MATRIX_T);
add("mdivide_right",MATRIX_T,MATRIX_T,MATRIX_T);
add("mdivide_left_tri_low",MATRIX_T,MATRIX_T,MATRIX_T);
add("mdivide_left_tri_low",VECTOR_T,MATRIX_T,VECTOR_T);
add("mdivide_right_tri_low",ROW_VECTOR_T,ROW_VECTOR_T,MATRIX_T);
add("mdivide_right_tri_low",MATRIX_T,MATRIX_T,MATRIX_T);
add("mean",DOUBLE_T,expr_type(DOUBLE_T,1));
add("mean",DOUBLE_T,VECTOR_T);
add("mean",DOUBLE_T,ROW_VECTOR_T);
add("mean",DOUBLE_T,MATRIX_T);
add("min",INT_T,expr_type(INT_T,1));
add("min",DOUBLE_T,expr_type(DOUBLE_T,1));
add("min",DOUBLE_T,VECTOR_T);
add("min",DOUBLE_T,ROW_VECTOR_T);
add("min",DOUBLE_T,MATRIX_T);
add("min",INT_T,INT_T,INT_T);
add("minus",DOUBLE_T,DOUBLE_T);
add("minus",VECTOR_T,VECTOR_T);
add("minus",ROW_VECTOR_T,ROW_VECTOR_T);
add("minus",MATRIX_T,MATRIX_T);
add("matrix_normal_prec_log",DOUBLE_T,MATRIX_T,MATRIX_T,MATRIX_T,MATRIX_T);
add("modified_bessel_first_kind",DOUBLE_T,INT_T,DOUBLE_T);
add("modified_bessel_second_kind",DOUBLE_T,INT_T,DOUBLE_T);
add("multi_gp_log",DOUBLE_T,MATRIX_T,MATRIX_T,VECTOR_T);
add("multi_normal_cholesky_log",DOUBLE_T, VECTOR_T,VECTOR_T,MATRIX_T);
add("multi_normal_prec_log",DOUBLE_T, VECTOR_T,VECTOR_T,MATRIX_T);
add("multi_normal_log",DOUBLE_T, VECTOR_T,VECTOR_T,MATRIX_T);
add("multi_normal_rng",VECTOR_T,VECTOR_T,MATRIX_T);
add("multi_student_t_log",DOUBLE_T, VECTOR_T,DOUBLE_T,VECTOR_T,MATRIX_T);
add("multi_student_t_rng",VECTOR_T, DOUBLE_T,VECTOR_T,MATRIX_T);
add("multinomial_log",DOUBLE_T, expr_type(INT_T,1U), VECTOR_T);
add("multinomial_rng",expr_type(INT_T,1U), VECTOR_T, INT_T);
add("multiply",DOUBLE_T,DOUBLE_T,DOUBLE_T);
add("multiply",VECTOR_T,VECTOR_T,DOUBLE_T);
add("multiply",ROW_VECTOR_T,ROW_VECTOR_T,DOUBLE_T);
add("multiply",MATRIX_T,MATRIX_T,DOUBLE_T);
add("multiply",DOUBLE_T,ROW_VECTOR_T,VECTOR_T);
add("multiply",MATRIX_T,VECTOR_T,ROW_VECTOR_T);  
add("multiply",VECTOR_T,MATRIX_T,VECTOR_T);
add("multiply",ROW_VECTOR_T,ROW_VECTOR_T,MATRIX_T);
add("multiply",MATRIX_T,MATRIX_T,MATRIX_T);
add("multiply",VECTOR_T,DOUBLE_T,VECTOR_T);
add("multiply",ROW_VECTOR_T,DOUBLE_T,ROW_VECTOR_T);
add("multiply",MATRIX_T,DOUBLE_T,MATRIX_T);
add("multiply_lower_tri_self_transpose",MATRIX_T,MATRIX_T);
add("tcrossprod",MATRIX_T,MATRIX_T);
add("crossprod",MATRIX_T,MATRIX_T);
add_binary("multiply_log");
for (size_t i = 0; i < int_vector_types.size(); ++i) 
  for (size_t j = 0; j < vector_types.size(); ++j) 
    for (size_t k = 0; k < vector_types.size(); ++k) {
      add("neg_binomial_cdf",DOUBLE_T,
          int_vector_types[i],vector_types[j],vector_types[k]);
      add("neg_binomial_ccdf_log",DOUBLE_T,
          int_vector_types[i],vector_types[j],vector_types[k]);
      add("neg_binomial_cdf_log",DOUBLE_T,
          int_vector_types[i],vector_types[j],vector_types[k]);
      add("neg_binomial_log",DOUBLE_T,
          int_vector_types[i],vector_types[j],vector_types[k]);
    }
add("neg_binomial_rng",INT_T,DOUBLE_T,DOUBLE_T);
add_nullary("negative_epsilon");
add_nullary("negative_infinity");
for (size_t i = 0; i < vector_types.size(); ++i) 
  for (size_t j = 0; j < vector_types.size(); ++j) 
    for (size_t k = 0; k < vector_types.size(); ++k) {
      add("normal_cdf",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("normal_ccdf_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("normal_cdf_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("normal_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
    }
add_binary("normal_rng");
add_nullary("not_a_number");
add("ordered_logistic_log",DOUBLE_T,INT_T,DOUBLE_T,VECTOR_T);
add("ordered_logistic_rng",INT_T,DOUBLE_T,VECTOR_T);
add_binary("owens_t");
for (size_t i = 0; i < vector_types.size(); ++i) 
  for (size_t j = 0; j < vector_types.size(); ++j) 
    for (size_t k = 0; k < vector_types.size(); ++k) {
      add("pareto_cdf",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("pareto_ccdf_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("pareto_cdf_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("pareto_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
    }
add_binary("pareto_rng");
add_unary("Phi");
add_unary("Phi_approx");
add_nullary("pi");
for (size_t i = 0; i < int_vector_types.size(); ++i)
  for (size_t j = 0; j < vector_types.size(); ++j) {
    add("poisson_cdf",DOUBLE_T, int_vector_types[i],vector_types[j]);
    add("poisson_ccdf_log",DOUBLE_T, int_vector_types[i],vector_types[j]);
    add("poisson_cdf_log",DOUBLE_T, int_vector_types[i],vector_types[j]);
    add("poisson_log",DOUBLE_T, int_vector_types[i],vector_types[j]);
    add("poisson_log_log",DOUBLE_T, int_vector_types[i],vector_types[j]);
  }
add("poisson_rng",INT_T,DOUBLE_T);
add_nullary("positive_infinity");
add_binary("pow");
add("prod",INT_T,expr_type(INT_T,1));
add("prod",DOUBLE_T,expr_type(DOUBLE_T,1));
add("prod",DOUBLE_T,VECTOR_T);
add("prod",DOUBLE_T,ROW_VECTOR_T);
add("prod",DOUBLE_T,MATRIX_T);
add("rank",INT_T,expr_type(INT_T,1),INT_T);
add("rank",INT_T,expr_type(DOUBLE_T,1),INT_T);
add("rank",INT_T,VECTOR_T,INT_T);
add("rank",INT_T,ROW_VECTOR_T,INT_T);
for (size_t i = 0; i < vector_types.size(); ++i)
  for (size_t j = 0; j < vector_types.size(); ++j) {
    add("rayleigh_log",
        DOUBLE_T, // result
        vector_types[i], vector_types[j]); // args
    add("rayleigh_cdf",
        DOUBLE_T, // result
        vector_types[i], vector_types[j]); // args
    add("rayleigh_ccdf_log",
        DOUBLE_T, // result
        vector_types[i], vector_types[j]); // args
    add("rayleigh_cdf_log",
        DOUBLE_T, // result
        vector_types[i], vector_types[j]); // args
  }
add_unary("rayleigh_rng");
for (size_t i = 0; i < base_types.size(); ++i) {
  add("rep_array",expr_type(base_types[i],1), base_types[i], INT_T);
  add("rep_array",expr_type(base_types[i],2), base_types[i], INT_T,INT_T);
  add("rep_array",expr_type(base_types[i],3), base_types[i], INT_T,INT_T,INT_T);
  for (size_t j = 1; j <= 3; ++j) {
    add("rep_array",expr_type(base_types[i],j + 1), expr_type(base_types[i],j),  INT_T);
    add("rep_array",expr_type(base_types[i],j + 2), expr_type(base_types[i],j),  INT_T,INT_T);
    add("rep_array",expr_type(base_types[i],j + 3), expr_type(base_types[i],j),  INT_T,INT_T,INT_T);
  }
}
add("rep_matrix", MATRIX_T, DOUBLE_T, INT_T,INT_T);
add("rep_matrix", MATRIX_T, VECTOR_T, INT_T);
add("rep_matrix", MATRIX_T, ROW_VECTOR_T, INT_T);
add("rep_row_vector", ROW_VECTOR_T, DOUBLE_T, INT_T);
add("rep_vector", VECTOR_T, DOUBLE_T, INT_T);
add_binary("rising_factorial");
add_unary("round");
add("row",ROW_VECTOR_T,MATRIX_T,INT_T);
add("rows",INT_T,VECTOR_T);
add("rows",INT_T,ROW_VECTOR_T);
add("rows",INT_T,MATRIX_T);
add("rows_dot_product",VECTOR_T,VECTOR_T,VECTOR_T);
add("rows_dot_product",VECTOR_T,ROW_VECTOR_T,ROW_VECTOR_T);
add("rows_dot_product",VECTOR_T,MATRIX_T,MATRIX_T);
add("rows_dot_self",VECTOR_T,VECTOR_T);
add("rows_dot_self",VECTOR_T,ROW_VECTOR_T);
add("rows_dot_self",VECTOR_T,MATRIX_T);
for (size_t i = 0; i < vector_types.size(); ++i) 
  for (size_t j = 0; j < vector_types.size(); ++j) 
    for (size_t k = 0; k < vector_types.size(); ++k) {
      add("scaled_inv_chi_square_cdf",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("scaled_inv_chi_square_ccdf_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("scaled_inv_chi_square_cdf_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
      add("scaled_inv_chi_square_log",
          DOUBLE_T, // result
          vector_types[i], vector_types[j], vector_types[k]); // args
    }
add_binary("scaled_inv_chi_square_rng");
add("sd",DOUBLE_T,expr_type(DOUBLE_T,1));
add("sd",DOUBLE_T,VECTOR_T);
add("sd",DOUBLE_T,ROW_VECTOR_T);
add("sd",DOUBLE_T,MATRIX_T);
add("segment",ROW_VECTOR_T, ROW_VECTOR_T,INT_T,INT_T);
add("segment",VECTOR_T, VECTOR_T,INT_T,INT_T);
for (size_t i = 0; i < base_types.size(); ++i) {
  add("segment",expr_type(base_types[i],1U), expr_type(base_types[i],1U),INT_T,INT_T);
  add("segment",expr_type(base_types[i],2U), expr_type(base_types[i],2U),INT_T,INT_T);
  add("segment",expr_type(base_types[i],3U), expr_type(base_types[i],3U),INT_T,INT_T);
 }
add_unary("sin");
add("singular_values",VECTOR_T,MATRIX_T);
add_unary("sinh");
for (size_t i = 0; i < vector_types.size(); ++i) 
  for (size_t j = 0; j < vector_types.size(); ++j) 
    for (size_t k = 0; k < vector_types.size(); ++k) 
      for (size_t l = 0; l < vector_types.size(); ++l) {
        add("skew_normal_cdf",
            DOUBLE_T, // result
            vector_types[i], vector_types[j], vector_types[k], vector_types[l]); // args
        add("skew_normal_ccdf_log",
            DOUBLE_T, // result
            vector_types[i], vector_types[j], vector_types[k], vector_types[l]); // args
        add("skew_normal_cdf_log",
            DOUBLE_T, // result
            vector_types[i], vector_types[j], vector_types[k], vector_types[l]); // args
        add("skew_normal_log",
            DOUBLE_T, // result
            vector_types[i], vector_types[j], vector_types[k], vector_types[l]); // args
      }
add_ternary("skew_normal_rng");
add("softmax",VECTOR_T,VECTOR_T);
add("sort_asc",expr_type(INT_T,1),expr_type(INT_T,1));
add("sort_asc",expr_type(DOUBLE_T,1),expr_type(DOUBLE_T,1));
add("sort_asc",VECTOR_T,VECTOR_T);
add("sort_asc",ROW_VECTOR_T,ROW_VECTOR_T);
add("sort_desc",expr_type(INT_T,1),expr_type(INT_T,1));
add("sort_desc",expr_type(DOUBLE_T,1),expr_type(DOUBLE_T,1));
add("sort_desc",VECTOR_T,VECTOR_T);
add("sort_desc",ROW_VECTOR_T,ROW_VECTOR_T);
add_unary("sqrt");
add_nullary("sqrt2");
add_unary("square");
add_unary("step");
for (size_t i = 0; i < vector_types.size(); ++i) 
  for (size_t j = 0; j < vector_types.size(); ++j) 
    for (size_t k = 0; k < vector_types.size(); ++k) 
      for (size_t l = 0; l < vector_types.size(); ++l) {
        add("student_t_cdf",
            DOUBLE_T, // result
            vector_types[i], vector_types[j], vector_types[k], vector_types[l]); // args
        add("student_t_ccdf_log",
            DOUBLE_T, // result
            vector_types[i], vector_types[j], vector_types[k], vector_types[l]); // args
        add("student_t_cdf_log",
            DOUBLE_T, // result
            vector_types[i], vector_types[j], vector_types[k], vector_types[l]); // args
        add("student_t_log",
            DOUBLE_T, // result
            vector_types[i], vector_types[j], vector_types[k], vector_types[l]); // args
      }
add_ternary("student_t_rng");
add("sub_col",VECTOR_T,MATRIX_T,INT_T,INT_T,INT_T);
add("sub_row",ROW_VECTOR_T,MATRIX_T,INT_T,INT_T,INT_T);
add("subtract",VECTOR_T,VECTOR_T,VECTOR_T);
add("subtract",ROW_VECTOR_T,ROW_VECTOR_T,ROW_VECTOR_T);
add("subtract",MATRIX_T,MATRIX_T,MATRIX_T);
add("subtract",VECTOR_T,VECTOR_T,DOUBLE_T);
add("subtract",ROW_VECTOR_T,ROW_VECTOR_T,DOUBLE_T);
add("subtract",MATRIX_T,MATRIX_T,DOUBLE_T);
add("subtract",VECTOR_T,DOUBLE_T,VECTOR_T);
add("subtract",ROW_VECTOR_T,DOUBLE_T,ROW_VECTOR_T);
add("subtract",MATRIX_T,DOUBLE_T,MATRIX_T);
add("sum",INT_T,expr_type(INT_T,1));
add("sum",DOUBLE_T,expr_type(DOUBLE_T,1));
add("sum",DOUBLE_T,VECTOR_T);
add("sum",DOUBLE_T,ROW_VECTOR_T);
add("sum",DOUBLE_T,MATRIX_T);
// see head() and segment()
add("tail",ROW_VECTOR_T, ROW_VECTOR_T,INT_T);
add("tail",VECTOR_T, VECTOR_T,INT_T);
for (size_t i = 0; i < base_types.size(); ++i) {
  add("tail",expr_type(base_types[i],1U), expr_type(base_types[i],1U),INT_T);
  add("tail",expr_type(base_types[i],2U), expr_type(base_types[i],2U),INT_T);
  add("tail",expr_type(base_types[i],3U), expr_type(base_types[i],3U),INT_T);
 }
add_unary("tan");
add_unary("tanh");
add_unary("tgamma");
add("to_vector", VECTOR_T, ROW_VECTOR_T);
add("to_vector", VECTOR_T, MATRIX_T);
add("trace",DOUBLE_T,MATRIX_T);
add("transpose",ROW_VECTOR_T,VECTOR_T);
add("transpose",VECTOR_T,ROW_VECTOR_T);
add("transpose",MATRIX_T,MATRIX_T);
add_unary("trunc");
for (size_t i = 0; i < vector_types.size(); ++i) 
  for (size_t j = 0; j < vector_types.size(); ++j) 
    for (size_t k = 0; k < vector_types.size(); ++k) {
        add("uniform_cdf",
            DOUBLE_T, // result
            vector_types[i], vector_types[j], vector_types[k]); // args
        add("uniform_ccdf_log",
            DOUBLE_T, // result
            vector_types[i], vector_types[j], vector_types[k]); // args
        add("uniform_cdf_log",
            DOUBLE_T, // result
            vector_types[i], vector_types[j], vector_types[k]); // args
        add("uniform_log",
            DOUBLE_T, // result
            vector_types[i], vector_types[j], vector_types[k]); // args
    }
add_binary("uniform_rng");
add("variance",DOUBLE_T,expr_type(DOUBLE_T,1));
add("variance",DOUBLE_T,VECTOR_T);
add("variance",DOUBLE_T,ROW_VECTOR_T);
add("variance",DOUBLE_T,MATRIX_T);
for (size_t i = 0; i < vector_types.size(); ++i)
  for (size_t j = 0; j < vector_types.size(); ++j) 
    for (size_t k = 0; k < vector_types.size(); ++k) {
        add("weibull_cdf",
            DOUBLE_T, // result
            vector_types[i], vector_types[j], vector_types[k]); // args
        add("weibull_ccdf_log",
            DOUBLE_T, // result
            vector_types[i], vector_types[j], vector_types[k]); // args
        add("weibull_cdf_log",
            DOUBLE_T, // result
            vector_types[i], vector_types[j], vector_types[k]); // args
        add("weibull_log",
            DOUBLE_T, // result
            vector_types[i], vector_types[j], vector_types[k]); // args
    }
add_binary("weibull_rng");
add("wishart_log",DOUBLE_T, MATRIX_T,DOUBLE_T,MATRIX_T);
add("wishart_rng",MATRIX_T, DOUBLE_T,MATRIX_T);

// dims() is polymorphic by size
for (size_t i = 0; i < 8; ++i) {
  add("dims",expr_type(INT_T,1),expr_type(INT_T,i));
  add("dims",expr_type(INT_T,1),expr_type(DOUBLE_T,i));
  add("dims",expr_type(INT_T,1),expr_type(VECTOR_T,i));
  add("dims",expr_type(INT_T,1),expr_type(ROW_VECTOR_T,i));
  add("dims",expr_type(INT_T,1),expr_type(MATRIX_T,i));
}
// size() is polymorphic over arrays, so start i at 1
for (size_t i = 1; i < 8; ++i) {
  add("size",INT_T,expr_type(INT_T,i));
  add("size",INT_T,expr_type(DOUBLE_T,i));
  add("size",INT_T,expr_type(VECTOR_T,i));
  add("size",INT_T,expr_type(ROW_VECTOR_T,i));
  add("size",INT_T,expr_type(MATRIX_T,i));
}
//------------------------------------------------------------

add("trunc_normal_log",DOUBLE_T, DOUBLE_T,DOUBLE_T,DOUBLE_T,DOUBLE_T,DOUBLE_T);

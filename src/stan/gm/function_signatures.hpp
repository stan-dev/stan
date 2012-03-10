// included from constructor for function_signatures() in src/stan/gm/ast.hpp
add_unary("exp");
add_unary("log");
add_unary("log10");
add_binary("pow");
add_unary("sqrt");

add_unary("cos");
add_unary("sin");
add_unary("tan");

add_unary("acos");
add_unary("asin");
add_unary("atan");
add_binary("atan2");

add_unary("cosh");
add_unary("sinh");
add_unary("tanh");

add_unary("fabs");
add_unary("floor");
add_unary("ceil");
add_binary("fmod");
add_unary("abs");

add("exp",VECTOR_T,VECTOR_T);
add("exp",ROW_VECTOR_T,ROW_VECTOR_T);
add("exp",MATRIX_T,MATRIX_T);

add("log",VECTOR_T,VECTOR_T);
add("log",ROW_VECTOR_T,ROW_VECTOR_T);
add("log",MATRIX_T,MATRIX_T);

add("rows",INT_T,VECTOR_T);
add("rows",INT_T,ROW_VECTOR_T);
add("rows",INT_T,MATRIX_T);

add("cols",INT_T,VECTOR_T);
add("cols",INT_T,ROW_VECTOR_T);
add("cols",INT_T,MATRIX_T);

add("determinant",DOUBLE_T,MATRIX_T);
add("trace",DOUBLE_T,MATRIX_T);

add("dot_product",DOUBLE_T,VECTOR_T,VECTOR_T);
add("dot_product",DOUBLE_T,ROW_VECTOR_T,ROW_VECTOR_T);
add("dot_product",DOUBLE_T,VECTOR_T,ROW_VECTOR_T);
add("dot_product",DOUBLE_T,ROW_VECTOR_T,VECTOR_T);

add("min",DOUBLE_T,expr_type(DOUBLE_T,1));
add("min",DOUBLE_T,VECTOR_T);
add("min",DOUBLE_T,ROW_VECTOR_T);
add("min",DOUBLE_T,MATRIX_T);

add("max",DOUBLE_T,expr_type(DOUBLE_T,1));
add("max",DOUBLE_T,VECTOR_T);
add("max",DOUBLE_T,ROW_VECTOR_T);
add("max",DOUBLE_T,MATRIX_T);

add("mean",DOUBLE_T,expr_type(DOUBLE_T,1));
add("mean",DOUBLE_T,VECTOR_T);
add("mean",DOUBLE_T,ROW_VECTOR_T);
add("mean",DOUBLE_T,MATRIX_T);

add("variance",DOUBLE_T,expr_type(DOUBLE_T,1));
add("variance",DOUBLE_T,VECTOR_T);
add("variance",DOUBLE_T,ROW_VECTOR_T);
add("variance",DOUBLE_T,MATRIX_T);

add("sd",DOUBLE_T,expr_type(DOUBLE_T,1));
add("sd",DOUBLE_T,VECTOR_T);
add("sd",DOUBLE_T,ROW_VECTOR_T);
add("sd",DOUBLE_T,MATRIX_T);

add("sum",DOUBLE_T,expr_type(DOUBLE_T,1));
add("sum",DOUBLE_T,VECTOR_T);
add("sum",DOUBLE_T,ROW_VECTOR_T);
add("sum",DOUBLE_T,MATRIX_T);

add("prod",DOUBLE_T,VECTOR_T);
add("prod",DOUBLE_T,ROW_VECTOR_T);
add("prod",DOUBLE_T,MATRIX_T);

add("add",VECTOR_T,VECTOR_T,VECTOR_T);
add("add",ROW_VECTOR_T,ROW_VECTOR_T,ROW_VECTOR_T);
add("add",MATRIX_T,MATRIX_T,MATRIX_T);

add("subtract",VECTOR_T,VECTOR_T,VECTOR_T);
add("subtract",ROW_VECTOR_T,ROW_VECTOR_T,ROW_VECTOR_T);
add("subtract",MATRIX_T,MATRIX_T,MATRIX_T);

add("minus",DOUBLE_T,DOUBLE_T);
add("minus",VECTOR_T,VECTOR_T);
add("minus",ROW_VECTOR_T,ROW_VECTOR_T);
add("minus",MATRIX_T,MATRIX_T);

add("divide",DOUBLE_T,DOUBLE_T,DOUBLE_T);
add("divide",VECTOR_T,VECTOR_T,DOUBLE_T);
add("divide",ROW_VECTOR_T,ROW_VECTOR_T,DOUBLE_T);
add("divide",MATRIX_T,MATRIX_T,DOUBLE_T);

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

add("elt_multiply",VECTOR_T,VECTOR_T,VECTOR_T);
add("elt_multiply",ROW_VECTOR_T,ROW_VECTOR_T,ROW_VECTOR_T);
add("elt_multiply",MATRIX_T,MATRIX_T,MATRIX_T);

add("elt_divide",VECTOR_T,VECTOR_T,VECTOR_T);
add("elt_divide",ROW_VECTOR_T,ROW_VECTOR_T,ROW_VECTOR_T);
add("elt_divide",MATRIX_T,MATRIX_T,MATRIX_T);

add("row",ROW_VECTOR_T,MATRIX_T,INT_T);
add("col",VECTOR_T,MATRIX_T,INT_T);
add("diagonal",VECTOR_T,MATRIX_T);
add("diag_matrix",MATRIX_T,VECTOR_T);

add("transpose",ROW_VECTOR_T,VECTOR_T);
add("tranpose",VECTOR_T,ROW_VECTOR_T);
add("transpose",MATRIX_T,MATRIX_T);

add("inverse",MATRIX_T,MATRIX_T);

add("eigenvalues",VECTOR_T,MATRIX_T);
add("eigenvectors",MATRIX_T,MATRIX_T);

add("eigenvalues_sym",VECTOR_T,MATRIX_T);
add("eigenvectors_sym",MATRIX_T,MATRIX_T);

add("cholesky_decompose",MATRIX_T,MATRIX_T);

add("singular_values",VECTOR_T,MATRIX_T);

// eigen_decompose, eigen_decompose_sym, svd return void
// so no calling in Stan GM

add_unary("acosh");
add_unary("asinh");
add_unary("atanh");

add_unary("erf");
add_unary("erfc");

add_unary("exp2");
add_unary("expm1");

add_binary("lbeta");
add_unary("lgamma");

add_unary("log1p");
add_unary("log1m");

add_ternary("fma");

add_binary("fmax");
add_binary("fmin");

add_binary("hypot");

add_unary("log2");
add_unary("cbrt");

add_unary("round");

add_unary("trunc");

add_binary("fdim");

add_unary("tgamma");

add("int_step",INT_T,DOUBLE_T);

add_unary("step");

add_unary("inv_cloglog");

add_unary("Phi");

add_unary("logit");
add_unary("inv_logit");

add_binary("log_loss");

add("log_sum_exp",DOUBLE_T, expr_type(DOUBLE_T,1U));
add_binary("log_sum_exp");

add_unary("square");

add("bernoulli_log",DOUBLE_T,INT_T,DOUBLE_T);
add_ternary("beta_log");
add("beta_binomial_log",DOUBLE_T,INT_T,INT_T,DOUBLE_T,DOUBLE_T);
add("binomial_log",DOUBLE_T,INT_T,INT_T,DOUBLE_T);
add("categorical_log",DOUBLE_T,INT_T,VECTOR_T);
add_ternary("cauchy_log");
add_binary("chi_square_log");
add("dirichlet_log",DOUBLE_T,VECTOR_T,VECTOR_T);
add_ternary("double_exponential_log");
add_binary("exponential_log");
add_ternary("gamma_log");
add("hypergeometric_log",DOUBLE_T, INT_T, INT_T, INT_T, INT_T);
add_binary("inv_chi_square_log");
add_ternary("inv_gamma_log");
add("inv_wishart_log",DOUBLE_T, MATRIX_T,DOUBLE_T,MATRIX_T);
add("lkj_corr_log",DOUBLE_T, MATRIX_T,DOUBLE_T);
add("lkj_cov_log",DOUBLE_T, MATRIX_T,MATRIX_T,MATRIX_T,DOUBLE_T);
add_ternary("logistic_log");
add_ternary("lognormal_log");
add("multi_normal_log",DOUBLE_T, VECTOR_T,VECTOR_T,MATRIX_T);
add("multi_student_t_log",DOUBLE_T, DOUBLE_T,VECTOR_T,VECTOR_T,MATRIX_T);
add("multinomial_log",DOUBLE_T, expr_type(INT_T,1U), VECTOR_T);
add("neg_binomial_log",DOUBLE_T, INT_T,DOUBLE_T,DOUBLE_T);
add_ternary("normal_log");
add_quaternary("normal_trunc_l_log");
add("normal_log",DOUBLE_T,expr_type(DOUBLE_T,1U),DOUBLE_T,DOUBLE_T); // vectorized
add_ternary("pareto_log");
add("poisson_log",DOUBLE_T, INT_T,DOUBLE_T);
add_ternary("scaled_inv_chi_square_log");
add_quaternary("student_t_log");
add_ternary("uniform_log");
add_ternary("weibull_log");
add("wishart_log",DOUBLE_T, MATRIX_T,DOUBLE_T,MATRIX_T);

add_ternary("weibull_p");


// MULTINOMIAL?  no vector<int> type

// CONSTANTS

add_nullary("e");
add_nullary("pi");
add_nullary("sqrt2");
add_nullary("log2");
add_nullary("log10");
add_nullary("infinity");
add_nullary("negative_infinity");
add_nullary("nan");
add_nullary("epsilon");
add_nullary("negative_epsilon");





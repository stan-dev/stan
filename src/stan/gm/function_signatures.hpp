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

add("min",DOUBLE_T,VECTOR_T);
add("min",DOUBLE_T,ROW_VECTOR_T);
add("min",DOUBLE_T,MATRIX_T);

add("max",DOUBLE_T,VECTOR_T);
add("max",DOUBLE_T,ROW_VECTOR_T);
add("max",DOUBLE_T,MATRIX_T);

add("mean",DOUBLE_T,VECTOR_T);
add("mean",DOUBLE_T,ROW_VECTOR_T);
add("mean",DOUBLE_T,MATRIX_T);

add("variance",DOUBLE_T,VECTOR_T);
add("variance",DOUBLE_T,ROW_VECTOR_T);
add("variance",DOUBLE_T,MATRIX_T);

add("sd",DOUBLE_T,VECTOR_T);
add("sd",DOUBLE_T,ROW_VECTOR_T);
add("sd",DOUBLE_T,MATRIX_T);

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

add_unary("step");

add_unary("inv_cloglog");

add_unary("Phi");

add_unary("inv_logit");

add_binary("log_loss");

add_binary("log_sum_exp");
add("log_sum_exp", expr_type(DOUBLE_T,1U));

add_unary("square");


	   







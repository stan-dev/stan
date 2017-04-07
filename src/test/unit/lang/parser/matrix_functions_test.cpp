#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>


TEST(lang_parser, csr_extract_w_function_signatures) {
  test_parsable("function-signatures/math/matrix/csr_extract_w");
}
TEST(lang_parser, csr_extract_v_function_signatures) {
  test_parsable("function-signatures/math/matrix/csr_extract_v");
}
TEST(lang_parser, csr_extract_u_function_signatures) {
  test_parsable("function-signatures/math/matrix/csr_extract_u");
}
TEST(lang_parser, csr_matrix_times_vector_function_signatures) {
  test_parsable("function-signatures/math/matrix/csr_matrix_times_vector");
}
TEST(lang_parser, csr_to_dense_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/csr_to_dense_matrix");
}


TEST(lang_parser, append_col_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/append_col");
}

TEST(lang_parser, append_row_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/append_row");
}

TEST(lang_parser, block_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/block");
}

TEST(lang_parser, broadcast_infix_operators_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/broadcast_infix_operators");
}

TEST(lang_parser, cholesky_decompose_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/cholesky_decompose");
}

TEST(lang_parser, col_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/col");
}

TEST(lang_parser, cols_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/cols");
}

TEST(lang_parser, columns_dot_product_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/columns_dot_product");
}

TEST(lang_parser, columns_dot_self_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/columns_dot_self");
}

TEST(lang_parser, cov_exp_quad_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/cov_exp_quad");
}

TEST(lang_parser, crossprod_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/crossprod");
}

TEST(lang_parser, cumulative_sum_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/cumulative_sum");
}

TEST(lang_parser, determinant_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/determinant");
}

TEST(lang_parser, diag_matrix_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/diag_matrix");
}

TEST(lang_parser, diag_post_multiply_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/diag_post_multiply");
}

TEST(lang_parser, diag_pre_multiply_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/diag_pre_multiply");
}

TEST(lang_parser, diagonal_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/diagonal");
}

TEST(lang_parser, dims_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/dims");
}

TEST(lang_parser, distance_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/distance");
}

TEST(lang_parser, dot_product_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/dot_product");
}

TEST(lang_parser, dot_self_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/dot_self");
}

TEST(lang_parser, eigenvalues_sym_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/eigenvalues_sym");
}

TEST(lang_parser, eigenvectors_sym_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/eigenvectors_sym");
}

TEST(lang_parser, elementwise_products_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/elementwise_products");
}

TEST(lang_parser, exp_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/exp");
}

TEST(lang_parser, head_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/head");
}

TEST(lang_parser, infix_matrix_operators_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/infix_matrix_operators");
}

TEST(lang_parser, inverse_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/inverse");
}

TEST(lang_parser, inverse_spd_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/inverse_spd");
}

TEST(lang_parser, log_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/log");
}

TEST(lang_parser, log_determinant_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/log_determinant");
}

TEST(lang_parser, log_softmax_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/log_softmax");
}

TEST(lang_parser, log_sum_exp_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/log_sum_exp");
}

TEST(lang_parser, division_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/matrix_division");
}

TEST(lang_parser, matrix_exp_matrix_function_signatures) {
    test_parsable("function-signatures/math/matrix/matrix_exp");
}

TEST(lang_parser, max_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/max");
}

TEST(lang_parser, mdivide_left_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/mdivide_left");
}

TEST(lang_parser, mdivide_left_tri_low_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/mdivide_left_tri_low");
}

TEST(lang_parser, mdivide_right_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/mdivide_right");
}

TEST(lang_parser, mdivide_right_tri_low_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/mdivide_right_tri_low");
}

TEST(lang_parser, mean_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/mean");
}

TEST(lang_parser, min_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/min");
}

TEST(lang_parser, multiply_lower_tri_self_transpose_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/multiply_lower_tri_self_transpose");
}

TEST(lang_parser, negation_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/negation");
}

TEST(lang_parser, prod_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/prod");
}

TEST(lang_parser, qr_Q_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/qr_Q");
}

TEST(lang_parser, qr_R_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/qr_R");
}

TEST(lang_parser, quad_form_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/quad_form");
}

TEST(lang_parser, quad_form_diag_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/quad_form_diag");
}

TEST(lang_parser, quad_form_sym_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/quad_form_sym");
}

TEST(lang_parser, rank_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/rep_matrix");
}

TEST(lang_parser, rep_matrix_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/rank");
}

TEST(lang_parser, rep_param_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/rep_param"); //mostly rep_array with some other rep_ tests
}

TEST(lang_parser, rep_row_vector_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/rep_row_vector");
}

TEST(lang_parser, rep_vector_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/rep_vector");
}

TEST(lang_parser, row_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/row");
}

TEST(lang_parser, rows_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/rows");
}

TEST(lang_parser, rows_dot_product_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/rows_dot_product");
}

TEST(lang_parser, rows_dot_self_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/rows_dot_self");
}

TEST(lang_parser, singular_values_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/singular_values");
}

TEST(lang_parser, segment_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/segment");
}

TEST(lang_parser, size_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/size");
}

TEST(lang_parser, sd_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/sd");
}

TEST(lang_parser, softmax_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/softmax");
}

TEST(lang_parser, sort_asc_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/sort_asc");
}

TEST(lang_parser, sort_desc_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/sort_desc");
}

TEST(lang_parser, sort_indices_asc_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/sort_indices_asc");
}

TEST(lang_parser, sort_indices_desc_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/sort_indices_desc");
}

TEST(lang_parser, squared_distance_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/squared_distance");
}

TEST(lang_parser, sub_col_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/sub_col");
}

TEST(lang_parser, sub_row_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/sub_row");
}

TEST(lang_parser, sum_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/sum");
}

TEST(lang_parser, tail_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/tail");
}

TEST(lang_parser, tcrossprod_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/tcrossprod");
}

TEST(lang_parser, to_array_1d_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/to_array_1d");
}

TEST(lang_parser, to_array_2d_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/to_array_2d");
}

TEST(lang_parser, to_matrix_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/to_matrix");
}

TEST(lang_parser, to_row_vector_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/to_row_vector");
}

TEST(lang_parser, to_vector_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/to_vector");
}

TEST(lang_parser, trace_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/trace");
}

TEST(lang_parser, trace_gen_quad_form_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/trace_gen_quad_form");
}

TEST(lang_parser, trace_quad_form_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/trace_quad_form");
}

TEST(lang_parser, transpose_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/transpose");
}

TEST(lang_parser, variance_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/variance");
}

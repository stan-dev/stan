#include <gtest/gtest.h>
#include <test/unit/gm/utility.hpp>

TEST(gm_parser, append_col_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/append_col");
}

TEST(gm_parser, append_row_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/append_row");
}

TEST(gm_parser, block_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/block");
}

TEST(gm_parser, broadcast_infix_operators_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/broadcast_infix_operators");
}

TEST(gm_parser, cholesky_decompose_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/cholesky_decompose");
}

TEST(gm_parser, col_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/col");
}

TEST(gm_parser, cols_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/cols");
}

TEST(gm_parser, columns_dot_product_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/columns_dot_product");
}

TEST(gm_parser, columns_dot_self_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/columns_dot_self");
}

TEST(gm_parser, crossprod_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/crossprod");
}

TEST(gm_parser, cumulative_sum_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/cumulative_sum");
}

TEST(gm_parser, determinant_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/determinant");
}

TEST(gm_parser, diag_matrix_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/diag_matrix");
}

TEST(gm_parser, diag_post_multiply_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/diag_post_multiply");
}

TEST(gm_parser, diag_pre_multiply_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/diag_pre_multiply");
}

TEST(gm_parser, diagonal_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/diagonal");
}

TEST(gm_parser, dims_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/dims");
}

TEST(gm_parser, distance_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/distance");
}

TEST(gm_parser, dot_product_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/dot_product");
}

TEST(gm_parser, dot_self_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/dot_self");
}

TEST(gm_parser, eigenvalues_sym_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/eigenvalues_sym");
}

TEST(gm_parser, eigenvectors_sym_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/eigenvectors_sym");
}

TEST(gm_parser, elementwise_products_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/elementwise_products");
}

TEST(gm_parser, exp_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/exp");
}

TEST(gm_parser, head_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/head");
}

TEST(gm_parser, infix_matrix_operators_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/infix_matrix_operators");
}

TEST(gm_parser, inverse_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/inverse");
}

TEST(gm_parser, inverse_spd_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/inverse_spd");
}

TEST(gm_parser, log_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/log");
}

TEST(gm_parser, log_determinant_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/log_determinant");
}

TEST(gm_parser, log_softmax_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/log_softmax");
}

TEST(gm_parser, log_sum_exp_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/log_sum_exp");
}

TEST(gm_parser, division_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/matrix_division");
}

TEST(gm_parser, max_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/max");
}

TEST(gm_parser, mdivide_left_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/mdivide_left");
}

TEST(gm_parser, mdivide_left_tri_low_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/mdivide_left_tri_low");
}

TEST(gm_parser, mdivide_right_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/mdivide_right");
}

TEST(gm_parser, mdivide_right_tri_low_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/mdivide_right_tri_low");
}

TEST(gm_parser, mean_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/mean");
}

TEST(gm_parser, min_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/min");
}

TEST(gm_parser, multiply_lower_tri_self_transpose_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/multiply_lower_tri_self_transpose");
}

TEST(gm_parser, negation_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/negation");
}

TEST(gm_parser, prod_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/prod");
}

TEST(gm_parser, qr_Q_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/qr_Q");
}

TEST(gm_parser, qr_R_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/qr_R");
}

TEST(gm_parser, quad_form_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/quad_form");
}

TEST(gm_parser, quad_form_diag_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/quad_form_diag");
}

TEST(gm_parser, quad_form_sym_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/quad_form_sym");
}

TEST(gm_parser, rank_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/rep_matrix");
}

TEST(gm_parser, rep_matrix_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/rank");
}

TEST(gm_parser, rep_param_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/rep_param"); //mostly rep_array with some other rep_ tests
}

TEST(gm_parser, rep_row_vector_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/rep_row_vector");
}

TEST(gm_parser, rep_vector_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/rep_vector");
}

TEST(gm_parser, row_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/row");
}

TEST(gm_parser, rows_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/rows");
}

TEST(gm_parser, rows_dot_product_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/rows_dot_product");
}

TEST(gm_parser, rows_dot_self_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/rows_dot_self");
}

TEST(gm_parser, singular_values_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/singular_values");
}

TEST(gm_parser, segment_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/segment");
}

TEST(gm_parser, size_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/size");
}

TEST(gm_parser, sd_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/sd");
}

TEST(gm_parser, softmax_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/softmax");
}

TEST(gm_parser, sort_asc_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/sort_asc");
}

TEST(gm_parser, sort_desc_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/sort_desc");
}

TEST(gm_parser, sort_indices_asc_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/sort_indices_asc");
}

TEST(gm_parser, sort_indices_desc_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/sort_indices_desc");
}

TEST(gm_parser, squared_distance_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/squared_distance");
}

TEST(gm_parser, sub_col_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/sub_col");
}

TEST(gm_parser, sub_row_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/sub_row");
}

TEST(gm_parser, sum_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/sum");
}

TEST(gm_parser, tail_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/tail");
}

TEST(gm_parser, tcrossprod_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/tcrossprod");
}

TEST(gm_parser, to_array_1d_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/to_array_1d");
}

TEST(gm_parser, to_array_2d_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/to_array_2d");
}

TEST(gm_parser, to_matrix_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/to_matrix");
}

TEST(gm_parser, to_row_vector_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/to_row_vector");
}

TEST(gm_parser, to_vector_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/to_vector");
}

TEST(gm_parser, trace_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/trace");
}

TEST(gm_parser, trace_gen_quad_form_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/trace_gen_quad_form");
}

TEST(gm_parser, trace_quad_form_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/trace_quad_form");
}

TEST(gm_parser, transpose_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/transpose");
}

TEST(gm_parser, variance_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/variance");
}



#include <test/agrad/fwd/matrix/to_fvar_test.cpp>  

//INTEGER-VALUED MATRIX SIZE FUNCTIONS -- DONE
#include <test/agrad/fwd/matrix/rows_test.cpp>       
#include <test/agrad/fwd/matrix/cols_test.cpp>       

//MATRIX ARITHMETIC OPEPRATORS -- DONE
//Negation Prefix Operators -- DONE
#include <test/agrad/fwd/matrix/minus_test.cpp>       

//Infix Matrix Operators -- DONE
#include <test/agrad/fwd/matrix/operator_addition_test.cpp>       
#include <test/agrad/fwd/matrix/operator_subtraction_test.cpp>       
#include <test/agrad/fwd/matrix/operator_multiplication_test.cpp>       
#include <test/agrad/fwd/matrix/operator_division_test.cpp>       

//Broadcast Infix Operators -- DONE
  //Operator+ is in <test/agrad/fwd/matrix/operator_addition_test.cpp>
  //Operator- is in <test/agrad/fwd/matrix/operator_subtraction_test.cpp>

//Elementwise Products -- DONE
#include <test/agrad/fwd/matrix/elt_multiply_test.cpp>       
#include <test/agrad/fwd/matrix/elt_divide_test.cpp>       

//Elementwise Logarithms -- DONE
#include <test/agrad/fwd/matrix/log_test.cpp>       
#include <test/agrad/fwd/matrix/exp_test.cpp>       

//Cumulative Sums -- DONE
#include <test/agrad/fwd/matrix/cumulative_sum_test.cpp>       

//Dot Products -- DONE
#include <test/agrad/fwd/matrix/dot_product_test.cpp>       
#include <test/agrad/fwd/matrix/columns_dot_product_test.cpp>       
#include <test/agrad/fwd/matrix/rows_dot_product_test.cpp>       
#include <test/agrad/fwd/matrix/dot_self_test.cpp>     
#include <test/agrad/fwd/matrix/rows_dot_self_test.cpp>     
#include <test/agrad/fwd/matrix/columns_dot_self_test.cpp>     

//Specialized Products -- DONE
#include <test/agrad/fwd/matrix/tcrossprod_test.cpp>       
#include <test/agrad/fwd/matrix/crossprod_test.cpp>       
#include <test/agrad/fwd/matrix/multiply_lower_tri_self_transpose_test.cpp>       
#include <test/agrad/fwd/matrix/diag_pre_multiply_test.cpp>       
#include <test/agrad/fwd/matrix/diag_post_multiply_test.cpp>       

//REDUCTIONS -- DONE
//Minimum and Maximum -- DONE
#include <test/agrad/fwd/matrix/min_test.cpp>     
#include <test/agrad/fwd/matrix/max_test.cpp>     

//Sums and Products -- DONE
#include <test/agrad/fwd/matrix/sum_test.cpp>     
#include <test/agrad/fwd/matrix/prod_test.cpp>     

//Sample Moments -- DONE
#include <test/agrad/fwd/matrix/mean_test.cpp>     
#include <test/agrad/fwd/matrix/variance_test.cpp>     
#include <test/agrad/fwd/matrix/sd_test.cpp>     

//BROADCAST FUNCTIONS -- DONE
#include <test/agrad/fwd/matrix/rep_vector_test.cpp>     
#include <test/agrad/fwd/matrix/rep_row_vector_test.cpp>     
#include <test/agrad/fwd/matrix/rep_matrix_test.cpp>     

//SLICE AND PACKAGE FUNCTIONS -- DONE
//Diagonal Matrices -- DONE
#include <test/agrad/fwd/matrix/diagonal_test.cpp>     
#include <test/agrad/fwd/matrix/diag_matrix_test.cpp>     
#include <test/agrad/fwd/matrix/col_test.cpp>     
#include <test/agrad/fwd/matrix/row_test.cpp>     

//Block Operation -- DONE
#include <test/agrad/fwd/matrix/block_test.cpp>     

//Transposition Postfix Operator -- DONE
#include <test/agrad/fwd/matrix/transpose_test.cpp>     

//SPECIAL MATRIX FUNCTIONS -- DONE
#include <test/agrad/fwd/matrix/softmax_test.cpp>     

//LINEAR ALGEBRA FUNCTIONS AND SOLVERS
//Matrix Division Infix Operators -- DONE
#include <test/agrad/fwd/matrix/mdivide_right_test.cpp>     
#include <test/agrad/fwd/matrix/mdivide_left_test.cpp>   

//Lower-Triangular Matrix-Division Functions -- DONE  
#include <test/agrad/fwd/matrix/mdivide_right_tri_low_test.cpp>     
#include <test/agrad/fwd/matrix/mdivide_left_tri_low_test.cpp>     

//Linear Algebra Functions
#include <test/agrad/fwd/matrix/trace_test.cpp>     
// #include <test/agrad/fwd/matrix/determinant_test.cpp>     
#include <test/agrad/fwd/matrix/log_determinant_test.cpp>     
 //#include <test/agrad/fwd/matrix/inverse_test.cpp> breaks when included

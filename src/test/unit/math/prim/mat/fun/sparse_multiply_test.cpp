#include <stan/math/prim/mat/fun/assign.hpp>
#include <stan/math/prim/mat/fun/sparse_extractors.hpp>
#include <stan/math/prim/mat/fun/dot_product.hpp>
#include <stan/math/prim/mat/fun/sparse_multiply.hpp>
#include <stan/math/prim/mat/fun/multiply.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <iostream>

TEST(SparseStuff,sparse_multiply_csc) {
  stan::math::matrix_d m(2,3);
	Eigen::SparseMatrix<double> a;
  m << 2.0, 4.0, 6.0, 8.0, 10.0, 12.0;

	a = m.sparseView();
  stan::math::vector_d X_w = stan::math::extract_w(a);
	std::vector<int> X_v = stan::math::extract_v(a);
  std::vector<int> X_u = stan::math::extract_u(a);
  std::vector<int> X_z = stan::math::extract_z(a);

	stan::math::vector_d b(3);
	b << 22, 33, 44;

//	std::cout << X_w << std::endl;
//	for (unsigned int i=0; i < X_v.size(); ++i) 	
//		std::cout << X_v[i] << ", ";
//	std::cout << ";" <<  std::endl;
//	for (unsigned int i=0; i < X_u.size(); ++i) 	
//		std::cout << X_u[i] << ", ";
//	std::cout << ";" <<  std::endl;
//	for (unsigned int i=0; i < X_z.size(); ++i) 	
//		std::cout << X_z[i] << ", ";
//	std::cout << ";" <<  std::endl;

	stan::math::vector_d result = stan::math::sparse_multiply_csc(
			2, 3, X_w, X_v, X_u, X_z, b);
//	stan::math::vector_d result = stan::math::multiply(m,b);

  EXPECT_FLOAT_EQ( 440.0,result(0));
  EXPECT_FLOAT_EQ(1034.0,result(1));
}

TEST(SparseStuff,sparse_multiply_csr) {
  stan::math::matrix_d m(2,3);
	Eigen::SparseMatrix<double,Eigen::RowMajor> a;
  m << 2.0, 4.0, 6.0, 8.0, 10.0, 12.0;

	a = m.sparseView();
  stan::math::vector_d X_w = stan::math::extract_w(a);
	std::vector<int> X_v = stan::math::extract_v(a);
  std::vector<int> X_u = stan::math::extract_u(a);
  std::vector<int> X_z = stan::math::extract_z(a);

	stan::math::vector_d b(3);
	b << 22, 33, 44;

//	std::cout << X_w << std::endl;
//	for (unsigned int i=0; i < X_v.size(); ++i) 	
//		std::cout << X_v[i] << ", ";
//	std::cout << ";" <<  std::endl;
//	for (unsigned int i=0; i < X_u.size(); ++i) 	
//		std::cout << X_u[i] << ", ";
//	std::cout << ";" <<  std::endl;
//	for (unsigned int i=0; i < X_z.size(); ++i) 	
//		std::cout << X_z[i] << ", ";
//	std::cout << ";" <<  std::endl;

	stan::math::vector_d result = stan::math::sparse_multiply_csr(
			2, 3, X_w, X_v, X_u, X_z, b);
//	stan::math::vector_d result = stan::math::multiply(m,b);

  EXPECT_FLOAT_EQ( 440.0,result(0));
  EXPECT_FLOAT_EQ(1034.0,result(1));
}

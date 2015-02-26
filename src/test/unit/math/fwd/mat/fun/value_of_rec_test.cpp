#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/value_of_rec.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>

TEST(AgradMatrix,value_of_rec) {
  using stan::agrad::fvar;
  using stan::math::value_of_rec;
  using std::vector;

  vector<double> a_vals;

  for (size_t i = 0; i < 10; ++i)
    a_vals.push_back(i + 1);

  vector<double> b_vals;

  for (size_t i = 10; i < 15; ++i)
    b_vals.push_back(i + 1);
  
  Eigen::Matrix<double,2,5> a; 
  fill(a_vals, a);
  Eigen::Matrix<double,5,1> b;
  fill(b_vals, b);

  Eigen::Matrix<fvar<double>,2,5> fd_a;
  fill(a_vals, fd_a);
  Eigen::Matrix<fvar<double>,5,1> fd_b;
  fill(b_vals, fd_b);

  Eigen::Matrix<fvar<fvar<double> >,2,5> ffd_a;
  fill(a_vals, ffd_a);
  Eigen::Matrix<fvar<fvar<double> >,5,1> ffd_b;
  fill(b_vals, ffd_b);

  Eigen::MatrixXd d_fd_a = value_of_rec(fd_a);
  Eigen::MatrixXd d_fd_b = value_of_rec(fd_b);
  Eigen::MatrixXd d_ffd_a = value_of_rec(ffd_a);
  Eigen::MatrixXd d_ffd_b = value_of_rec(ffd_b);

  for (size_type i = 0; i < 5; ++i){
    EXPECT_FLOAT_EQ(b(i), d_fd_b(i));
    EXPECT_FLOAT_EQ(b(i), d_ffd_b(i));
  }

  for (size_type i = 0; i < 2; ++i)
    for (size_type j = 0; j < 5; ++j){
      EXPECT_FLOAT_EQ(a(i,j), d_fd_a(i,j));
      EXPECT_FLOAT_EQ(a(i,j), d_ffd_a(i,j));
    }
}

#include <iostream>
#include <vector>
#include <stan/maths/matrix.hpp>
#include <stan/agrad/matrix.hpp>
#include <stan/agrad/special_functions.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;
using stan::agrad::var;
using stan::maths::subtract;
using boost::math::tools::promote_args;


var 
multivar_norm(Matrix<var,Dynamic,1>& y,
              Matrix<var,Dynamic,1>& mu,
              Matrix<var,Dynamic,Dynamic>& L) {

  var lp(0.0);

  // if #1
  lp += 2.0 * y.rows();
  
  // if #2
  for (unsigned int m = 0; m < L.rows(); ++m)
    lp += log(L(m,m));
  
  // if #3
  Eigen::Matrix<var,Dynamic,1> diff 
    = subtract(y,mu);

  Eigen::Matrix<var,Dynamic,Dynamic> L_inv
    = L.triangularView<Eigen::Lower>().solve(Matrix<var,Dynamic,Dynamic>::Identity(L.rows(),L.rows()));

  std::cout << "L_inv=" << L_inv << std::endl;

  Eigen::Matrix<var,Dynamic,1> half 
    = multiply(L_inv,diff);

  lp -= 0.5 * half.dot(half);

  return lp;

}

int main() {

  Eigen::Matrix<var,Dynamic,1> y(3);
  y << -1, -2, -3;

  Eigen::Matrix<var,Dynamic,1> mu(3);
  mu << 0.0, -1.0, 1.0;

  // only care about lower-part of argument A, but create it all
  Eigen::Matrix<var,Dynamic,Dynamic> A(3,3);
  A << 1, 2, 3,
       2, 4, 8, 
       5, 20, 50;

  // L now is just the lower-triangular part of A
  Eigen::Matrix<var,Dynamic,Dynamic> L
    = A.triangularView<Eigen::Lower>();

  var lp = multivar_norm(y,mu,L);
  std::cout << "lp=" << lp << std::endl;
}


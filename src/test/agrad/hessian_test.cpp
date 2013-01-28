#include <gtest/gtest.h>

#include <boost/math/special_functions/fpclassify.hpp>

#include <stan/agrad/hessian.hpp>

struct fun1 {
  template <typename T>
  inline
  T operator()(const std::vector<T>& x) const {
    return x[0] * x[0] * x[1] 
      + 3.0 * x[1] * x[1]; 
  }
};

TEST(AgradHessian,one) {
  using std::vector;
  using stan::agrad::hessian;

  fun1 f;
  
  vector<double> x(2);
  x[0] = 2;
  x[1] = -3;
  
  vector<double> v(2);
  v[0] = 8;
  v[1] = 5;
  
  vector<double> Hv;
  double y = hessian(f,x,v,Hv);

  double expected_Hv0 = 2 * x[1] * v[0] + 2 * x[0] * v[1];
  double expected_Hv1 = 2 * x[0] * v[0] + 6 * v[1];

  // std::cout << "expected Hv0=" << expected_Hv0
  // << "; Hv1=" << expected_Hv1 << std::endl;

  EXPECT_FLOAT_EQ(expected_Hv0, Hv[0]);
  EXPECT_FLOAT_EQ(expected_Hv1, Hv[1]);
}



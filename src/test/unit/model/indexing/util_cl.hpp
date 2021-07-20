#ifdef STAN_OPENCL
#include <stan/math.hpp>
#include <stan/model/indexing/rvalue.hpp>

/**
 * Convert an index to a type usable with OpenCL overloads.
 * @param i index
 * @return OpenCL index
 */
template <typename T>
T opencl_index(T i) {
  return i;
}
stan::math::matrix_cl<int> opencl_index(const stan::model::index_multi& i) {
  return stan::math::to_matrix_cl(i.ns_);
}

/**
 * Set adjoint using pattern 1.
 * @param[in,out] var or matrix of vars or var<matrix_cl>.
 */
template <typename T, stan::require_not_rev_kernel_expression_t<T>* = nullptr>
void set_adjoints1(T& v) {
  for (int i = 0; i < v.rows(); i++) {
    for (int j = 0; j < v.cols(); j++) {
      v(i, j).adj() += i + 10 * j + 100;
    }
  }
}
template <typename T, stan::require_rev_kernel_expression_t<T>* = nullptr>
void set_adjoints1(T& v) {
  Eigen::MatrixXd adj(v.rows(), v.cols());

  for (int i = 0; i < v.rows(); i++) {
    for (int j = 0; j < v.cols(); j++) {
      adj(i, j) = i + 10 * j + 100;
    }
  }
  stan::math::matrix_cl<double> adj_cl(adj);
  v.adj() += adj_cl;
}
void set_adjoints1(stan::math::var v) { v.adj() = 3; }

/**
 * Set adjoint using pattern 2.
 * @param[in,out] matrix of vars or var<matrix_cl>.
 */
template <typename T, stan::require_not_rev_kernel_expression_t<T>* = nullptr>
void set_adjoints2(T& v) {
  for (int i = 0; i < v.rows(); i++) {
    for (int j = 0; j < v.cols(); j++) {
      v(i, j).adj() += i + 10 * j + 10000;
    }
  }
}
template <typename T, stan::require_rev_kernel_expression_t<T>* = nullptr>
void set_adjoints2(T& v) {
  Eigen::MatrixXd adj(v.rows(), v.cols());

  for (int i = 0; i < v.rows(); i++) {
    for (int j = 0; j < v.cols(); j++) {
      adj(i, j) = i + 10 * j + 10000;
    }
  }
  stan::math::matrix_cl<double> adj_cl(adj);
  v.adj() += adj_cl;
}

/**
 * Return scalar argument or call `from_matrix_cl` on non-scalars.
 * @param a argument
 */
template <typename T>
auto from_matrix_cl_nonscalar(const T& a) {
  return stan::math::from_matrix_cl(a);
}
auto from_matrix_cl_nonscalar(const stan::math::var a) { return a; }
auto from_matrix_cl_nonscalar(const double a) { return a; }
auto from_matrix_cl_nonscalar(const int a) { return a; }

#endif

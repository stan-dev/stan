#ifndef STAN_MCMC_HMC_HAMILTONIANS_PS_POINT_HPP
#define STAN_MCMC_HMC_HAMILTONIANS_PS_POINT_HPP

#include <stan/callbacks/writer.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <boost/lexical_cast.hpp>
#include <string>
#include <vector>

namespace stan {
namespace mcmc {

/**
 * Equivalent to `Eigen::Matrix`, except that the data is stored on AD stack.
 * That makes these objects triviali destructible and usable in `vari`s.
 *
 * @tparam MatrixType Eigen matrix type this works as (`MatrixXd`, `VectorXd`
 * ...)
 */
template <typename MatrixType>
class stack_matrix : public Eigen::Map<MatrixType> {
 public:
  using Scalar = value_type_t<MatrixType>;
  using Base = Eigen::Map<MatrixType>;
  using PlainObject = std::decay_t<MatrixType>;
  static constexpr int RowsAtCompileTime = MatrixType::RowsAtCompileTime;
  static constexpr int ColsAtCompileTime = MatrixType::ColsAtCompileTime;
  stan::math::stack_alloc* stack_{nullptr};
  /**
   * Default constructor.
   */
  stack_matrix()
      : Base::Map(nullptr,
                  RowsAtCompileTime == Eigen::Dynamic ? 0 : RowsAtCompileTime,
                  ColsAtCompileTime == Eigen::Dynamic ? 0 : ColsAtCompileTime) {
  }

  /**
   * Constructs `stack_matrix` with given number of rows and columns.
   * @param rows number of rows
   * @param cols number of columns
   */
  stack_matrix(Eigen::Index rows, Eigen::Index cols, stan::math::stack_alloc* stack)
      : Base::Map(
          stack->alloc_array<Scalar>(rows * cols),
          rows, cols), stack_(stack) {}

  /**
   * Constructs `stack_matrix` with given size. This only works if
   * `MatrixType` is row or col vector.
   * @param size number of elements
   */
  explicit stack_matrix(Eigen::Index size, stan::math::stack_alloc* stack)
      : Base::Map(
          stack->alloc_array<Scalar>(size),
          size), stack_(stack) {}

  /**
   * Constructs `stack_matrix` from an expression.
   * @param other expression
   */
  template <typename T, require_eigen_t<T>* = nullptr>
  stack_matrix(const T& other, stan::math::stack_alloc* stack)  // NOLINT
      : Base::Map(
          stack->alloc_array<Scalar>(
              other.size()),
          (RowsAtCompileTime == 1 && T::ColsAtCompileTime == 1)
                  || (ColsAtCompileTime == 1 && T::RowsAtCompileTime == 1)
              ? other.cols()
              : other.rows(),
          (RowsAtCompileTime == 1 && T::ColsAtCompileTime == 1)
                  || (ColsAtCompileTime == 1 && T::RowsAtCompileTime == 1)
              ? other.rows()
              : other.cols()),
              stack_(stack) {
    *this = other;
  }

  /**
   * Constructs `stack_matrix` from an expression. This makes an assumption that
   * any other `Eigen::Map` also contains memory allocated in the arena.
   * @param other expression
   */
  stack_matrix(const Base& other, stan::math::stack_alloc* stack)  // NOLINT
      : Base::Map(other), stack_(stack) {}

  /**
   * Copy constructor.
   * @param other matrix to copy from
   */
  stack_matrix(const stack_matrix<MatrixType>& other, stan::math::stack_alloc* stack)
      : Base::Map(const_cast<Scalar*>(other.data()), other.rows(),
                  other.cols()), stack_(stack) {}

  // without this using, compiler prefers combination of implicit construction
  // and copy assignment to the inherited operator when assigned an expression
  using Base::operator=;

  /**
   * Copy assignment operator.
   * @param other matrix to copy from
   * @return `*this`
   */
  stack_matrix& operator=(const stack_matrix<MatrixType>& other) {
    // placement new changes what data map points to - there is no allocation
    new (this)
        Base(const_cast<Scalar*>(other.data()), other.rows(), other.cols());
    return *this;
  }

  /**
   * Assignment operator for assigning an expression.
   * @param a expression to evaluate into this
   * @return `*this`
   */
  template <typename T>
  stack_matrix& operator=(const T& a) {
    if (stack_ == nullptr) {
      throw std::domain_error("NO!");
    }
    // do we need to transpose?
    if ((RowsAtCompileTime == 1 && T::ColsAtCompileTime == 1)
        || (ColsAtCompileTime == 1 && T::RowsAtCompileTime == 1)) {
      // placement new changes what data map points to - there is no allocation
      new (this) Base(
          stack_->alloc_array<Scalar>(a.size()),
          a.cols(), a.rows());

    } else {
      new (this) Base(
          stack_->alloc_array<Scalar>(a.size()),
          a.rows(), a.cols());
    }
    Base::operator=(a);
    return *this;
  }
};

}  // namespace math
}  // namespace stan

namespace Eigen {
namespace internal {

template <typename T>
struct traits<stan::mcmc::stack_matrix<T>> {
  using base = traits<Eigen::Map<T>>;
  using XprKind = typename Eigen::internal::traits<std::decay_t<T>>::XprKind;
  enum {
    PlainObjectTypeInnerSize = base::PlainObjectTypeInnerSize,
    InnerStrideAtCompileTime = base::InnerStrideAtCompileTime,
    OuterStrideAtCompileTime = base::OuterStrideAtCompileTime,
    Alignment = base::Alignment,
    Flags = base::Flags
  };
};

}  // namespace internal
}  // namespace Eigen


namespace stan {
namespace mcmc {
using Eigen::Dynamic;
class ps_point_map;
/**
 * Point in a generic phase space
 */
class ps_point {
 public:
  explicit ps_point(int n) : q(n), p(n), g(n) {}
  explicit ps_point(ps_point_map& ps);
  explicit ps_point(ps_point_map&& ps);
  ps_point& operator=(ps_point_map& ps);
  ps_point& operator=(ps_point_map&& ps);

  Eigen::VectorXd q;
  Eigen::VectorXd p;
  Eigen::VectorXd g;
  double V{0};

  virtual inline void get_param_names(std::vector<std::string>& model_names,
                                      std::vector<std::string>& names) {
    names.reserve(q.size() + p.size() + g.size());
    for (int i = 0; i < q.size(); ++i)
      names.emplace_back(model_names[i]);
    for (int i = 0; i < p.size(); ++i)
      names.emplace_back(std::string("p_") + model_names[i]);
    for (int i = 0; i < g.size(); ++i)
      names.emplace_back(std::string("g_") + model_names[i]);
  }

  virtual inline void get_params(std::vector<double>& values) {
    values.reserve(q.size() + p.size() + g.size());
    for (int i = 0; i < q.size(); ++i)
      values.push_back(q[i]);
    for (int i = 0; i < p.size(); ++i)
      values.push_back(p[i]);
    for (int i = 0; i < g.size(); ++i)
      values.push_back(g[i]);
  }

  /**
   * Writes the metric
   *
   * @param writer writer callback
   */
  virtual inline void write_metric(stan::callbacks::writer& writer) {}
};

namespace internal {
  template <typename Mem>
  inline auto make_vec(stan::math::stack_alloc* mem, int size) {
    return stack_matrix<Eigen::VectorXd>(size, mem);
  }
}
class ps_point_map {
 public:
  explicit ps_point_map(stan::math::stack_alloc& mem, int n) :
   q(n, &mem), p(n, &mem), g(n, &mem) {}
  template <typename PsPoint>
  ps_point_map(stan::math::stack_alloc& mem, PsPoint& ps) : q(ps.q, &mem), p(ps.p, &mem), g(ps.g, &mem) {
  }
  ps_point_map& operator=(ps_point& ps) {
    this->q = ps.q;
    this->p = ps.p;
    this->g = ps.g;
    return *this;
  }
  stack_matrix<Eigen::VectorXd> q;
  stack_matrix<Eigen::VectorXd> p;
  stack_matrix<Eigen::VectorXd> g;
  double V{0};

  virtual inline void get_param_names(std::vector<std::string>& model_names,
                                      std::vector<std::string>& names) {
    names.reserve(q.size() + p.size() + g.size());
    for (int i = 0; i < q.size(); ++i)
      names.emplace_back(model_names[i]);
    for (int i = 0; i < p.size(); ++i)
      names.emplace_back(std::string("p_") + model_names[i]);
    for (int i = 0; i < g.size(); ++i)
      names.emplace_back(std::string("g_") + model_names[i]);
  }

  virtual inline void get_params(std::vector<double>& values) {
    values.reserve(q.size() + p.size() + g.size());
    for (int i = 0; i < q.size(); ++i)
      values.push_back(q[i]);
    for (int i = 0; i < p.size(); ++i)
      values.push_back(p[i]);
    for (int i = 0; i < g.size(); ++i)
      values.push_back(g[i]);
  }

  /**
   * Writes the metric
   *
   * @param writer writer callback
   */
  virtual inline void write_metric(stan::callbacks::writer& writer) {}
};

ps_point::ps_point(ps_point_map& ps) : q(ps.q), p(ps.p), g(ps.g) {}
ps_point& ps_point::operator=(ps_point_map& ps) {
  this->q = ps.q;
  this->p = ps.p;
  this->g = ps.g;
  return *this;
}
ps_point::ps_point(ps_point_map&& ps) : q(std::move(ps.q)), p(std::move(ps.p)), g(std::move(ps.g)) {}
ps_point& ps_point::operator=(ps_point_map&& ps) {
  this->q = std::move(ps.q);
  this->p = std::move(ps.p);
  this->g = std::move(ps.g);
  return *this;
}
}  // namespace mcmc
}  // namespace stan
#endif

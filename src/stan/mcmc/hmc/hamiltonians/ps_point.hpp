#ifndef STAN_MCMC_HMC_HAMILTONIANS_PS_POINT_HPP
#define STAN_MCMC_HMC_HAMILTONIANS_PS_POINT_HPP

#include <stan/callbacks/writer.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <boost/lexical_cast.hpp>
#include <string>
#include <vector>

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
  inline auto make_vec(Mem& mem, int size) {
    return Eigen::Map<Eigen::VectorXd>(mem.template alloc_array<double>(size), size);
  }
}
class ps_point_map {
 public:
  explicit ps_point_map(stan::math::stack_alloc& mem, int n) :
   q(internal::make_vec(mem, n)), p(internal::make_vec(mem, n)), g(internal::make_vec(mem, n)) {}
  template <typename PsPoint>
  ps_point_map(stan::math::stack_alloc& mem, PsPoint& ps) :
   q(internal::make_vec(mem, ps.q.size())), p(internal::make_vec(mem, ps.q.size())), g(internal::make_vec(mem, ps.q.size())) {
      this->q = ps.q;
      this->p = ps.p;
      this->g = ps.g;
  }
  ps_point_map& operator=(ps_point& ps) {
    this->q = ps.q;
    this->p = ps.p;
    this->g = ps.g;
    return *this;
  }
  Eigen::Map<Eigen::VectorXd> q;
  Eigen::Map<Eigen::VectorXd> p;
  Eigen::Map<Eigen::VectorXd> g;
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

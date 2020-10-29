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

/**
 * Point in a generic phase space
 */
class ps_point {
 public:
  explicit ps_point(int n) : q(n), p(n), g(n) {}

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

}  // namespace mcmc
}  // namespace stan
#endif

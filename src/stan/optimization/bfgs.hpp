#ifndef STAN_OPTIMIZATION_BFGS_HPP
#define STAN_OPTIMIZATION_BFGS_HPP

#include <stan/math/prim.hpp>
#include <stan/model/log_prob_propto.hpp>
#include <stan/model/log_prob_grad.hpp>
#include <stan/optimization/bfgs_linesearch.hpp>
#include <stan/optimization/bfgs_update.hpp>
#include <stan/optimization/lbfgs_update.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace stan {
namespace optimization {
typedef enum {
  TERM_SUCCESS = 0,
  TERM_ABSX = 10,
  TERM_ABSF = 20,
  TERM_RELF = 21,
  TERM_ABSGRAD = 30,
  TERM_RELGRAD = 31,
  TERM_MAXIT = 40,
  TERM_LSFAIL = -1
} TerminationCondition;

template <typename Scalar = double>
class ConvergenceOptions {
 public:
  size_t maxIts{10000};
  Scalar tolAbsX{1e-8};
  Scalar tolAbsF{1e-12};
  Scalar tolRelF{1e+4};
  Scalar fScale{1.0};
  Scalar tolAbsGrad{1e-8};
  Scalar tolRelGrad{1e+3};
};

template <typename Scalar = double>
class LSOptions {
 public:
  Scalar c1{1e-4};
  Scalar c2{0.9};
  Scalar alpha0{1e-3};
  Scalar minAlpha{1e-12};
  Scalar maxLSIts{20};
  Scalar maxLSRestarts{10};
};
template <typename FunctorType, typename QNUpdateType, typename Scalar = double,
          int DimAtCompile = Eigen::Dynamic>
class BFGSMinimizer {
 public:
  typedef Eigen::Matrix<Scalar, DimAtCompile, 1> VectorT;
  typedef Eigen::Matrix<Scalar, DimAtCompile, DimAtCompile> HessianT;

 protected:
  FunctorType _func;
  VectorT _gk, _gk_1, _xk_1, _xk, _pk, _pk_1;
  Scalar _fk, _fk_1, _alphak_1;
  Scalar _alpha, _alpha0;
  size_t _itNum;
  std::string _note;
  QNUpdateType _qn;

 public:
  using ls_options_t = LSOptions<Scalar>;
  ls_options_t _ls_opts;
  using convergence_options_t = ConvergenceOptions<Scalar>;
  convergence_options_t _conv_opts;

  inline QNUpdateType &get_qnupdate() noexcept { return _qn; }
  inline const QNUpdateType &get_qnupdate() const noexcept { return _qn; }

  inline const Scalar &curr_f() const noexcept { return _fk; }
  inline const VectorT &curr_x() const noexcept { return _xk; }
  inline const VectorT &curr_g() const noexcept { return _gk; }
  inline const VectorT &curr_p() const noexcept { return _pk; }

  inline const Scalar &prev_f() const noexcept { return _fk_1; }
  inline const VectorT &prev_x() const noexcept { return _xk_1; }
  inline const VectorT &prev_g() const noexcept { return _gk_1; }
  inline const VectorT &prev_p() const noexcept { return _pk_1; }
  inline Scalar prev_step_size() const { return _pk_1.norm() * _alphak_1; }

  inline Scalar rel_grad_norm() const {
    return -_pk.dot(_gk) / std::max(std::fabs(_fk), _conv_opts.fScale);
  }
  inline Scalar rel_obj_decrease() const {
    return std::fabs(_fk_1 - _fk)
           / std::max(std::fabs(_fk_1),
                      std::max(std::fabs(_fk), _conv_opts.fScale));
  }

  inline const Scalar &alpha0() const noexcept { return _alpha0; }
  inline const Scalar &alpha() const noexcept { return _alpha; }
  inline const size_t iter_num() const noexcept { return _itNum; }

  inline const std::string &note() const noexcept { return _note; }

  inline std::string get_code_string(int retCode) const noexcept {
    switch (retCode) {
      case TERM_SUCCESS:
        return std::string("Successful step completed");
      case TERM_ABSF:
        return std::string(
            "Convergence detected: absolute change "
            "in objective function was below tolerance");
      case TERM_RELF:
        return std::string(
            "Convergence detected: relative change "
            "in objective function was below tolerance");
      case TERM_ABSGRAD:
        return std::string(
            "Convergence detected: "
            "gradient norm is below tolerance");
      case TERM_RELGRAD:
        return std::string(
            "Convergence detected: relative "
            "gradient magnitude is below tolerance");
      case TERM_ABSX:
        return std::string(
            "Convergence detected: "
            "absolute parameter change was below tolerance");
      case TERM_MAXIT:
        return std::string(
            "Maximum number of iterations hit, "
            "may not be at an optima");
      case TERM_LSFAIL:
        return std::string(
            "Line search failed to achieve a sufficient "
            "decrease, no more progress can be made");
      default:
        return std::string("Unknown termination code");
    }
  }
  template <typename Func>
  explicit BFGSMinimizer(Func &&f) : _func(std::forward<Func>(f)) {}
  template <typename Func, typename Vec, require_vector_t<Vec> * = nullptr,
            typename LSOpt, typename ConvergeOpt, typename QnUpdater>
  explicit BFGSMinimizer(Func &&f, Vec &&params_r, LSOpt &&ls_opt,
                         ConvergeOpt &&conv_opt, QnUpdater &&updater)
      : _func(std::forward<Func>(f)),
        _qn(std::forward<QnUpdater>(updater)),
        _ls_opts(std::forward<LSOpt>(ls_opt)),
        _conv_opts(std::forward<ConvergeOpt>(conv_opt)) {}

  template <typename Vec, require_vector_t<Vec> * = nullptr>
  void initialize(Vec &&x0) {
    int ret;
    _gk.resize(x0.size());
    _xk = Eigen::Map<Eigen::VectorXd>(x0.data(), x0.size());
    ret = _func(_xk, _fk, _gk);
    if (ret) {
      throw std::runtime_error("Error evaluating initial BFGS point.");
    }
    _pk = -_gk;

    _itNum = 0;
    _note = "";
  }

  int step() {
    Scalar gradNorm, stepNorm;
    VectorT sk, yk;
    int retCode(0);
    int resetB(0);

    _itNum++;

    if (_itNum == 1) {
      resetB = 1;
      _note = "";
    } else {
      resetB = 0;
      _note = "";
    }

    while (true) {
      if (resetB) {
        // Reset the Hessian approximation
        _pk.noalias() = -_gk;
      }

      // Get an initial guess for the step size (alpha)
      if (_itNum > 1 && resetB != 2) {
        // use cubic interpolation based on the previous step
        _alpha0 = _alpha = std::min(
            1.0, 1.01
                     * CubicInterp(_gk_1.dot(_pk_1), _alphak_1, _fk - _fk_1,
                                   _gk.dot(_pk_1), _ls_opts.minAlpha, 1.0));
      } else {
        // On the first step (or, after a reset) use the default step size
        _alpha0 = _alpha = _ls_opts.alpha0;
      }

      // Perform the line search.  If successful, the results are in the
      // variables: _xk_1, _fk_1 and _gk_1.
      retCode
          = WolfeLineSearch(_func, _alpha, _xk_1, _fk_1, _gk_1, _pk, _xk, _fk,
                            _gk, _ls_opts.c1, _ls_opts.c2, _ls_opts.minAlpha,
                            _ls_opts.maxLSIts, _ls_opts.maxLSRestarts);
      if (retCode) {
        // Line search failed...
        if (resetB) {
          // did a Hessian reset and it still failed,
          // and nothing left to try
          retCode = TERM_LSFAIL;
          return retCode;
        } else {
          // try resetting the Hessian approximation
          resetB = 2;
          _note += "LS failed, Hessian reset";
          continue;
        }
      } else {
        break;
      }
    }

    // Swap things so that k is the most recent iterate
    std::swap(_fk, _fk_1);
    _xk.swap(_xk_1);
    _gk.swap(_gk_1);
    _pk.swap(_pk_1);

    sk.noalias() = _xk - _xk_1;
    yk.noalias() = _gk - _gk_1;

    gradNorm = _gk.norm();
    stepNorm = sk.norm();

    // Update QN approximation
    if (resetB) {
      // If the QN approximation was reset, automatically scale it
      // and update the step-size accordingly
      Scalar B0fact = _qn.update(yk, sk, true);
      _pk_1 /= B0fact;
      _alphak_1 = _alpha * B0fact;
    } else {
      _qn.update(yk, sk);
      _alphak_1 = _alpha;
    }
    // Compute search direction for next step
    _qn.search_direction(_pk, _gk);

    // Check for convergence
    if (std::fabs(_fk_1 - _fk) < _conv_opts.tolAbsF) {
      // Objective function improvement wasn't sufficient
      retCode = TERM_ABSF;
    } else if (gradNorm < _conv_opts.tolAbsGrad) {
      retCode = TERM_ABSGRAD;  // Gradient norm was below threshold
    } else if (stepNorm < _conv_opts.tolAbsX) {
      retCode = TERM_ABSX;  // Change in x was too small
    } else if (_itNum >= _conv_opts.maxIts) {
      retCode = TERM_MAXIT;  // Max number of iterations hit
    } else if (rel_obj_decrease()
               < _conv_opts.tolRelF * std::numeric_limits<Scalar>::epsilon()) {
      // Relative improvement in objective function wasn't sufficient
      retCode = TERM_RELF;
    } else if (rel_grad_norm() < _conv_opts.tolRelGrad
                                     * std::numeric_limits<Scalar>::epsilon()) {
      // Relative gradient norm was below threshold
      retCode = TERM_RELGRAD;
    } else {
      // Step was successful more progress to be made
      retCode = TERM_SUCCESS;
    }

    return retCode;
  }

  int minimize(VectorT &x0) {
    int retcode;
    initialize(x0);
    while (!(retcode = step()))
      continue;
    x0 = _xk;
    return retcode;
  }
};

template <class M, bool jacobian = false>
class ModelAdaptor {
 private:
  M &_model;
  std::vector<int> _params_i;
  std::ostream *_msgs;
  std::vector<double> _x, _g;
  size_t _fevals;

 public:
  ModelAdaptor(M &model, const std::vector<int> &params_i, std::ostream *msgs)
      : _model(model), _params_i(params_i), _msgs(msgs), _fevals(0) {}

  size_t fevals() const { return _fevals; }
  int operator()(const Eigen::Matrix<double, Eigen::Dynamic, 1> &x, double &f) {
    using Eigen::Dynamic;
    using Eigen::Matrix;
    using stan::math::index_type;
    using stan::model::log_prob_propto;
    typedef typename index_type<Matrix<double, Dynamic, 1> >::type idx_t;

    _x.resize(x.size());
    for (idx_t i = 0; i < x.size(); i++)
      _x[i] = x[i];

    try {
      f = -log_prob_propto<jacobian>(_model, _x, _params_i, _msgs);
    } catch (const std::domain_error &e) {
      if (_msgs)
        (*_msgs) << e.what() << std::endl;
      return 1;
    }

    if (std::isfinite(f)) {
      return 0;
    } else {
      if (_msgs)
        *_msgs << "Error evaluating model log probability: "
                  "Non-finite function evaluation."
               << std::endl;
      return 2;
    }
  }
  int operator()(const Eigen::Matrix<double, Eigen::Dynamic, 1> &x, double &f,
                 Eigen::Matrix<double, Eigen::Dynamic, 1> &g) {
    using Eigen::Dynamic;
    using Eigen::Matrix;
    using stan::math::index_type;
    using stan::model::log_prob_grad;
    typedef typename index_type<Matrix<double, Dynamic, 1> >::type idx_t;

    _x.resize(x.size());
    for (idx_t i = 0; i < x.size(); i++)
      _x[i] = x[i];

    _fevals++;

    try {
      f = -log_prob_grad<true, jacobian>(_model, _x, _params_i, _g, _msgs);
    } catch (const std::domain_error &e) {
      if (_msgs)
        (*_msgs) << e.what() << std::endl;
      return 1;
    }

    g.resize(_g.size());
    for (size_t i = 0; i < _g.size(); i++) {
      if (!std::isfinite(_g[i])) {
        if (_msgs)
          *_msgs << "Error evaluating model log probability: "
                    "Non-finite gradient."
                 << std::endl;
        return 3;
      }
      g[i] = -_g[i];
    }

    if (std::isfinite(f)) {
      return 0;
    } else {
      if (_msgs)
        *_msgs << "Error evaluating model log probability: "
               << "Non-finite function evaluation." << std::endl;
      return 2;
    }
  }
  int df(const Eigen::Matrix<double, Eigen::Dynamic, 1> &x,
         Eigen::Matrix<double, Eigen::Dynamic, 1> &g) {
    double f;
    return (*this)(x, f, g);
  }
};

/**
 * @tparam jacobian `true` to include Jacobian adjustment (default `false`)
 */
template <typename M, typename QNUpdateType, typename Scalar = double,
          int DimAtCompile = Eigen::Dynamic, bool jacobian = false>
class BFGSLineSearch
    : public BFGSMinimizer<ModelAdaptor<M, jacobian>, QNUpdateType, Scalar,
                           DimAtCompile> {
 public:
  typedef BFGSMinimizer<ModelAdaptor<M, jacobian>, QNUpdateType, Scalar,
                        DimAtCompile>
      BFGSBase;
  typedef typename BFGSBase::VectorT vector_t;
  typedef typename stan::math::index_type<vector_t>::type idx_t;

  template <typename Vec, require_vector_t<Vec> * = nullptr>
  BFGSLineSearch(M &model, Vec &&params_r, const std::vector<int> &params_i,
                 std::ostream *msgs = 0)
      : BFGSBase(ModelAdaptor<M, jacobian>(model, params_i, msgs)) {
    BFGSBase::initialize(params_r);
  }

  template <typename Vec, typename LSOpt, typename ConvergeOpt,
            typename QnUpdater, require_vector_t<Vec> * = nullptr>
  BFGSLineSearch(M &model, Vec &&params_r, const std::vector<int> &params_i,
                 LSOpt &&ls_options, ConvergeOpt &&convergence_options,
                 QnUpdater &&qn_update, std::ostream *msgs = 0)
      : BFGSBase(ModelAdaptor<M, jacobian>(model, params_i, msgs), params_r,
                 ls_options, convergence_options, qn_update) {
    BFGSBase::initialize(params_r);
  }

  template <typename Vec, require_vector_t<Vec> * = nullptr>
  void initialize(Vec &&params_r) {
    BFGSBase::initialize(params_r);
  }

  size_t grad_evals() { return this->_func.fevals(); }
  double logp() { return -(this->curr_f()); }
  double grad_norm() { return this->curr_g().norm(); }
  void grad(std::vector<double> &g) {
    const vector_t &cg(this->curr_g());
    g.resize(cg.size());
    for (idx_t i = 0; i < cg.size(); i++)
      g[i] = -cg[i];
  }
  void params_r(std::vector<double> &x) {
    const vector_t &cx(this->curr_x());
    x.resize(cx.size());
    for (idx_t i = 0; i < cx.size(); i++)
      x[i] = cx[i];
  }
};

}  // namespace optimization

}  // namespace stan

#endif

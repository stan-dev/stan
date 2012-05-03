#ifndef __STAN__MCMC__DUALAVERAGE_H__
#define __STAN__MCMC__DUALAVERAGE_H__

#include <cmath>
#include <cstddef>

#include <vector>

#include <iostream>

namespace stan {

  namespace mcmc {

    /**
     * Implements Nesterov's dual average algorithm.
     *
     * Use by repeatedly calling update() with the gradient evaluated at
     * xk(). When finished, use the average value of all the xk's by
     * calling xbar().
     *
     * Iterates are shrunk towards a prespecified point x0. The level
     * of shrinkage is controlled by the parameter gamma and gets
     * weaker with each iteration.
     * 
     * Note that this is an algorithm for MINIMIZATION, not MAXIMIZATION.
     */
    class DualAverage {
    protected:
      std::vector<double> _gbar, _xbar, _x0, _lastx;
      int _k;
      double _gamma;

    public:
      /**
       * Constructor.
       *
       * @param gamma Regularization parameter. Higher values mean that
       * iterates will be shrunk towards x0 more strongly.
       * @param x0 Point towards which iterates are shrunk.
       */
      DualAverage(double gamma, const std::vector<double>& x0) 
        : _gbar(x0.size(), 0), _xbar(x0.size(), 0), _x0(x0), _lastx(x0),
          _k(0), _gamma(gamma){
      }

      /**
       * Produces the next iterate xk given the current gradient g.
       *
       * @param[in]  g The new subgradient/stochastic gradient.
       * @param[out] xk The next iterate produced by the algorithm.
       */
      void update(const std::vector<double>& g, std::vector<double>& xk) {
        _k++;
        xk.resize(_gbar.size());
        double avgeta = 1.0 / (_k + 10);
        double xbar_avgeta = pow(_k, -0.75);
        double muk = 0.5 * sqrt(_k) / _gamma;
        for (size_t i = 0; i < _gbar.size(); ++i) {
          _gbar[i] = avgeta * g[i] + (1 - avgeta) * _gbar[i];
          xk[i] = _x0[i] - muk * _gbar[i];
          // fprintf(stderr, "DUALAVERAGE update %d: g = %f, gbar = %f, lastx = %f",
          //   _k, g[0], _gbar[0], _lastx[0]);
          _lastx[i] = xk[i];
          _xbar[i] = xbar_avgeta * xk[i] + (1 - xbar_avgeta) * _xbar[i];
        }
        // fprintf(stderr, ", xk = %f\n", xk[0]);
      }

      /**
       * Set the point towards which each iterate is shrunk.
       *
       * @param x0 The point towards each iterate will be shrunk.
       */
      inline void setx0(const std::vector<double>& x0) {
        _x0.assign(x0.begin(), x0.end());
      }

      /**
       * Get the exponentially weighted moving average of all previous
       * iterates.
       *
       * @param[out] xbar Where to return the exponentially weighted moving
       * average of all previous iterates.
       */
      inline void xbar(std::vector<double>& xbar) {
        xbar.assign(_xbar.begin(), _xbar.end());
      }
      /**
       * Get the average of all previous gradients.
       *
       * @param[out] gbar Where to return the average of all previous gradients.
       */
      inline void gbar(std::vector<double>& gbar) {
        gbar.assign(_gbar.begin(), _gbar.end());
      }
      /**
       * Get the current iterate.
       *
       * @param[out] xk Where to return the current iterate.
       */
      inline void xk(std::vector<double>& xk) {
        xk.assign(_lastx.begin(), _lastx.end());
      }
      /**
       * Get the point towards which we're shrinking the iterates.
       *
       * @param x0 Where to return the point towards which we're
       * shrinking the iterates.
       */
      inline void x0(std::vector<double>& x0) {
        x0.assign(_x0.begin(), _x0.end());
      }
      /**
       * Return how many iterations we've run for.
       *
       * @return how many iterations we've run for.
       */
      inline int k() { return _k; }
      /**
       * Return the regularization parameter gamma.
       *
       * @return the regularization parameter gamma.
       */
      inline double gamma() { return _gamma; }
    };

  }
}

#endif

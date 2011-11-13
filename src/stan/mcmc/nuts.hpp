#ifndef __STAN__MCMC__NUTS_H__
#define __STAN__MCMC__NUTS_H__

#include <ctime>
#include <vector>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_01.hpp>
#include "stan/mcmc/adaptive_sampler.hpp"
#include "stan/mcmc/dualaverage.hpp"
#include "stan/maths/util.hpp"

namespace stan {

  namespace mcmc {

    using namespace stan::util;

    /**
     * No-U-Turn Sampler (NUTS).
     *
     * The NUTS sampler requires a probability model with the ability
     * to compute gradients, characterized as an instance of
     * <code>prob_grad</code>.  
     *
     * Samples from the sampler are returned through the
     * base class <code>sampler</code>.
     */
    class nuts : public adaptive_sampler {
    protected:
      // Provides the target distribution we're trying to sample from
      mcmc::prob_grad& _model;
    
      // The most recent setting of the real-valued parameters
      std::vector<double> _x;
      // The most recent setting of the discrete parameters
      std::vector<int> _z;
      // The most recent gradient with respect to the real parameters
      std::vector<double> _g;
      // The most recent log-likelihood
      double _logp;

      // The step size used in the Hamiltonian simulation
      double _epsilon;
      // The desired value of E[number of states in slice in last doubling]
      double _delta;

      // How many in-slice states we've seen this iteration
      int _ninslice;

      // RNGs
      boost::mt19937 _rand_int;
      boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > _rand_unit_norm;
      boost::uniform_01<boost::mt19937&> _rand_uniform_01;

      // Stop immediately if H < u - _maxchange
      const double _maxchange;

      // Class implementing Nesterov's primal-dual averaging
      DualAverage _da;
      // Gamma parameter for dual averaging.
      static double da_gamma() { return 0.05; }

      /**
       * Determine whether we've started to make a "U-turn" at either end
       * of the position-state trajectory beginning with {xminus, mminus}
       * and ending with {xplus, mplus}.
       *
       * @return 0 if we've made a U-turn, 1 otherwise.
       */
      inline static int computeCriterion(std::vector<double>& xplus,
                                         std::vector<double>& xminus,
                                         std::vector<double>& mplus,
                                         std::vector<double>& mminus) {
        std::vector<double> total_direction;
        sub(xplus, xminus, total_direction);
        return (dot(total_direction, mminus) > 0) &&
          (dot(total_direction, mplus) > 0);
      }

    public:

      double epsilon() { return _epsilon; }
      void setEpsilon(double epsilon) { _epsilon = epsilon; }

      /**
       * Construct a No-U-Turn Sampler (NUTS) for the specified model,
       * using the specified step size and number of leapfrog steps,
       * with the specified random seed for randomization.
       *
       * If the same seed is used twice, the series of samples should
       * be the same.  This property is most helpful for testing.  If no
       * random seed is specified, the <code>std::time(0)</code> function is
       * called from the <code>ctime</code> library.
       * 
       * @param model Probability model with gradients.
       * @param epsilon Optional (initial) Hamiltonian dynamics simulation
       * step size. If not specified or set < 0, find_reasonable_parameters()
       * will be called to initialize epsilon.
       * @param delta Optional target value between 0 and 1 used to tune 
       * epsilon. Lower delta => higher epsilon => more efficiency, unless
       * epsilon gets _too_ big in which case efficiency suffers.
       * If not specified, defaults to the usually reasonable value of 0.5.
       * @param random_seed Optional Seed for random number generator; if not
       * specified, generate new seed based on system time.
       */
      nuts(mcmc::prob_grad& model, double delta = 0.5, double epsilon = -1,
           unsigned int random_seed = static_cast<unsigned int>(std::time(0)))
        : _model(model),
          _x(model.num_params_r()),
          _z(model.num_params_i()),
          _g(model.num_params_r()),

          _epsilon(epsilon),
          _delta(delta),

          _ninslice(0),

          _rand_int(random_seed),
          _rand_unit_norm(_rand_int,
                          boost::normal_distribution<>()),
          _rand_uniform_01(_rand_int),

          _maxchange(-1000),

          _da(da_gamma(), std::vector<double>(1, 0)) {
        model.init(_x, _z);
        _logp = model.grad_log_prob(_x, _z, _g);
        if (_epsilon <= 0)
          find_reasonable_parameters();
        // Err on the side of regularizing epsilon towards being too big;
        // the logic is that it's cheaper to run NUTS when epsilon's large.
        _da.setx0(std::vector<double>(1, log(_epsilon * 10)));
      }

      /**
       * Destroy this sampler.
       *
       * The implementation for this class is a no-op.
       */
      virtual ~nuts() {
      }

      /**
       * Set the model real and integer parameters to the specified
       * values.  
       *
       * This method will typically be used to set the parameters
       * by the client of this class after initialization.  
       *
       * @param x Real parameters.
       * @param z Integer parameters.
       */
      virtual void set_params(std::vector<double> x,
                              std::vector<int> z) {
        assert(x.size() == _x.size());
        assert(z.size() == _z.size());
        _x = x;
        _z = z;
        _logp = _model.grad_log_prob(_x,_z,_g);
      }

      /**
       * Search for a roughly reasonable (within a factor of 2)
       * setting of the step size epsilon.
       */
      virtual void find_reasonable_parameters() {
        _epsilon = 1;
        std::vector<double> x = _x;
        std::vector<double> m(_model.num_params_r());
        for (unsigned int i = 0; i < m.size(); ++i)
          m[i] = _rand_unit_norm();
        std::vector<double> g = _g;
        double lastlogp = _logp;
        double logp = leapfrog(_model, _z, x, m, g, _epsilon);
        double H = logp - lastlogp;
        int direction = H > log(0.5) ? 1 : -1;
//         fprintf(stderr, "epsilon = %f.  initial logp = %f, lf logp = %f\n", 
//                 _epsilon, lastlogp, logp);
        while (1) {
          x = _x;
          g = _g;
          for (unsigned int i = 0; i < m.size(); ++i)
            m[i] = _rand_unit_norm();
          logp = leapfrog(_model, _z, x, m, g, _epsilon);
          H = logp - lastlogp;
//           fprintf(stderr, "epsilon = %f.  initial logp = %f, lf logp = %f\n", 
//                   _epsilon, lastlogp, logp);
          if ((direction == 1) && (H < log(0.5)))
            break;
          else if ((direction == -1) && (H > log(0.5)))
            break;
          else
            _epsilon = direction == 1 ? 2 * _epsilon : 0.5 * _epsilon;
        }
      }

      /**
       * Return the next sample.
       *
       * @return The next sample.
       */
      sample next() {
        // Initialize the algorithm
        std::vector<double> mminus(_model.num_params_r());
        for (unsigned int i = 0; i < mminus.size(); ++i)
          mminus[i] = _rand_unit_norm();
        std::vector<double> mplus(mminus);
        // The log-joint probability of the momentum and position terms, i.e.
        // -(kinetic energy + potential energy)
        double H = -0.5 * dot_self(mminus) + _logp;

        std::vector<double> gradminus(_g);
        std::vector<double> gradplus(_g);
        std::vector<double> xminus(_x);
        std::vector<double> xplus(_x);

        // Sample the slice variable
        double u = log(_rand_uniform_01()) + H;

        // Do the first iteration by hand
        int depth = 1;
        int nvalid = 1;
        int direction = _rand_uniform_01() < 0.5;
        if (direction == 0) {
          // Go backwards
          double logpminus = leapfrog(_model, _z, xminus, mminus, gradminus,
                                      -_epsilon);
          H = -0.5 * dot_self(mminus) + logpminus;
          if (u < H) {
            _x = xminus;
            _g = gradminus;
            _logp = logpminus;
          }
        } else {
          // Go forwards
          double logpplus = leapfrog(_model, _z, xplus, mplus, gradplus,
                                     _epsilon);
          H = -0.5 * dot_self(mplus) + logpplus;
          if (u < H) {
            _x = xplus;
            _g = gradplus;
            _logp = logpplus;
          }
        }
        // Bookkeeping
        ++_nfevals;
        nvalid += u < H;
        _ninslice = u < H;
        // Stop if we're turning around or the error is enormous
        int criterion = computeCriterion(xplus, xminus, mplus, mminus);
        criterion &= H - u > _maxchange;

        // Now repeatedly double the set of points we've visited
        std::vector<double> newsample, newgrad, tempx, tempm, tempg;
        double newlogp = -1;
        int nvalid1 = -1;
        int nvalid2 = -1;
        while (criterion) {
          _ninslice = 0;
          direction = _rand_uniform_01() < 0.5;
          if (direction == 0)
            build_tree(direction, xminus, mminus, gradminus, depth,
                       u, newsample, newgrad,
                       newlogp, xminus, tempx, mminus, tempm, gradminus, tempg,
                       criterion, nvalid2);
          else
            build_tree(direction, xplus, mplus, gradplus, depth,
                       u, newsample, newgrad,
                       newlogp, tempx, xplus, tempm, mplus, tempg, gradplus,
                       criterion, nvalid2);
          // We can't look at the results of this last doubling if criterion==0
          if (!criterion)
            break;
          nvalid1 = nvalid;
          nvalid += nvalid2;
          criterion = computeCriterion(xplus, xminus, mplus, mminus);
          // Metropolis-Hastings to determine if we can jump to a point in
          // the new half-tree
          if (_rand_uniform_01() < float(nvalid2) / (1e-100+float(nvalid1))) {
            _x = newsample;
            _g = newgrad;
            _logp = newlogp;
          }
          ++depth;
        }

        // Now we just have to update epsilon, if adaptation is on.
        double adapt_stat = float(_ninslice) / float(1 << (depth-1));
        if (_adapt) {
          double adapt_g = adapt_stat - _delta;
          std::vector<double> gvec(1, -adapt_g);
          std::vector<double> result;
          _da.update(gvec, result);
          _epsilon = exp(result[0]);
          ++_n_adapt_steps;
        }
        std::vector<double> result;
        _da.xbar(result);
        fprintf(stderr, "xbar = %f\n", exp(result[0]));
        ++_n_steps;
        double avg_eta = 1.0 / _n_steps;
        _mean_stat = avg_eta * adapt_stat + (1 - avg_eta) * _mean_stat;

        mcmc::sample s(_x, _z, _logp);
        return s;
      }

      /**
       * The core recursion in NUTS.
       *
       * @param direction Simulate backwards if -1, forwards if 1.
       * @param x The position value to start from.
       * @param m The momentum value to start from.
       * @param grad The gradient at the initial position.
       * @param depth The depth of the tree to build---we'll run 2^depth
       * leapfrog steps.
       * @param u The slice variable.
       * @param newsample Returns the position of the new sample selected from
       * this subtree.
       * @param newgrad Returns the gradient at the new sample selected from
       * this subtree.
       * @param newlogp Returns the log-probability of the new sample selected
       * from this subtree.
       * @param xminus Returns the position of the backwardmost leaf of this
       * subtree.
       * @param xplus Returns the position of the forwardmost leaf of this
       * subtree.
       * @param mminus Returns the momentum of the backwardmost leaf of this
       * subtree.
       * @param mplus Returns the momentum of the forwardmost leaf of this
       * subtree.
       * @param gradminus Returns the gradient at xminus.
       * @param gradplus Returns the gradient at xplus.
       * @param criterion Returns 1 if the subtree is usable, 0 if not.
       * @param nvalid Returns the number of usable points in the subtree.
       */
      void build_tree(int direction, std::vector<double>& x,
                      std::vector<double>& m,
                      std::vector<double>& grad,
                      int depth,
                      double u,
                      std::vector<double>& newsample, 
                      std::vector<double>& newgrad, 
                      double& newlogp, std::vector<double>& xminus,
                      std::vector<double>& xplus,
                      std::vector<double>& mminus,
                      std::vector<double>& mplus,
                      std::vector<double>& gradminus,
                      std::vector<double>& gradplus,
                      int& criterion, int& nvalid) {
        int criterion1 = 1;
        int criterion2 = 1;
        if (depth == 1) { // base case
          double logpplus = -1e100;
          double logpminus = -1e100;
          double Hplus = -1e100;
          double Hminus = -1e100;
          if (direction == 0) {
            xplus = x;
            mplus = m;
            gradplus = grad;
            logpplus = leapfrog(_model, _z, xplus, mplus, gradplus, -_epsilon);
            Hplus = logpplus - 0.5 * dot_self(mplus);
            criterion1 &= Hplus - u > _maxchange;
            if (criterion1) {
              xminus = xplus;
              mminus = mplus;
              gradminus = gradplus;
              logpminus = leapfrog(_model, _z, xminus, mminus, gradminus,-_epsilon);
              Hminus = logpminus - 0.5 * dot_self(mminus);
              criterion2 &= Hminus - u > _maxchange;
              ++_nfevals;
            }
          } else {
            xminus = x;
            mminus = m;
            gradminus = grad;
            logpminus = leapfrog(_model, _z, xminus, mminus, gradminus, _epsilon);
            Hminus = logpminus - 0.5 * dot_self(mminus);
            criterion1 &= Hminus - u > _maxchange;
            if (criterion1) {
              xplus = xminus;
              mplus = mminus;
              gradplus = gradminus;
              logpplus = leapfrog(_model, _z, xplus, mplus, gradplus, _epsilon);
              Hplus = logpplus - 0.5 * dot_self(mplus);
              criterion2 &= Hplus - u > _maxchange;
              ++_nfevals;
            }
          }
          ++_nfevals;
          if ((u < Hplus) && ((_rand_uniform_01() < 0.5) || (u > Hminus))) {
            newsample = xplus;
            newgrad = gradplus;
            newlogp = logpplus;
          } else {
            newsample = xminus;
            newgrad = gradminus;
            newlogp = logpminus;
          }
          nvalid = (u < Hplus) + (u < Hminus);
          _ninslice += nvalid;
        } else { // depth > 1
          std::vector<double> newsample1, newgrad1, newsample2, newgrad2,
            tempx, tempm, tempg;
          double newlogp1, newlogp2;
          int nvalid1, nvalid2;
          build_tree(direction, x, m, grad, depth-1,
                  u, newsample1, newgrad1, newlogp1,
                  xminus, xplus, mminus, mplus, gradminus, gradplus,
                  criterion1, nvalid1);
          if (criterion1) {
            if (direction == 0)
              build_tree(direction, xminus, mminus, gradminus, depth-1,
                      u, newsample2, newgrad2, newlogp2,
                      xminus, tempx, mminus, tempm, gradminus, tempg,
                      criterion2, nvalid2);
            else
              build_tree(direction, xplus, mplus, gradplus, depth-1,
                      u, newsample2, newgrad2, newlogp2,
                      tempx, xplus, tempm, mplus, tempg, gradplus,
                      criterion2, nvalid2);
          }

          if (criterion1 && criterion2) {
            if (_rand_uniform_01() < float(nvalid1) / float(nvalid1 + nvalid2)) {
              newsample = newsample1;
              newgrad = newgrad1;
              newlogp = newlogp1;
            } else {
              newsample = newsample2;
              newgrad = newgrad2;
              newlogp = newlogp2; 
            }
            nvalid = nvalid1 + nvalid2;
          } else {
            nvalid = 0;
          }
        }

        criterion = computeCriterion(xplus, xminus, mplus, mminus);
        criterion &= criterion1 && criterion2;
      }

      /**
       * Turn off parameter adaptation. Because we're using
       * primal-dual averaging, once we're done adapting we want to
       * set epsilon=the _average_ value of epsilon over each
       * adaptation step. This results in a lower-variance estimate of
       * the optimal epsilon.
       */
      virtual void adapt_off() {
        _adapt = 0;
        std::vector<double> result;
        _da.xbar(result);
        _epsilon = exp(result[0]);
      }

      /**
       * Return the value of epsilon.
       *
       * @param params Where to store epsilon.
       */
      virtual void get_parameters(std::vector<double>& params) {
        params.assign(1, _epsilon);
      }
    };

  }

}

#endif

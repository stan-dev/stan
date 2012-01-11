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
    template <typename BaseRNG = boost::mt19937>
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

      // RNGs
      BaseRNG _rand_int;
      boost::variate_generator<BaseRNG&, boost::normal_distribution<> > _rand_unit_norm;
      boost::uniform_01<BaseRNG&> _rand_uniform_01;

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
       * If not specified, defaults to the usually reasonable value of 0.6.
       * @param random_seed Optional Seed for random number generator; if not
       * specified, generate new seed based on system time.
       */
      nuts(mcmc::prob_grad& model, 
           double delta = 0.6, 
           double epsilon = -1,
           BaseRNG base_rng = BaseRNG(std::time(0)))
        : _model(model),
          _x(model.num_params_r()),
          _z(model.num_params_i()),
          _g(model.num_params_r()),

          _epsilon(epsilon),
          _delta(delta),

          _rand_int(base_rng),
          _rand_unit_norm(_rand_int,
                          boost::normal_distribution<double>()),
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
        double H0 = -0.5 * dot_self(mminus) + _logp;

        std::vector<double> gradminus(_g);
        std::vector<double> gradplus(_g);
        std::vector<double> xminus(_x);
        std::vector<double> xplus(_x);

        // Sample the slice variable
        double u = log(_rand_uniform_01()) + H0;
        int depth = 0;
        int nvalid = 1;
        int direction = 2 * (_rand_uniform_01() > 0.5) - 1;
        int criterion = 1;

        // Repeatedly double the set of points we've visited
        std::vector<double> newx, newgrad, dummy1, dummy2, dummy3;
        double newlogp = -1;
        double prob_sum = -1;
        int newnvalid = -1;
        int n_considered = 0;
        while (criterion) {
          direction = 2 * (_rand_uniform_01() > 0.5) - 1;
          if (direction == -1)
            build_tree(xminus, mminus, gradminus, u, direction, depth,
                       H0, xminus, mminus, gradminus, dummy1, dummy2, dummy3,
                       newx, newgrad, newlogp, newnvalid, criterion, prob_sum,
                       n_considered);
          else
            build_tree(xplus, mplus, gradplus, u, direction, depth,
                       H0, dummy1, dummy2, dummy3, xplus, mplus, gradplus, 
                       newx, newgrad, newlogp, newnvalid, criterion, prob_sum,
                       n_considered);
          // We can't look at the results of this last doubling if criterion==0
          if (!criterion)
            break;
          criterion = computeCriterion(xplus, xminus, mplus, mminus);
          // Metropolis-Hastings to determine if we can jump to a point in
          // the new half-tree
          if (_rand_uniform_01() < float(newnvalid) / (1e-100+float(nvalid))) {
            _x = newx;
            _g = newgrad;
            _logp = newlogp;
          }
          nvalid += newnvalid;
          ++depth;
        }

        // Now we just have to update epsilon, if adaptation is on.
        double adapt_stat = prob_sum / float(n_considered);
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
//         fprintf(stderr, "xbar = %f\n", exp(result[0]));
        ++_n_steps;
        double avg_eta = 1.0 / _n_steps;
        _mean_stat = avg_eta * adapt_stat + (1 - avg_eta) * _mean_stat;

        mcmc::sample s(_x, _z, _logp);
        return s;
      }

      /**
       * The core recursion in NUTS.
       *
       * @param x The position value to start from.
       * @param m The momentum value to start from.
       * @param grad The gradient at the initial position.
       * @param u The slice variable.
       * @param direction Simulate backwards if -1, forwards if 1.
       * @param depth The depth of the tree to build---we'll run 2^depth
       * leapfrog steps.
       * @param H0 The joint probability of the position-momentum we started
       * from initially---used to compute statistic to adapt epsilon.
       * @param newsample Returns the position of the new sample selected from
       * this subtree.
       * @param xminus Returns the position of the backwardmost leaf of this
       * subtree.
       * @param mminus Returns the momentum of the backwardmost leaf of this
       * subtree.
       * @param gradminus Returns the gradient at xminus.
       * @param xplus Returns the position of the forwardmost leaf of this
       * subtree.
       * @param mplus Returns the momentum of the forwardmost leaf of this
       * subtree.
       * @param gradplus Returns the gradient at xplus.
       * @param newx Returns the new position sample selected from
       * this subtree.
       * @param newgrad Returns the gradient at the new sample selected from
       * this subtree.
       * @param newlogp Returns the log-probability of the new sample selected
       * from this subtree.
       * @param n Returns the number of usable points in the subtree.
       * @param criterion Returns 1 if the subtree is usable, 0 if not.
       * @param prob_sum Returns the sum of the HMC acceptance probabilities
       * at each point in the subtree.
       * @param n_considered Returns the number of states in the subtree.
       */
      void build_tree(const std::vector<double>& x, 
                      const std::vector<double>& m,
                      const std::vector<double>& grad,
                      double u,
                      int direction,
                      int depth, 
                      double H0,
                      std::vector<double>& xminus,
                      std::vector<double>& mminus,
                      std::vector<double>& gradminus,
                      std::vector<double>& xplus,
                      std::vector<double>& mplus,
                      std::vector<double>& gradplus,
                      std::vector<double>& newx,
                      std::vector<double>& newgrad,
                      double& newlogp,
                      int& nvalid, 
                      int& criterion, 
                      double& prob_sum, 
                      int& n_considered) {
        if (depth == 0) {   // base case
          xminus = x;
          gradminus = grad;
          mminus = m;
          newlogp = leapfrog(_model, _z, xminus, mminus, gradminus,
                             direction * _epsilon);
          newx = xminus;
          newgrad = gradminus;
          xplus = xminus;
          mplus = mminus;
          gradplus = gradminus;
          double newH = newlogp - 0.5 * dot_self(mminus);
          if (newH != newH) // treat nan as -inf
            newH = -std::numeric_limits<double>::infinity();
          nvalid = newH > u;
          criterion = newH - u > _maxchange;
          prob_sum = min(1, exp(newH - H0));
          n_considered = 1;
          ++_nfevals;
        } else {            // depth >= 1
          build_tree(x, m, grad, u, direction, depth-1, H0, xminus, mminus,
                     gradminus, xplus, mplus, gradplus, newx, newgrad, newlogp,
                     nvalid, criterion, prob_sum, n_considered);
          if (criterion) {
            std::vector<double> dummy1, dummy2, dummy3;
            std::vector<double> newx2;
            std::vector<double> newgrad2;
            double newlogp2;
            int nvalid2;
            int criterion2;
            double prob_sum2;
            int n_considered2;
            if (direction == -1)
              build_tree(xminus, mminus, gradminus, u, direction, depth-1, H0,
                         xminus, mminus, gradminus, dummy1, dummy2, dummy3,
                         newx2, newgrad2, newlogp2, nvalid2, criterion2, 
                         prob_sum2, n_considered2);
            else
              build_tree(xplus, mplus, gradplus, u, direction, depth-1, H0,
                         dummy1, dummy2, dummy3, xplus, mplus, gradplus, 
                         newx2, newgrad2, newlogp2, nvalid2, criterion2,
                         prob_sum2, n_considered2);
            if (criterion && 
                (_rand_uniform_01() < float(nvalid2) / float(nvalid+nvalid2))){
              newx = newx2;
              newgrad = newgrad2;
              newlogp = newlogp2;
            }
            n_considered += n_considered2;
            prob_sum += prob_sum2;
            criterion &= criterion2;
            nvalid += nvalid2;
          }
          criterion &= computeCriterion(xplus, xminus, mplus, mminus);
        }
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

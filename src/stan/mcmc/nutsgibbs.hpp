#ifndef __STAN__MCMC__NUTSGIBBS_H__
#define __STAN__MCMC__NUTSGIBBS_H__

#include <ctime>
#include <vector>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_01.hpp>
#include "stan/mcmc/sampler.hpp"
#include "stan/mcmc/prob_grad.hpp"
#include "stan/mcmc/hmc.hpp"
#include "stan/mcmc/nuts.hpp"


namespace stan {


  namespace mcmc {

    /**
     * No-U-Turn Sampler (NUTSGIBBS).
     *
     * The NUTSGIBBS sampler requires a probability model with the ability
     * to compute gradients, characterized as an instance of
     * <code>prob_grad</code>.  
     *
     * Samples from the sampler are returned through the
     * base class <code>sampler</code>.
     */
    class nutsgibbs : public sampler {
    private:
      mcmc::prob_grad& _model;
    
      std::vector<double> _x;
      std::vector<double> _lastx;
      std::vector<unsigned int> _z;
      std::vector<double> _g;
      double _E;

      double _epsilon;
      int _nsamples;
      int _adapttime;
      double _delta;

      double _H0;
      double _meanacceptprob;

      boost::mt19937 _rand_int;
      boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > _rand_unit_norm;
      boost::uniform_01<boost::mt19937&> _rand_uniform_01;

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

      /**
       * Construct a No-U-Turn Sampler (NUTSGIBBS) for the specified model,
       * using the specified step size and number of leapfrog steps,
       * with the specified random seed for randomization.
       *
       * If the same seed is used twice, the series of samples should
       * be the same.  This property is most helpful for testing.  If no
       * random seed is specified, the <code>std::time(0)</code> function is
       * called from the <code>ctime</code> library.
       * 
       * @param model Probability model with gradients.
       * @param epsilon (initial) Hamiltonian dynamics simulation step size.
       * @param adapttime How many iterations to spend adapting epsilon. Samples
       * taken before the adaptation of epsilon is complete may have undesirable
       * properties.
       * @param random_seed Seed for random number generator; optional, if not
       * specified, generate new seen based on system time.
       */
      nutsgibbs(mcmc::prob_grad& model,
             double epsilon, int adapttime, double delta,
             unsigned int random_seed = static_cast<unsigned int>(std::time(0)))
	: _model(model),
	  _x(model.num_params_r()),
	  _z(model.num_params_i()),
	  _g(model.num_params_r()),

	  _epsilon(epsilon),
          _nsamples(0),
          _adapttime(adapttime),
          _delta(delta),

          _H0(0),
          _meanacceptprob(0),

	  _rand_int(random_seed),
	  _rand_unit_norm(_rand_int,
			  boost::normal_distribution<>()),
	  _rand_uniform_01(_rand_int) {

	model.init(_x,_z);
	_E = -model.grad_log_prob(_x,_z,_g);
      }

      /**
       * Destroy this sampler.
       *
       * The implementation for this class is a no-op.
       */
      virtual ~nutsgibbs() {
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
       * @throw std::invalid_argument if x or z do not match size 
       *    of parameters specified by the model.
       */
      virtual void set_params(std::vector<double> x,
                              std::vector<unsigned int> z) {
	if (x.size() != _x.size() || z.size() != _z.size())
	  throw std::invalid_argument();
	_x = x;
        _z = z;
	_E = -_model.grad_log_prob(_x,_z,_g);
      }

      /**
       * Returns log(exp(a) + exp(b)), being careful not to over/underflow
       */
      inline static double logSumExp(double a, double b) {
        if (a > b)
          return log(1 + exp(b - a)) + a;
        else
          return log(exp(a - b) + 1) + b;
      }

      /**
       * Returns exp(a) / (exp(a)+exp(b)), being careful not to over/underflow
       */
      inline static double expRatio(double a, double b) {
        if (a > b)
          return 1.0 / (1.0 + exp(b - a));
        else
          return 1.0 - 1.0 / (1.0 + exp(a - b));
      }

      /**
       * Return the next sample.
       *
       * @return The next sample.
       */
      sample next() {
        _lastx = _x;
	std::vector<double> mminus(_model.num_params_r());
	for (unsigned int i = 0; i < mminus.size(); ++i)
	  mminus[i] = _rand_unit_norm();
	std::vector<double> mplus(mminus);
	double H = -0.5 * dot_self(mminus) - _E;
        _H0 = H;
        _meanacceptprob = 0;

	std::vector<double> gradminus(_g);
	std::vector<double> gradplus(_g);
	std::vector<double> xminus(_x);
	std::vector<double> xplus(_x);

        int depth = 1;
        double totalweight = H;
        int direction = _rand_uniform_01() < 0.5;
        if (direction == 0) {
          double logpminus = leapfrog(_model, _z, xminus, mminus, gradminus,
                                      -_epsilon);
          H = -0.5 * dot_self(mminus) + logpminus;
          if (_rand_uniform_01() < exp(H - totalweight)) {
//           if (u < H) {
//           if ((u < H) && (_rand_uniform_01() < 0.5)) {
            _x = xminus;
            _g = gradminus;
            _E = -logpminus;
          }
        } else {
          double logpplus = leapfrog(_model, _z, xplus, mplus, gradplus,
                                     _epsilon);
          H = -0.5 * dot_self(mplus) + logpplus;
          if (_rand_uniform_01() < exp(H - totalweight)) {
//           if (u < H) {
//           if ((u < H) && (_rand_uniform_01() < 0.5)) {
            _x = xplus;
            _g = gradplus;
            _E = -logpplus;
          }
        }
//         double temp = exp(H - _H0);
//         _meanacceptprob += temp > 1 ? 1 : temp;
        _meanacceptprob += H - _H0;
//         fprintf(stderr, "0: meanacceptprob = %f, H = %f, H0 = %f\n",
//                 _meanacceptprob, H, _H0);
        ++_nfevals;
        totalweight = logSumExp(totalweight, H);
        int criterion = computeCriterion(xplus, xminus, mplus, mminus);
//         criterion &= nvalid == 2;

        std::vector<double> newsample, newgrad, tempx, tempm, tempg;
        double newlogp = -1;
        double newweight = -std::numeric_limits<double>::infinity();
        while (criterion) {
//           fprintf(stderr, "doubling. depth = %d\n", depth);
          direction = _rand_uniform_01() < 0.5;
          if (direction == 0)
            recurse(direction, xminus, mminus, gradminus, depth, _epsilon, 
                    _model, _z, _rand_uniform_01, newsample, newgrad,
                    newlogp, xminus, tempx, mminus, tempm, gradminus, tempg,
                    criterion, newweight);
          else
            recurse(direction, xplus, mplus, gradplus, depth, _epsilon,
                    _model, _z, _rand_uniform_01, newsample, newgrad,
                    newlogp, tempx, xplus, tempm, mplus, tempg, gradplus,
                    criterion, newweight);
          if (!criterion)
            newweight = -std::numeric_limits<double>::infinity();
          double oldweight = totalweight;
          totalweight = logSumExp(oldweight, newweight);
          criterion = computeCriterion(xplus, xminus, mplus, mminus);
//           criterion &= nvalid == pow(2, depth+1);
//           if (_rand_uniform_01() < float(nvalid2) / float(nvalid1 + nvalid2)) {
//           if (_rand_uniform_01() < float(nvalid2) / (1e-100+float(nvalid1))) {
          if (_rand_uniform_01() < exp(newweight - oldweight)) {
            _x = newsample;
            _g = newgrad;
            _E = -newlogp;
          }
          ++depth;
        }

        if (_nsamples < _adapttime) {
          int nsteps = (1 << depth) - 1;
          fprintf(stderr, "meanacceptprob = %f, nsteps = %d\n", 
                  _meanacceptprob / nsteps, nsteps);
          if (_meanacceptprob != _meanacceptprob)
            _meanacceptprob = -1e100;
          double eta = pow(_nsamples + 2, -0.75);
          _meanacceptprob /= nsteps;
//           const double delta = 0.65;
          double g = _meanacceptprob - log(_delta);
          const double maxg = 2;
          g = g > maxg ? maxg : g;
          g = g < -maxg ? -maxg : g;
//           double g = _meanacceptprob - 0.65;
//           double g = _meanacceptprob - _delta;
          fprintf(stderr, "eta = %f, g = %f\n", eta, g);
          _epsilon *= exp(eta * g);
          fprintf(stderr, "%d: epsilon = %.4f\n", _nsamples, _epsilon);
        }
        ++_nsamples;

	mcmc::sample s(_x,_z,-_E);
	return s;
      }

      void recurse(int direction, std::vector<double>& x,
                          std::vector<double>& m,
                          std::vector<double>& grad,
                          int depth, double epsilon,
                          mcmc::prob_grad& model, std::vector<unsigned int>& z,
                          boost::uniform_01<boost::mt19937&>& rand01,
                          std::vector<double>& newsample, 
                          std::vector<double>& newgrad, 
                          double& newlogp, std::vector<double>& xminus,
                          std::vector<double>& xplus,
                          std::vector<double>& mminus,
                          std::vector<double>& mplus,
                          std::vector<double>& gradminus,
                          std::vector<double>& gradplus,
                          int& criterion, double& newweight) {
        if (depth == 1) { // base case
          double logpplus, logpminus, Hplus, Hminus;
          if (direction == 0) {
            xplus = x;
            mplus = m;
            gradplus = grad;
            logpplus = leapfrog(model, z, xplus, mplus, gradplus, -epsilon);
            Hplus = logpplus - 0.5 * dot_self(mplus);
            xminus = xplus;
            mminus = mplus;
            gradminus = gradplus;
            logpminus = leapfrog(model, z, xminus, mminus, gradminus,-epsilon);
            Hminus = logpminus - 0.5 * dot_self(mminus);
          } else {
            xminus = x;
            mminus = m;
            gradminus = grad;
            logpminus = leapfrog(model, z, xminus, mminus, gradminus, epsilon);
            Hminus = logpminus - 0.5 * dot_self(mminus);
            xplus = xminus;
            mplus = mminus;
            gradplus = gradminus;
            logpplus = leapfrog(model, z, xplus, mplus, gradplus, epsilon);
            Hplus = logpplus - 0.5 * dot_self(mplus);
          }
          _nfevals += 2;
//           if ((u < Hplus) && ((rand01() < 0.5) || (u > Hminus))) {
          if (rand01() < expRatio(Hplus, Hminus)) {
            newsample = xplus;
            newgrad = gradplus;
            newlogp = logpplus;
          } else {
            newsample = xminus;
            newgrad = gradminus;
            newlogp = logpminus;
          }
          double temp = exp(Hplus - _H0);
          _meanacceptprob += Hplus - _H0;
//           _meanacceptprob += temp > 1 ? 1 : temp;
//         fprintf(stderr, "%d: meanacceptprob = %f, H = %f, H0 = %f\n",
//                 _nfevals, _meanacceptprob, Hplus, _H0);
          temp = exp(Hminus - _H0);
          _meanacceptprob += Hplus - _H0;
//           _meanacceptprob += temp > 1 ? 1 : temp;
//         fprintf(stderr, "%d: meanacceptprob = %f, H = %f, H0 = %f\n",
//                 _nfevals, _meanacceptprob, Hminus, _H0);
//         fprintf(stderr, "%d: distance of xminus to lastx = %f, logp = %f\n",
//                 _nfevals, dist(xminus, _lastx), logpminus);
          newweight = logSumExp(Hminus, Hplus);
        } else { // depth > 1
          std::vector<double> newsample1, newgrad1, newsample2, newgrad2,
            tempx, tempm, tempg;
          double newlogp1, newlogp2, weight1, weight2;
          int criterion1, criterion2;
          recurse(direction, x, m, grad, depth-1, epsilon,
                  model, z, rand01, newsample1, newgrad1, newlogp1,
                  xminus, xplus, mminus, mplus, gradminus, gradplus,
                  criterion1, weight1);
          if (direction == 0)
            recurse(direction, xminus, mminus, gradminus, depth-1, epsilon,
                    model, z, rand01, newsample2, newgrad2, newlogp2,
                    xminus, tempx, mminus, tempm, gradminus, tempg,
                    criterion2, weight2);
          else
            recurse(direction, xplus, mplus, gradplus, depth-1, epsilon,
                    model, z, rand01, newsample2, newgrad2, newlogp2,
                    tempx, xplus, tempm, mplus, tempg, gradplus,
                    criterion2, weight2);

          if (!criterion1)
            weight1 = -std::numeric_limits<double>::infinity();
          if (!criterion2)
            weight2 = -std::numeric_limits<double>::infinity();
//           if (rand01() < float(nvalid1) / float(nvalid1 + nvalid2)) {
          if (rand01() < expRatio(weight1, weight2)) {
            newsample = newsample1;
            newgrad = newgrad1;
            newlogp = newlogp1;
          } else {
            newsample = newsample2;
            newgrad = newgrad2;
            newlogp = newlogp2; 
          }
          newweight = logSumExp(weight1, weight2);
        }

        criterion = computeCriterion(xplus, xminus, mplus, mminus);
//         criterion &= nvalid == pow(2, depth);
      }

    };

  }

}

#endif

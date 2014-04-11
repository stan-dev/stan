#ifndef __STAN__OPTIMIZATION__BFGS_HPP__
#define __STAN__OPTIMIZATION__BFGS_HPP__

#include <cmath>
#include <cstdlib>
#include <string>
#include <limits>

#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/circular_buffer.hpp>

#include <stan/math/matrix/Eigen.hpp>
#include <stan/model/util.hpp>

namespace stan {
  namespace optimization {
    /**
     * Find the minima in an interval [loX, hiX] of a cubic function which
     * interpolates the points, function values and gradients provided.
     *
     * Implicitly, this function constructs an interpolating polynomial
     *     g(x) = a_3 x^3 + a_2 x^2 + a_1 x + a_0
     * such that g(0) = 0, g(x1) = f1, g'(0) = df0, g'(x1) = df1 where 
     *     g'(x) = 3 a_3 x^2 + 2 a_2 x + a_1 
     * is the derivative of g(x).  It then computes the roots of g'(x) and
     * finds the minimal value of g(x) on the interval [loX,hiX] including
     * the end points.
     *
     * This function implements the full parameter version of CubicInterp(). 
     *
     * @param df0 First derivative value, f'(x0)
     * @param x1 Second point
     * @param f1 Second function value, f(x1)
     * @param df1 Second derivative value, f'(x1)
     * @param loX Lower bound on the interval of solutions
     * @param hiX Upper bound on the interval of solutions
     **/
    template<typename Scalar>
    Scalar CubicInterp(const Scalar &df0,
                       const Scalar &x1, const Scalar &f1, const Scalar &df1,
                       const Scalar &loX, const Scalar &hiX)
    {
      const Scalar c3((-12*f1 + 6*x1*(df0 + df1))/(x1*x1*x1));
      const Scalar c2(-(4*df0 + 2*df1)/x1 + 6*f1/(x1*x1));
      const Scalar &c1(df0);
        
      const Scalar t_s = std::sqrt(c2*c2 - 2.0*c1*c3);
      const Scalar s1 = - (c2 + t_s)/c3;
      const Scalar s2 = - (c2 - t_s)/c3;
        
      Scalar tmpF;
      Scalar minF, minX;
        
      // Check value at lower bound
      minF = loX*(loX*(loX*c3/3.0 + c2)/2.0 + c1);
      minX = loX;
        
      // Check value at upper bound
      tmpF = hiX*(hiX*(hiX*c3/3.0 + c2)/2.0 + c1);
      if (tmpF < minF) {
        minF = tmpF;
        minX = hiX;
      }
        
      // Check value of first root
      if (loX < s1 && s1 < hiX) {
        tmpF = s1*(s1*(s1*c3/3.0 + c2)/2.0 + c1);
        if (tmpF < minF) {
          minF = tmpF;
          minX = s1;
        }
      }
        
      // Check value of second root
      if (loX < s2 && s2 < hiX) {
        tmpF = s2*(s2*(s2*c3/3.0 + c2)/2.0 + c1);
        if (tmpF < minF) {
          minF = tmpF;
          minX = s2;
        }
      }
        
      return minX;
    }

    /**
     * Find the minima in an interval [loX, hiX] of a cubic function which
     * interpolates the points, function values and gradients provided.
     *
     * Implicitly, this function constructs an interpolating polynomial
     *     g(x) = a_3 x^3 + a_2 x^2 + a_1 x + a_0
     * such that g(x0) = f0, g(x1) = f1, g'(x0) = df0, g'(x1) = df1 where 
     *     g'(x) = 3 a_3 x^2 + 2 a_2 x + a_1 
     * is the derivative of g(x).  It then computes the roots of g'(x) and
     * finds the minimal value of g(x) on the interval [loX,hiX] including
     * the end points.
     *
     * @param x0 First point
     * @param f0 First function value, f(x0)
     * @param df0 First derivative value, f'(x0)
     * @param x1 Second point
     * @param f1 Second function value, f(x1)
     * @param df1 Second derivative value, f'(x1)
     * @param loX Lower bound on the interval of solutions
     * @param hiX Upper bound on the interval of solutions
     **/
    template<typename Scalar>
    Scalar CubicInterp(const Scalar &x0, const Scalar &f0, const Scalar &df0,
                       const Scalar &x1, const Scalar &f1, const Scalar &df1,
                       const Scalar &loX, const Scalar &hiX)
    {
      return x0 + CubicInterp(df0,x1-x0,f1-f0,df1,loX-x0,hiX-x0);
    }

    /**
     * A utility function for implementing WolfeLineSearch()
     **/
    template<typename FunctorType, typename Scalar, typename XType>
    int WolfLSZoom(Scalar &alpha, XType &newX, Scalar &newF, XType &newDF,
                   FunctorType &func,
                   const XType &x, const Scalar &f, const Scalar &dfp,
                   const Scalar &c1dfp, const Scalar &c2dfp, const XType &p,
                   Scalar alo, Scalar aloF, Scalar aloDFp,
                   Scalar ahi, Scalar ahiF, Scalar ahiDFp,
                   const Scalar &min_range)
    {
      Scalar d1, d2, newDFp;
      int itNum(0);
 
      while (1) {
        itNum++;
          
        if (std::fabs(alo-ahi) < min_range)
          return 1;
          
        if (itNum%5 == 0) {
          alpha = 0.5*(alo+ahi);
        }
        else {
          // Perform cubic interpolation to determine next point to try
          d1 = aloDFp + ahiDFp - 3*(aloF-ahiF)/(alo-ahi);
          d2 = std::sqrt(d1*d1 - aloDFp*ahiDFp);
          if (ahi < alo)
            d2 = -d2;
          alpha = ahi - (ahi - alo)*(ahiDFp + d2 - d1)/(ahiDFp - aloDFp + 2*d2);
          if (!boost::math::isfinite(alpha) ||
              alpha < std::min(alo,ahi)+0.01*std::fabs(alo-ahi) ||
              alpha > std::max(alo,ahi)-0.01*std::fabs(alo-ahi))
            alpha = 0.5*(alo+ahi);
        }

        newX = x + alpha*p;
        while (func(newX,newF,newDF)) {
          alpha = 0.5*(alpha+std::min(alo,ahi));
          if (std::fabs(std::min(alo,ahi)-alpha) < min_range)
            return 1;
          newX = x + alpha*p;
        }
        newDFp = newDF.dot(p);
        if (newF > (f + alpha*c1dfp) || newF >= aloF) {
          ahi = alpha;
          ahiF = newF;
          ahiDFp = newDFp;
        }
        else {
          if (std::fabs(newDFp) <= -c2dfp)
            break;
          if (newDFp*(ahi-alo) >= 0) {
            ahi = alo;
            ahiF = aloF;
            ahiDFp = aloDFp;
          }
          alo = alpha;
          aloF = newF;
          aloDFp = newDFp;
        }
      }
      return 0;
    }

    /**
     * Perform a line search which finds an approximate solution to:
     * \f[
     *       \min_\alpha f(x_0 + \alpha p)
     * \f]
     * satisfying the strong Wolfe conditions:
     *  1) \f$ f(x_0 + \alpha p) \leq f(x_0) + c_1 \alpha p^T g(x_0) \f$
     *  2) \f$ \vert p^T g(x_0 + \alpha p) \vert \leq c_2 \vert p^T g(x_0) \vert \f$
     * where \f$g(x) = \frac{\partial f}{\partial x}\f$ is the gradient of f(x).
     *
     * @tparam FunctorType A type which supports being called as 
     *        ret = func(x,f,g)
     * where x is the input point, f and g are the function value and
     * gradient at x and ret is non-zero if function evaluation fails.
     *
     * @param func Function which is being minimized.
     *
     * @param alpha First value of \f$ \alpha \f$ to try.  Upon return this 
     * contains the final value of the \f$ \alpha \f$.
     *
     * @param x1 Final point, equal to \f$ x_0 + \alpha p \f$.
     *
     * @param f1 Final point function value, equal to \f$ f(x_0 + \alpha p) \f$.
     *
     * @param gradx1 Final point gradient, equal to \f$ g(x_0 + \alpha p) \f$.
     *
     * @param p Search direction.  It is assumed to be a descent direction such
     * that \f$ p^T g(x_0) < 0 \f$.
     *
     * @param x0 Value of starting point, \f$ x_0 \f$.
     *
     * @param f0 Value of function at starting point, \f$ f(x_0) \f$.
     *
     * @param gradx0 Value of function gradient at starting point, \f$ g(x_0) \f$.
     *
     * @param c1 Parameter of the Wolfe conditions. \f$ 0 < c_1 < c_2 < 1 \f$
     * Typically c1 = 1e-4.
     *
     * @param c2 Parameter of the Wolfe conditions. \f$ 0 < c_1 < c_2 < 1 \f$
     * Typically c2 = 0.9.
     *
     * @param minAlpha Smallest allowable step-size. 
     *
     * @return Returns zero on success, non-zero otherwise.
     **/
    template<typename FunctorType, typename Scalar, typename XType>
    int WolfeLineSearch(FunctorType &func,
                        Scalar &alpha,
                        XType &x1, Scalar &f1, XType &gradx1,
                        const XType &p,
                        const XType &x0, const Scalar &f0, const XType &gradx0,
                        const Scalar &c1, const Scalar &c2,
                        const Scalar &minAlpha)
    {
      const Scalar dfp(gradx0.dot(p));
      const Scalar c1dfp(c1*dfp);
      const Scalar c2dfp(c2*dfp);

      Scalar alpha0(minAlpha);
      Scalar alpha1(alpha);

      Scalar prevF(f0);
      XType prevDF(gradx0);
      Scalar prevDFp(dfp);
      Scalar newDFp;

      int retCode = 0, nits = 0, ret;
        
      while (1) {
        x1.noalias() = x0 + alpha1*p;
        ret = func(x1,f1,gradx1);
        if (ret!=0) {
          alpha1 = 0.5*(alpha0+alpha1);
          continue;
        }
        newDFp = gradx1.dot(p);
        if ((f1 > f0 + alpha*c1dfp) || (f1 >= prevF && nits > 0)) {
          retCode = WolfLSZoom(alpha, x1, f1, gradx1,
                               func,
                               x0, f0, dfp,
                               c1dfp, c2dfp, p,
                               alpha0, prevF, prevDFp,
                               alpha1, f1, newDFp,
                               1e-16);
          break;
        }
        if (std::fabs(newDFp) <= -c2dfp) {
          alpha = alpha1;
          break;
        }
        if (newDFp >= 0) {
          retCode = WolfLSZoom(alpha, x1, f1, gradx1,
                               func,
                               x0, f0, dfp,
                               c1dfp, c2dfp, p,
                               alpha1, f1, newDFp,
                               alpha0, prevF, prevDFp,
                               1e-16);
          break;
        }
          
        alpha0 = alpha1;
        prevF = f1;
        std::swap(prevDF,gradx1);
        prevDFp = newDFp;

        alpha1 *= 10.0;
          
        nits++;
      }
      return retCode;
    }

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
    
    template<typename Scalar = double>
    class ConvergenceOptions {
    public:
      ConvergenceOptions() {
        maxIts = 10000;
        fScale = 1.0;

        tolAbsX = 1e-8;
        tolAbsF = 1e-8;
        tolAbsGrad = 1e-8;

        tolRelF = 1e+7;
        tolRelGrad = 1e+3;
      }
      size_t maxIts;
      Scalar tolAbsX;
      Scalar tolAbsF;
      Scalar tolRelF;
      Scalar fScale;
      Scalar tolAbsGrad;
      Scalar tolRelGrad;
    };

    template<typename Scalar = double>
    class LSOptions {
    public:
      LSOptions() {
        c1 = 1e-4;
        c2 = 0.9;
        minAlpha = 1e-12;
        alpha0 = 1e-3;
      }
      Scalar c1;
      Scalar c2;
      Scalar alpha0;
      Scalar minAlpha;
    };


    /**
     * Implement a limited memory version of the BFGS update.  This
     * class maintains a circular buffer of inverse Hessian updates
     * which can be applied to compute the search direction.
     **/
    template<typename Scalar = double,
             int DimAtCompile = Eigen::Dynamic>
    class LBFGSUpdate {
    public:
      typedef Eigen::Matrix<Scalar,DimAtCompile,1> VectorT;
      typedef Eigen::Matrix<Scalar,DimAtCompile,DimAtCompile> HessianT;
      typedef boost::tuple<Scalar,VectorT,VectorT> UpdateT;

      LBFGSUpdate(size_t L = 5) : _buf(L) {}

      /**
       * Set the number of inverse Hessian updates to keep.
       *
       * @param L New size of buffer.
       **/
      void set_history_size(size_t L) {
        _buf.rset_capacity(L);
      }

      /**
       * Add a new set of update vectors to the history.
       *
       * @param yk Difference between the current and previous gradient vector.
       * @param sk Difference between the current and previous state vector.
       * @param reset Whether to reset the approximation, forgetting about previous values.
       * @return In the case of a reset, returns the optimal scaling of the initial Hessian
       * approximation which is useful for predicting step-sizes.
       **/
      inline Scalar update(const VectorT &yk, const VectorT &sk, bool reset = false) {
        Scalar skyk = yk.dot(sk);

        Scalar B0fact;
        if (reset) {
          B0fact = yk.squaredNorm()/skyk;
          _buf.clear();
        }
        else {
          B0fact = 1.0;
        }

        // New updates are pushed to the "back" of the circular buffer
        Scalar invskyk = 1.0/skyk;
        _gammak = skyk/yk.squaredNorm();
        _buf.push_back();
        _buf.back() = boost::tie(invskyk,yk,sk);

        return B0fact;
      }

      /**
       * Compute the search direction based on the current (inverse) Hessian
       * approximation and given gradient.
       *
       * @param[out] pk The negative product of the inverse Hessian and gradient
       * direction gk.
       * @param[in] gk Gradient direction.
       **/
      inline void search_direction(VectorT &pk, const VectorT &gk) const {
        std::vector<Scalar> alphas(_buf.size());
        typename boost::circular_buffer<UpdateT>::const_reverse_iterator buf_rit;
        typename boost::circular_buffer<UpdateT>::const_iterator buf_it;
        typename std::vector<Scalar>::const_iterator alpha_it;
        typename std::vector<Scalar>::reverse_iterator alpha_rit;

        pk.noalias() = -gk;
        for (buf_rit = _buf.rbegin(), alpha_rit = alphas.rbegin();
             buf_rit != _buf.rend();
             buf_rit++, alpha_rit++) {
          Scalar alpha;
          const Scalar &rhoi(boost::get<0>(*buf_rit));
          const VectorT &yi(boost::get<1>(*buf_rit));
          const VectorT &si(boost::get<2>(*buf_rit));

          alpha = rhoi * si.dot(pk);
          pk -= alpha * yi;
          *alpha_rit = alpha;
        }
        pk *= _gammak;
        for (buf_it = _buf.begin(), alpha_it = alphas.begin();
             buf_it != _buf.end();
             buf_it++, alpha_it++) {
          Scalar beta;
          const Scalar &rhoi(boost::get<0>(*buf_it));
          const VectorT &yi(boost::get<1>(*buf_it));
          const VectorT &si(boost::get<2>(*buf_it));

          beta = rhoi*yi.dot(pk);
          pk += (*alpha_it - beta)*si;
        }
      }

    private:
      boost::circular_buffer<UpdateT> _buf;
      Scalar _gammak;
    };

    template<typename Scalar = double,
             int DimAtCompile = Eigen::Dynamic>
    class BFGSUpdate_HInv {
    public:
      typedef Eigen::Matrix<Scalar,DimAtCompile,1> VectorT;
      typedef Eigen::Matrix<Scalar,DimAtCompile,DimAtCompile> HessianT;

      /**
       * Update the inverse Hessian approximation.
       *
       * @param yk Difference between the current and previous gradient vector.
       * @param sk Difference between the current and previous state vector.
       * @param reset Whether to reset the approximation, forgetting about previous values.
       * @return In the case of a reset, returns the optimal scaling of the initial Hessian
       * approximation which is useful for predicting step-sizes.
       **/
      inline Scalar update(const VectorT &yk, const VectorT &sk, bool reset = false) {
        Scalar rhok, skyk, B0fact;
        HessianT Hupd;

        skyk = yk.dot(sk);
        rhok = 1.0/skyk;

        Hupd.noalias() = HessianT::Identity(yk.size(),yk.size()) 
                                        - rhok*sk*yk.transpose();
        if (reset) {
          B0fact = yk.squaredNorm()/skyk;
          _Hk.noalias() = ((1.0/B0fact)*Hupd)*Hupd.transpose();
        }
        else {
          B0fact = 1.0;
          _Hk = Hupd*_Hk*Hupd.transpose();
        }
        _Hk.noalias() += rhok*sk*sk.transpose();
        return B0fact;
      }

      /**
       * Compute the search direction based on the current (inverse) Hessian
       * approximation and given gradient.
       *
       * @param[out] pk The negative product of the inverse Hessian and gradient direction gk.
       * @param[in] gk Gradient direction.
       **/
      inline void search_direction(VectorT &pk, const VectorT &gk) const {
        pk.noalias() = -(_Hk*gk);
      }

    private:
      HessianT _Hk;
    };

    template<typename FunctorType, typename Scalar = double,
             int DimAtCompile = Eigen::Dynamic,
             typename QNUpdateType = LBFGSUpdate<Scalar,DimAtCompile> >
    class BFGSMinimizer {
    public:
      typedef Eigen::Matrix<Scalar,DimAtCompile,1> VectorT;
      typedef Eigen::Matrix<Scalar,DimAtCompile,DimAtCompile> HessianT;
      
    protected:
      FunctorType &_func;
      VectorT _gk, _gk_1, _xk_1, _xk, _pk, _pk_1;
      Scalar _fk, _fk_1, _alphak_1;
      Scalar _alpha, _alpha0;
      size_t _itNum;
      std::string _note;
      QNUpdateType _qn;
      
    public:
      LSOptions<Scalar> _ls_opts;
      ConvergenceOptions<Scalar> _conv_opts;

      QNUpdateType &get_qnupdate() { return _qn; }
      const QNUpdateType &get_qnupdate() const { return _qn; }
      
      const Scalar &curr_f() const { return _fk; }
      const VectorT &curr_x() const { return _xk; }
      const VectorT &curr_g() const { return _gk; }
      const VectorT &curr_p() const { return _pk; }
      
      const Scalar &prev_f() const { return _fk_1; }
      const VectorT &prev_x() const { return _xk_1; }
      const VectorT &prev_g() const { return _gk_1; }
      const VectorT &prev_p() const { return _pk_1; }
      Scalar prev_step_size() const { return _pk_1.norm()*_alphak_1; }

      inline Scalar rel_grad_norm() const {
        return -_pk.dot(_gk) / std::max(std::fabs(_fk),_conv_opts.fScale);
      }
      inline Scalar rel_obj_decrease() const { 
        return std::fabs(_fk_1 - _fk) / std::max(std::fabs(_fk_1),std::max(std::fabs(_fk),_conv_opts.fScale));
      }

      const Scalar &alpha0() const { return _alpha0; }
      const Scalar &alpha() const { return _alpha; }
      const size_t iter_num() const { return _itNum; }
      
      const std::string &note() const { return _note; }
      
      std::string get_code_string(int retCode) {
        switch(retCode) {
          case TERM_SUCCESS:
            return std::string("Successful step completed");
          case TERM_ABSF:
            return std::string("Convergence detected: absolute change in objective function was below tolerance");
          case TERM_RELF:
            return std::string("Convergence detected: relative change in objective function was below tolerance");
          case TERM_ABSGRAD:
            return std::string("Convergence detected: gradient norm is below tolerance");
          case TERM_RELGRAD:
            return std::string("Convergence detected: relative gradient magnitude is below tolerance");
          case TERM_ABSX:
            return std::string("Convergence detected: absolute parameter change was below tolerance");
          case TERM_MAXIT:
            return std::string("Maximum number of iterations hit, may not be at an optima");
          case TERM_LSFAIL:
            return std::string("Line search failed to achieve a sufficient decrease, no more progress can be made");
          default:
            return std::string("Unknown termination code");
        }
      }
      
      BFGSMinimizer(FunctorType &f) : _func(f) { }
      
      void initialize(const VectorT &x0) {
        int ret;
        _xk = x0;
        ret = _func(_xk,_fk,_gk);
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
        }
        else {
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
            _alpha0 = _alpha = std::min(1.0,
                                        1.01*CubicInterp(_gk_1.dot(_pk_1),
                                                         _alphak_1,
                                                         _fk - _fk_1,
                                                         _gk.dot(_pk_1),
                                                         _ls_opts.minAlpha, 1.0));
          }
          else {
            // On the first step (or, after a reset) use the default step size
            _alpha0 = _alpha = _ls_opts.alpha0;
          }

          // Perform the line search.  If successful, the results are in the 
          // variables: _xk_1, _fk_1 and _gk_1.
          retCode = WolfeLineSearch(_func, _alpha, _xk_1, _fk_1, _gk_1,
                                    _pk, _xk, _fk, _gk,
                                    _ls_opts.c1, _ls_opts.c2, 
                                    _ls_opts.minAlpha);
          if (retCode) {
            // Line search failed...
            if (resetB) {
              // did a Hessian reset and it still failed, and nothing left to try
              retCode = TERM_LSFAIL;
              return retCode;
            }
            else {
              // try resetting the Hessian approximation
              resetB = 2;
              _note += "LS failed, Hessian reset";
              continue;
            }
          }
          else {
            break;
          }
        }

        // Swap things so that k is the most recent iterate
        std::swap(_fk,_fk_1);
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
          Scalar B0fact = _qn.update(yk,sk,true);
          _pk_1 /= B0fact;
          _alphak_1 = _alpha*B0fact;
        }
        else {
          _qn.update(yk,sk);
          _alphak_1 = _alpha;
        }


        // Compute search direction for next step
        _qn.search_direction(_pk,_gk);

        // Check for convergence
        if (std::fabs(_fk_1 - _fk) < _conv_opts.tolAbsF) {
          retCode = TERM_ABSF; // Objective function improvement wasn't sufficient
        }
        else if (gradNorm < _conv_opts.tolAbsGrad) {
          retCode = TERM_ABSGRAD; // Gradient norm was below threshold
        }
        else if (stepNorm < _conv_opts.tolAbsX) {
          retCode = TERM_ABSX; // Change in x was too small
        }
        else if (_itNum >= _conv_opts.maxIts) {
          retCode = TERM_MAXIT; // Max number of iterations hit
        }
        else if (rel_obj_decrease() < _conv_opts.tolRelF*std::numeric_limits<Scalar>::epsilon()) {
          retCode = TERM_RELF; // Relative improvement in objective function wasn't sufficient
        }
        else if (rel_grad_norm() < _conv_opts.tolRelGrad*std::numeric_limits<Scalar>::epsilon()) {
          retCode = TERM_RELGRAD; // Relative gradient norm was below threshold
        }
        else {
          retCode = TERM_SUCCESS; // Step was successful more progress to be made
        }

        return retCode;
      }
      
      int minimize(VectorT &x0) {
        int retcode;
        initialize(x0);
        while (!(retcode = step()));
        x0 = _xk;
        return retcode;
      }
    };

                                

    template <typename M>
    double lp_no_jacobian(const M& model,
                          std::vector<double>& params_r,
                          std::vector<int>& params_i,
                          std::ostream* o = 0) {
      // FIXME: is this supposed to return the log probability from the model?
      return 0;
    }

    template <class M>
    class ModelAdaptor {
    private:
      M& _model;
      std::vector<int> _params_i;
      std::ostream* _msgs;
      std::vector<double> _x, _g;
      size_t _fevals;

    public:
      ModelAdaptor(M& model,
                   const std::vector<int>& params_i,
                   std::ostream* msgs)
      : _model(model), _params_i(params_i), _msgs(msgs), _fevals(0) {}
                  
      size_t fevals() const { return _fevals; }
      int operator()(const Eigen::Matrix<double,Eigen::Dynamic,1> &x, 
                     double &f) {
        _x.resize(x.size());
        for (Eigen::Matrix<double,Eigen::Dynamic,1>::size_type i = 0; 
             i < x.size(); i++)
          _x[i] = x[i];

        try {
          f = - _model.template log_prob<false,false>(_x,_params_i,
                                                      _msgs);
        } catch (const std::exception& e) {
          if (_msgs)
            (*_msgs) << e.what() << std::endl;
          return 1;
        }

        if (boost::math::isfinite(f))
          return 0;
        else {
          if (_msgs)
            *_msgs << "Error evaluating model log probability: " 
                      "Non-finite function evaluation." << std::endl;
          return 2;
        }
      }
      int operator()(const Eigen::Matrix<double,Eigen::Dynamic,1> &x, 
                     double &f, Eigen::Matrix<double,Eigen::Dynamic,1> &g) {
        _x.resize(x.size());
        for (Eigen::Matrix<double,Eigen::Dynamic,1>::size_type i = 0; 
             i < x.size(); i++)
          _x[i] = x[i];
        
        _fevals++;

        try {
          f = - stan::model::log_prob_grad<true,false>(_model,
                                                       _x, _params_i,
                                                       _g, _msgs);
        } catch (const std::exception& e) {
          if (_msgs)
            (*_msgs) << e.what() << std::endl;
          return 1;
        }

        g.resize(_g.size());
        for (size_t i = 0; i < _g.size(); i++) {
          if (!boost::math::isfinite(_g[i])) {
            if (_msgs)
              *_msgs << "Error evaluating model log probability: " 
                                 "Non-finite gradient." << std::endl;
            return 3;
          }
          g[i] = -_g[i];
        }

        if (boost::math::isfinite(f))
          return 0;
        else {
          if (_msgs)
            *_msgs << "Error evaluating model log probability: " 
                               "Non-finite function evaluation." 
                   << std::endl;
          return 2;
        }
      }
      int df(const Eigen::Matrix<double,Eigen::Dynamic,1> &x, 
             Eigen::Matrix<double,Eigen::Dynamic,1> &g) {
        double f;
        return (*this)(x,f,g);
      }
    };
    
    template<typename M, typename Scalar = double,
             int DimAtCompile = Eigen::Dynamic,
             typename QNUpdateType = LBFGSUpdate<Scalar,DimAtCompile> >
    class BFGSLineSearch : public BFGSMinimizer<ModelAdaptor<M>,Scalar,DimAtCompile,QNUpdateType> {
    private:
      ModelAdaptor<M> _adaptor;
    public:
      typedef BFGSMinimizer<ModelAdaptor<M>,Scalar,DimAtCompile,QNUpdateType> BFGSBase;
      typedef typename BFGSBase::VectorT vector_t;
      typedef typename vector_t::size_type idx_t;
      
      BFGSLineSearch(M& model,
                     const std::vector<double>& params_r,
                     const std::vector<int>& params_i,
                     std::ostream* msgs = 0)
        : BFGSBase(_adaptor),
          _adaptor(model,params_i,msgs)
      {

        Eigen::Matrix<double,Eigen::Dynamic,1> x;
        x.resize(params_r.size());
        for (size_t i = 0; i < params_r.size(); i++)
          x[i] = params_r[i];
        this->initialize(x);
      }

      size_t grad_evals() { return _adaptor.fevals(); }
      double logp() { return -(this->curr_f()); }
      double grad_norm() { return this->curr_g().norm(); }
      void grad(std::vector<double>& g) { 
        const vector_t &cg(this->curr_g());
        g.resize(cg.size());
        for (idx_t i = 0; i < cg.size(); i++)
          g[i] = -cg[i];
      }
      void params_r(std::vector<double>& x) {
        const vector_t &cx(this->curr_x());
        x.resize(cx.size());
        for (idx_t i = 0; i < cx.size(); i++)
          x[i] = cx[i];
      }
    };

  }

}

#endif

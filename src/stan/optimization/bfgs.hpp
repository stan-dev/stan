#ifndef __STAN__OPTIMIZATION__BFGS_HPP__
#define __STAN__OPTIMIZATION__BFGS_HPP__

#include <stan/model/prob_grad.hpp>
#include <cstdlib>
#include <cmath>

namespace stan {
  namespace optimization {
    namespace {
      template<typename FunctorType, typename Scalar, typename XType>
      int BTLineSearch(FunctorType &func,
                       Scalar &alpha, 
                       XType &x1, Scalar &f1, XType &gradx1,
                       const XType &p,
                       const XType &x0, const Scalar &f0, const XType &gradx0,
                       const Scalar &rho, const Scalar &c,
                       const Scalar &minAlpha)
      {
        const Scalar cdfdp(c*gradx0.dot(p));
        int ret;
        
        while (1) {
          x1 = x0 + alpha*p;
          ret = func(x1,f1);
          if (ret!=0 && f1 <= f0 + alpha*cdfdp)
            break;
          else
            alpha *= rho;
          
          if (alpha < minAlpha)
            return 1;
        }
        func.df(x1,gradx1);
        return 0;
      }
      
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
          
          if (std::abs(alo-ahi) < min_range)
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
            if (alpha < std::min(alo,ahi) || alpha > std::max(alo,ahi))
              alpha = 0.5*(alo+ahi);
          }
          
          newX = x + alpha*p;
          while (func(newX,newF,newDF)) {
            alpha = 0.5*(alpha+std::min(alo,ahi));
            if (std::abs(alo-alpha) < min_range)
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
            if (std::abs(newDFp) <= -c2dfp)
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
      
      template<typename FunctorType, typename Scalar, typename XType>
      int WolfeLineSearch(FunctorType &func,
                          Scalar &alpha,
                          XType &x1, Scalar &f1, XType &gradx1,
                          const XType &p,
                          const XType &x0, const Scalar &f0, const XType &gradx0,
                          const Scalar &c1, const Scalar &c2,
                          const Scalar &minAlpha, const Scalar &maxAlpha)
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
          if (std::abs(newDFp) <= -c2dfp) {
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
          
          alpha1 = std::min(1.25*alpha0,maxAlpha);
          
          nits++;
        }
        return retCode;
      }

      template<typename FunctorType, typename Scalar = double,
               int DimAtCompile = Eigen::Dynamic,
               int LineSearchMethod = 1>
      class BFGSMinimizer {
      public:
        typedef Eigen::Matrix<Scalar,DimAtCompile,1> VectorT;
        typedef Eigen::Matrix<Scalar,DimAtCompile,DimAtCompile> HessianT;
        
      protected:
        FunctorType &_func;
        Scalar _f0, _f1, _alpha;
        size_t _itNum;
        HessianT _B;
        Eigen::LDLT< HessianT > _ldlt;
        VectorT _gradx0, _gradx1, _x1, _x0;
        VectorT _s;
        
      public:
        const Scalar &curr_f() const { return _f0; }
        const VectorT &curr_x() const { return _x0; }
        const VectorT &curr_g() const { return _gradx0; }
        
        const HessianT &curr_H() const { return _B; }

        const Scalar &prev_f() const { return _f1; }
        const VectorT &prev_x() const { return _x1; }
        const VectorT &prev_g() const { return _gradx1; }

        const Scalar &step_size() const { return _alpha; }
        
        struct BFGSOptions {
          BFGSOptions() {
            maxIts = 10000;
            rho = 0.75;
            c1 = 1e-4;
            c2 = 0.9;
            minStep = 1e-16;
            minGradNorm = 1e-6;
            minAlpha = 1e-12;
            maxAlpha = 5.0;
            alpha0 = 1e-3;
          }
          size_t maxIts;
          Scalar rho;
          Scalar c1;
          Scalar c2;
          Scalar minStep;
          Scalar minGradNorm;
          Scalar alpha0;
          Scalar minAlpha;
          Scalar maxAlpha;
        } _opts;
        
        
        BFGSMinimizer(FunctorType &f) : _func(f) { }
        
        void initialize(const VectorT &x0) {
          _x0 = x0;
          _func(_x0,_f0,_gradx0);
          
          _itNum = 0;
        }
        
        int step() {
          Scalar gradNorm, thetak, skyk, skBksk;
          VectorT sk, yk, Bksk, rk;
          int retCode;
          bool resetB(false);
          
          _itNum++;
          
          if (_itNum == 1 || _ldlt.info() != Eigen::Success || _ldlt.isNegative()) {
            Scalar Bscale;
            resetB = true;
            if (_itNum == 1) {
              Bscale = 1.0/_opts.alpha0;
            }
            else {
              Bscale = _B.diagonal().maxCoeff();
            }
            _B.setIdentity(_x0.size(),_x0.size());
            _B *= Bscale;
            _ldlt.compute(_B);
            _s = -_gradx0/Bscale;
          }
          else {
            _s = -_ldlt.solve(_gradx0);
          }
          
          _alpha = 1.0;
          if (LineSearchMethod == 0) {
            retCode = BTLineSearch(_func, _alpha, _x1, _f1, _gradx1,
                                   _s, _x0, _f0, _gradx0, _opts.rho, 
                                   _opts.c1, _opts.minAlpha);
          }
          else if (LineSearchMethod == 1) {
            retCode = WolfeLineSearch(_func, _alpha, _x1, _f1, _gradx1,
                                      _s, _x0, _f0, _gradx0,
                                      _opts.c1, _opts.c2, 
                                      _opts.minAlpha, _opts.maxAlpha);
          }
          if (retCode) {
            // Line-search failed
            retCode = 10;
            return retCode;
          }
          std::swap(_f0,_f1);
          _x0.swap(_x1);
          _gradx0.swap(_gradx1);
          
          gradNorm = _gradx0.squaredNorm();
          sk.noalias() = _x0 - _x1;
          if (sk.array().abs().maxCoeff() <= _opts.minStep) {
            if (gradNorm <= _opts.minGradNorm) {
              retCode = 1;
            }
            else {
              retCode = 2;
            }
          }
          else {
            retCode = 0;
          }
          
          yk.noalias() = _gradx0 - _gradx1;
          skyk = yk.dot(sk);
          if (resetB) {
            Scalar B0fact = yk.squaredNorm()/skyk;
            _B.setIdentity(_x0.size(),_x0.size());
            _B *= B0fact;
            Bksk.noalias() = B0fact*sk;
          }
          else {
            Bksk.noalias() = _B*sk;
          }
          skBksk = sk.dot(Bksk);
          if (skyk >= 0.2*skBksk) {
            // Full update
            thetak = 1;
            rk = yk;
          }
          else {
            // Damped update (Proceedure 18.2)
            thetak = 0.8*skBksk/(skBksk - skyk);
            rk = thetak*yk + (1.0 - thetak)*Bksk;
          }
          _B.noalias() += rk*rk.transpose()/sk.dot(rk) - Bksk*Bksk.transpose()/skBksk;
          _ldlt.rankUpdate(rk,1.0/sk.dot(rk));
          _ldlt.rankUpdate(Bksk,-1.0/skBksk);
          
          return retCode;
        }
        
        int minimize(VectorT &x0) {
          int retcode;
          initialize(x0);
          while (!(retcode = step()));
          x0 = _x0;
          return retcode;
        }
      };
    }
    
    class ModelAdaptor {
    private:
      stan::model::prob_grad& _model;
      std::vector<int> _params_i;
      std::ostream* _output_stream;
      std::vector<double> _x, _g;

    public:
      ModelAdaptor(stan::model::prob_grad& model,
                   const std::vector<int>& params_i,
                   std::ostream* output_stream)
      : _model(model), _params_i(params_i), _output_stream(output_stream) {}
                   
      int operator()(const Eigen::Matrix<double,Eigen::Dynamic,1> &x, double &f) {
        _x.resize(x.size());
        for (size_t i = 0; i < x.size(); i++)
          _x[i] = x[i];

        try {
          f = -_model.log_prob(_x, _params_i, _output_stream);
        } catch (const std::exception& e) {
          return 1;
        }

        if (boost::math::isfinite(f))
          return 0;
        else
          return 2;
      }
      int operator()(const Eigen::Matrix<double,Eigen::Dynamic,1> &x, double &f, Eigen::Matrix<double,Eigen::Dynamic,1> &g) {
        _x.resize(x.size());
        for (size_t i = 0; i < x.size(); i++)
          _x[i] = x[i];

        try {
          f = -_model.grad_log_prob(_x, _params_i, _g, _output_stream);
        } catch (const std::exception& e) {
          return 1;
        }

        g.resize(_g.size());
        for (size_t i = 0; i < _g.size(); i++) {
          if (!boost::math::isfinite(_g[i]))
            return 3;
          g[i] = -_g[i];
        }

        if (boost::math::isfinite(f))
          return 0;
        else
          return 2;
      }
      int df(const Eigen::Matrix<double,Eigen::Dynamic,1> &x, Eigen::Matrix<double,Eigen::Dynamic,1> &g) {
        double f;
        return (*this)(x,f,g);
      }
    };
    
    class BFGSLineSearch {
    private:
      ModelAdaptor _adaptor;
      BFGSMinimizer<ModelAdaptor> _bfgsOpt;
      
    public:
      BFGSLineSearch(stan::model::prob_grad& model,
                     const std::vector<double>& params_r,
                     const std::vector<int>& params_i,
                     std::ostream* output_stream = 0)
      : _adaptor(model,params_i,output_stream),
        _bfgsOpt(_adaptor) {

        Eigen::Matrix<double,Eigen::Dynamic,1> x;
        x.resize(params_r.size());
        for (size_t i = 0; i < params_r.size(); i++)
          x[i] = params_r[i];
        _bfgsOpt.initialize(x);
      }

      double step_size() { return _bfgsOpt.step_size(); }
      double logp() { return -_bfgsOpt.curr_f(); }
      void grad(std::vector<double>& g) { 
        const BFGSMinimizer<ModelAdaptor>::VectorT &cg(_bfgsOpt.curr_g());
        g.resize(cg.size());
        for (size_t i = 0; i < cg.size(); i++)
          g[i] = -cg[i];
      }
      void params_r(std::vector<double>& x) {
        const BFGSMinimizer<ModelAdaptor>::VectorT &cx(_bfgsOpt.curr_g());
        x.resize(cx.size());
        for (size_t i = 0; i < cx.size(); i++)
          x[i] = cx[i];
      }

      int step() {
        return _bfgsOpt.step();
      }
    };

  }

}

#endif

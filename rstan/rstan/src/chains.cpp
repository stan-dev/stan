// #include <stan/mcmc/chains.hpp> 
#include <stan/math/matrix.hpp>
#include <stan/prob/autocorrelation.hpp>
#include <stan/prob/autocovariance.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random/additive_combine.hpp> // L'Ecuyer RNG
#include <boost/random/uniform_int_distribution.hpp>

#include <Rcpp.h>

namespace rstan {

  namespace { 
    /*
     * Wrap up the arguments for creating a sequence of indexes for permutation 
     * Rcpp::List and set the defaults if not available. 
     *
     * Arguments n and chains must be in the named list; others are optional. 
     *
     * <ul>
     * <li>  n: total length 
     * <li>  chains: number of chains 
     * <li>  seed: seed for RNG 
     * <li>  chain_id: chain id 
     * </ul>
     */ 
    class perm_args {
    private:
      int n, chains, chain_id;  
      unsigned int seed; 

      inline unsigned int sexp2seed(SEXP seed) { 
        if (TYPEOF(seed) == STRSXP)  
          return boost::lexical_cast<unsigned int>(Rcpp::as<std::string>(seed));
        return Rcpp::as<unsigned int>(seed); 
      }
  
    public:
      perm_args(Rcpp::List& lst) : chain_id(1) { 
        if (!lst.containsElementNamed("n")) 
          throw std::runtime_error("number of iterations kept (n) is not specified"); 
        n = Rcpp::as<int>(lst["n"]); 
  
        if (!lst.containsElementNamed("chains")) 
          throw std::runtime_error("number of chains is not specified"); 
        chains = Rcpp::as<int>(lst["chains"]);
  
        if (lst.containsElementNamed("chain_id")) 
          chain_id = Rcpp::as<int>(lst["chain_id"]); 
  
        if (lst.containsElementNamed("seed")) 
          seed = sexp2seed(lst["seed"]); 
        else 
          seed = std::time(0); 
      } 
  
      inline int get_n() const { return n; } 
      inline int get_chain_id() const { return chain_id; } 
      inline unsigned int get_seed() const { return seed; } 
      inline int get_chains() const { return chains; } 
  
      inline SEXP perm_args_to_rlist() const {
        Rcpp::List lst; 
        std::stringstream ss; 
        ss << seed; 
        lst["seed"] = ss.str();  
        lst["n"] = n; 
        lst["chain_id"] = chain_id;
        lst["chains"] = chains; 
        return lst; 
      }
    }; 

    /**
     * @param sim An R list that has element chains, n_flatnames, samples,
     *  n_save, warmup2, etc. In particular, 
     *  chains: number of chains. 
     *  n_flatnames: the total number of param in form of scalars.  
     *  n_save: sizes of saved iterations for all chains. 
     *  warmup2: simiar to n_save, but for warmup sizes. Note this warmup
     *  might be different from the warmup specified by the user for running
     *  the sampler because thinning is accounted in warmup2. 
     *  samples: A list for saved samples. Each element is a list for a chain. 
     *    Each chain is a list as well, every element of which is a chain of
     *    samples for a parameter. 
     *  
     */ 
    void validate_sim(SEXP sim) {
      std::vector<std::string> snames;
      snames.push_back("chains"); 
      snames.push_back("n_flatnames"); 
      snames.push_back("n_save");
      snames.push_back("warmup2");
      snames.push_back("samples");
      snames.push_back("permutation");
      Rcpp::List lst(sim); 
      std::vector<std::string> names = lst.names(); 
      
      for (std::vector<std::string>::const_iterator it = snames.begin(); 
           it != snames.end(); 
           ++it) { 
         if (std::find(names.begin(), names.end(), *it) == names.end()) {
           std::stringstream msg;
           msg << "the simulation results (sim) does not contain " << *it; 
           throw std::domain_error(msg.str()); 
         } 
      }
  
      unsigned int type = TYPEOF(lst["chains"]); 
      if (type != INTSXP &&  type != REALSXP) { 
        std::stringstream msg;
        msg << "wrong type of chains in sim; found " 
            << Rf_type2char(type) 
            << ", but INTSXP/REALSXP needed"; 
        throw std::domain_error(msg.str()); 
      } 
    } 
  
    unsigned int num_chains(SEXP sim) {
      Rcpp::List lst(sim); 
      return Rcpp::as<unsigned int>(lst["chains"]); 
    } 
  
    unsigned int num_params(SEXP sim) {
      Rcpp::List lst(sim); 
      return Rcpp::as<unsigned int>(lst["n_flatnames"]); 
    } 
    /** 
     *
     * @param k Chain index starting from 0
     * @param n Parameer index starting from 0
     */ 
    void get_kept_samples(SEXP sim, const size_t k, const size_t n, 
                          std::vector<double>& samples) {
      Rcpp::List lst(sim); 
      if (TYPEOF(lst["samples"]) != VECSXP) { 
        std::stringstream msg;
        msg << "sim$samples is not a list"; 
        throw std::domain_error(msg.str()); 
      } 
      Rcpp::List allsamples(static_cast<SEXP>(lst["samples"])); 
      Rcpp::IntegerVector n_save(static_cast<SEXP>(lst["n_save"])); 
      Rcpp::IntegerVector warmup2(static_cast<SEXP>(lst["warmup2"])); 
    
      Rcpp::List slst(static_cast<SEXP>(allsamples[k]));  // chain k
      Rcpp::NumericVector nv(static_cast<SEXP>(slst[n])); // parameter n  
      samples.assign(warmup2[k] + nv.begin(), nv.end()); 
    } 
  
    void validate_param_idx(SEXP sim, size_t n) {
      if (n < num_params(sim))
        return;
      std::stringstream msg;
      msg << "parameter index must be less than number of params"
          << "; found n=" << n;
      throw std::out_of_range(msg.str());
    }
  
    void validate_chain_idx(SEXP sim, unsigned int k) {
      unsigned int m = num_chains(sim); 
      if (k >= m) { 
        std::stringstream msg;
        msg << "chain must be less than number of chains."
            << "; num chains=" << m 
            << "; chain=" << k;
        throw std::out_of_range(msg.str());
      }
    }
  
    template <typename F>
    void apply_kept_samples(SEXP sim, size_t k,
                            size_t n,
                            F& f) {
      Rcpp::List lst(sim); 
      Rcpp::List allsamples(static_cast<SEXP>(lst["samples"])); 
      Rcpp::IntegerVector n_save(static_cast<SEXP>(lst["n_save"])); 
      Rcpp::IntegerVector warmup2(static_cast<SEXP>(lst["warmup2"])); 
  
      Rcpp::List slst(static_cast<SEXP>(allsamples[k]));  // chain k
      Rcpp::NumericVector nv(static_cast<SEXP>(slst[n])); // parameter n  
      // use int instead of size_t since these are R integers. 
      for (int i = warmup2[k]; i < n_save[k]; i++) {
        f(nv[i]); 
      } 
    }
  
    double get_chain_mean(SEXP sim, size_t k, size_t n) {
      using boost::accumulators::accumulator_set;
      using boost::accumulators::stats;
      using boost::accumulators::tag::mean;
      validate_chain_idx(sim,k);
      validate_param_idx(sim,n);
      accumulator_set<double, stats<mean> > acc;
      apply_kept_samples(sim, k,n,acc);
      return boost::accumulators::mean(acc);
    }
  
    /** 
     * Returns the autocovariance for the specified parameter in the
     * kept samples of the chain specified.
     * 
     * @param[in] k Chain index
     * @param[in] n Parameter index
     * @param[out] acov Autocovariances
     */
    void autocovariance(SEXP sim, const size_t k, const size_t n, 
                        std::vector<double>& acov) {
      std::vector<double> samples;
      get_kept_samples(sim,k,n,samples);
      stan::prob::autocovariance(samples, acov);
    }
  } 
} 

RcppExport SEXP effective_sample_size(SEXP sim, SEXP n_); 
RcppExport SEXP split_potential_scale_reduction(SEXP sim, SEXP n_); 
RcppExport SEXP seq_permutation(SEXP conf);  

/** 
 * Returns the effective sample size for the specified parameter
 * across all kept samples.
 *
 * The implementation matches BDA3's effective size description.
 * 
 * Current implementation takes the minimum number of samples
 * across chains as the number of samples per chain.
 *
 * @param[in] sim An R list containing simulated samples 
 *  and all other related information such as number of 
 *  iterations (warmup, etc), and parameter information.  
 * @param[in] n Parameter index
 * 
 * @return the effective sample size.
 */
// FIXME: reimplement using autocorrelation.
SEXP effective_sample_size(SEXP sim, SEXP n_) { 
  BEGIN_RCPP; 
  rstan::validate_sim(sim); 
  Rcpp::List lst(sim); 
  unsigned int n = Rcpp::as<unsigned int>(n_); 
  rstan::validate_param_idx(sim,n);
  unsigned int m(rstan::num_chains(sim)); 
  // need to generalize to each jagged samples per chain
  
  std::vector<unsigned int> ns_save = 
    Rcpp::as<std::vector<unsigned int> >(lst["n_save"]); 

  std::vector<unsigned int> ns_warmup2 = 
    Rcpp::as<std::vector<unsigned int> >(lst["warmup2"]); 

  std::vector<unsigned int> ns_kept(ns_save); 
  for (size_t i = 0; i < ns_kept.size(); i++) 
    ns_kept[i] -= ns_warmup2[i]; 

  unsigned int n_samples = ns_kept[0]; 
  for (size_t chain = 1; chain < m; chain++) {
    n_samples = std::min(n_samples, ns_kept[chain]); 
  }

  using std::vector;
  vector< vector<double> > acov;
  for (size_t chain = 0; chain < m; chain++) {
    vector<double> acov_chain;
    rstan::autocovariance(sim, chain, n, acov_chain);
    acov.push_back(acov_chain);
  }
  
  vector<double> chain_mean;
  vector<double> chain_var;
  for (size_t chain = 0; chain < m; chain++) {
    double n_kept_samples = ns_kept[chain]; 
    chain_mean.push_back(rstan::get_chain_mean(sim,chain,n));
    chain_var.push_back(acov[chain][0]*n_kept_samples/(n_kept_samples-1));
  }
  double mean_var = stan::math::mean(chain_var);
  double var_plus = mean_var*(n_samples-1)/n_samples;
  if (m > 1) var_plus += stan::math::variance(chain_mean);
  vector<double> rho_hat_t;
  double rho_hat = 0;
  for (size_t t = 1; (t < n_samples && rho_hat >= 0); t++) {
    vector<double> acov_t(m);
    for (size_t chain = 0; chain < m; chain++) {
      acov_t[chain] = acov[chain][t];
    }
    rho_hat = 1 - (mean_var - stan::math::mean(acov_t)) / var_plus;
    if (rho_hat >= 0)
      rho_hat_t.push_back(rho_hat);
  }
  
  double ess = m*n_samples;
  if (rho_hat_t.size() > 0) {
    ess /= 1 + 2 * stan::math::sum(rho_hat_t);
  }
  return Rcpp::wrap(ess);
  END_RCPP; 
}



/** 
 * Return the split potential scale reduction (split R hat)
 * for the specified parameter.
 *
 * Current implementation takes the minimum number of samples
 * across chains as the number of samples per chain.
 * 
 * @param[in] n Parameter index
 * 
 * @return split R hat.
 */
SEXP split_potential_scale_reduction(SEXP sim, SEXP n_) { 

  BEGIN_RCPP; 
  rstan::validate_sim(sim); 
  Rcpp::List lst(sim); 
  unsigned int n = Rcpp::as<unsigned int>(n_); 
  // Rcpp::Rcout << "n=" << n << std::endl; 
  unsigned int n_chains(rstan::num_chains(sim)); 
  // Rcpp::Rcout << "n_chains=" << n_chains << std::endl; 

  std::vector<unsigned int> ns_save = 
    Rcpp::as<std::vector<unsigned int> >(lst["n_save"]); 

  std::vector<unsigned int> ns_warmup2 = 
    Rcpp::as<std::vector<unsigned int> >(lst["warmup2"]); 

  std::vector<unsigned int> ns_kept(ns_save); 
  for (size_t i = 0; i < ns_kept.size(); i++) 
    ns_kept[i] -= ns_warmup2[i]; 

  unsigned int n_samples = ns_kept[0]; 
  for (size_t chain = 1; chain < n_chains; chain++) {
    n_samples = std::min(n_samples, ns_kept[chain]); 
  }

  if (n_samples % 2 == 1)
    n_samples--;
  
  std::vector<double> split_chain_mean;
  std::vector<double> split_chain_var;

  for (size_t chain = 0; chain < n_chains; chain++) {
    std::vector<double> samples; // (n_samples);
    rstan::get_kept_samples(sim, chain, n, samples);
    // Rcpp::Rcout << samples[0] << ", " << samples.size() << std::endl; 
    
    std::vector<double> split_chain(n_samples/2);
    split_chain.assign(samples.begin(),
                       samples.begin()+n_samples/2);
    split_chain_mean.push_back(stan::math::mean(split_chain));
    split_chain_var.push_back(stan::math::variance(split_chain));
    
    split_chain.assign(samples.end()-n_samples/2, 
                       samples.end());
    split_chain_mean.push_back(stan::math::mean(split_chain));
    split_chain_var.push_back(stan::math::variance(split_chain));
  }

  double var_between = n_samples/2 * stan::math::variance(split_chain_mean);
  double var_within = stan::math::mean(split_chain_var);
  
  // rewrote [(n-1)*W/n + B/n]/W as (n-1+ B/W)/n
  double srhat = sqrt((var_between/var_within + n_samples/2 -1)/(n_samples/2));
  return Rcpp::wrap(srhat);
  END_RCPP; 
}

/*
 * Obtain a permutation of size n. 
 * see <code>permutation</code> in <code>mcmc::chains.hpp</code>. 
 *
 * @param conf an R named list contains elements: n, chains, seed, chain_id. 
 * 
 * @return A permutation of length 'n' starting from 0.
 */ 
SEXP seq_permutation(SEXP conf) { 
  BEGIN_RCPP; 
  Rcpp::List lst(conf); 
  rstan::perm_args args(lst);
  boost::uintmax_t DISCARD_STRIDE = static_cast<boost::uintmax_t>(1) << 50; 
  int n = args.get_n();
  int cid = args.get_chain_id() + args.get_chains(); 
  typedef boost::random::ecuyer1988 RNG;   
  RNG rng(args.get_seed()); 
  rng.discard(DISCARD_STRIDE * (cid - 1));
  Rcpp::IntegerVector x(n); 
  for (int i = 0; i < n; ++i)
    x[i] = i;
  if (n < 2) return x; 
  for (int i = n; --i != 0; ) {
    boost::random::uniform_int_distribution<int> uid(0, i);
    int j = uid(rng);
    int temp = x[i];
    x[i] = x[j];
    x[j] = temp;
  } 
  return x; 
  END_RCPP;
} 

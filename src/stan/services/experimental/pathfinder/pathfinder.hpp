

/**
 * Run the single-path Pathfinder algorithm for the specified model,
 * using the specified random seed and chain ID, using the specified
 * fixed and random initialization, given the L-BFGS parameters, number
 * of ELBO samples to evaluate, and final sample size to return,
 * writing results to the specified callbacks.
 *
 * From the specified initialization, which may be part or wholly
 * random, Pathfinder runs the limited memory BFGS (L-BFGS) algorithm
 * until the objective (log density) has converged within a specified
 * relative tolerance, or the maximum number of iterations is hit.
 * Then for each point on the optimization trajectory, a multivariate
 * normal with covariance given approximately by L-BFGS, the evidence
 * lower bound (ELBO) is evaluated using a specified number of Monte
 * Carlo draws.  Finally, draws are taken from the normal
 * approxiamtion with highest ELBO and transformed back to the
 * constrained scale to be returned as a sample.
 *
 * Algorithm description and evaluation:
 * Lu Zhang, Bob Carpenter, Andrew Gelman, Aki Vehtari (2022)
 * Pathfinder: parallel quasi-Newton variational inference.  arXiv.
 * https://arxiv.org/pdf/2108.03782.pdf
 * 
 * @tparam M type of model class
 * @param model Stan model
 * @param random_seed seed for PRNG
 * @param chain_id chain identifier for multiple chains
 * @param init fixed initial values
 * @param init_radius uniform random initialization bounds 
 * @param lbfgs_max_history max history size, determining rank of
 * covariance estimate
 * @param lbfgs_max_iterations maximum iterations for L-BFGS
 * @param lbfgs_rel_tolerarnce relative tolerance on L-BFGS
 * convergence of the log density
 * @param elbo_samples number of Monte Carlo samples used to estimate
 * ELBO 
 * @param sample_size number of approximate draws to return
 * @param interrupt callback for interrupting process
 * @param logger callback for writing log messages to console
 * @param init_writer callback for writing initialization
 * @param parameter_writer callback for writing sample draws
 * @param diagnostic_writer callback for writing diagnostics
 */
template <typename M>
int pathfinder(Model& model,
	       unsigned int random_seed,
	       unsigned int chain_id,
	       const stan::io::var_context& init,         // pi_0
	       double init_radius,                        // pi_0
	       double lbfgs_max_history,                  // J
	       unsigned int lbfgs_max_iterations,         // L
	       double lbfgs_rel_tolerance,                // tau_rel
	       unsigned int elbo_samples,                 // K
	       int sample_size,                           // M

	       callbacks::interrupt& interrupt,
	       callbacks::logger& logger,
	       callbacks::writer& init_writer,
	       callbacks::writer& parameter_writer,
	       callbacks::writer& diagnostic_writer);

/**
 * Run the multi-path Pathfinder algorithm for the specified model,
 * using the specified random seed and chain ID, using the specified
 * fixed and random initialization, given the LBFGS parameters, number
 * of ELBO samples to evaluate, and final sample size to return,
 * writing results to the specified callbacks.
 *
 * Multi-path Pathfinder runs a specified number of single-path
 * Pathfinder instances, collecting a specified number of draws per
 * run, then importance resamples a final sample of a specified size.
 * Importance resampling is done with Pareto smoothing to reduce
 * variance. 
 * 
 * Algorithm description and evaluation:
 * Lu Zhang, Bob Carpenter, Andrew Gelman, Aki Vehtari (2022)
 * Pathfinder: parallel quasi-Newton variational inference.  arXiv.
 * https://arxiv.org/pdf/2108.03782.pdf
 * 
 * @tparam M type of model class
 * @param model Stan model
 * @param random_seed seed for PRNG
 * @param chain_id chain identifier for multiple chains
 * @param init fixed initial values
 * @param init_radius uniform random initialization bounds 
 * @param lbfgs_max_history max history size, determining rank of
 * covariance estimate
 * @param lbfgs_max_iterations maximum iterations for L-BFGS
 * @param lbfgs_rel_tolerarnce relative tolerance on L-BFGS
 * convergence of the log density
 * @param elbo_samples number of Monte Carlo samples used to estimate
 * ELBO 
 * @param sample_size number of approximate draws to return
 * @param interrupt callback for interrupting process
 * @param logger callback for writing log messages to console
 * @param init_writer callback for writing initialization
 * @param parameter_writer callback for writing sample draws
 * @param diagnostic_writer callback for writing diagnostics
 */
template <typename M>
int multi_pathfinder(Model& model,                  
		     unsigned int random_seed,
		     unsigned int chain,
		     const stan::io::var_context& init,     // pi_0
		     double init_radius,                    // pi_0
		     double lbfgs_max_history,              // J
		     unsigned int lbfgs_max_iterations,     // L
		     double lbfgs_rel_tolerance,            // tau_rel
		     unsigned int elbo_samples,             // K
		     int single_path_runs,                  // I
		     int single_path_sample_size,           // M
		     int sample_size,                       // R

		     callbacks::interrupt& interrupt,
		     callbacks::logger& logger,
		     callbacks::writer& init_writer,
		     callbacks::writer& parameter_writer,
		     callbacks::writer& diagnostic_writer);


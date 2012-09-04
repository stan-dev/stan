
# test user-provided boost library 
# Notes:
#  If there is a boost library installed at say /usr/include, neither
#  the boost included in rstan or the one specified by eigen_lib 
#  is used. 

library(rstan)
print(rstan_options('eigen_lib'))
stan(model_code = "parameters { real y; } model {y ~ normal(0,1);}", 
     eigen_lib = "/opt/eigen-eigen-43d9075b23ef")

stan_model(model_code = "parameters { real y; } model {y ~ normal(0,1);}", 
           boost_lib = "/opt/boost_1_50_0", 
           eigen_lib = "/opt/eigen-eigen-43d9075b23ef")

print(rstan_options('eigen_lib'))

# exception expected since the following eigen path does not exist
# stan(model_code = "parameters { real y; } model {y ~ normal(0,1);}", 
#      eigen_lib = "/opt/eigen") 

print(rstan_options('eigen_lib'))

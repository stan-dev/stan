
# test user-provided boost library 
# Notes:
#  If there is a boost library installed at say /usr/include, neither
#  the boost included in rstan or the one specified by boost_lib 
#  is used. 

library(rstan)
print(rstan_options('boost_lib'))
stan(model_code = "parameters { real y; } model {y ~ normal(0,1);}", 
     boost_lib = "/opt/boost_1_50_0") 
print(rstan_options('boost_lib'))

# exception expected since the boost path does not exist
stan(model_code = "parameters { real y; } model {y ~ normal(0,1);}", 
     boost_lib = "/opt/boost_1_50_00000") 

print(rstan_options('boost_lib'))

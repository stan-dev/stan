
# test user-provided boost library 
# Notes:
#  If there is a boost library installed at say /usr/include, neither
#  the boost included in rstan or the one specified by boost.lib 
#  is used. 

library(rstan)

print(rstan.options('boost.lib'))
stan(model.code = "parameters { real y; } model {y ~ normal(0,1);}", 
     boost.lib = "/opt/boost_1_50_0") 
print(rstan.options('boost.lib'))

stan(model.code = "parameters { real y; } model {y ~ normal(0,1);}", 
     boost.lib = "/opt/boost_1_50_00000") 

print(rstan.options('boost.lib'))

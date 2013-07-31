Data
====

1. nes.data.R
- N     : number of observations
- vote  : voted for George Bush? 1: Yes, 0: No
- income: income level (five-point scale)

2. wells.data.R
- N      : number of observations
- switc  : household switched to new well? 1: Yes, 0: No
- arsenic: level of arsenic of respondent's well
- dist   : distance (in meters) to closest known safe well
- assoc  : any household members active in community organizations? 1: Yes, 0: No
- educ   : education level of head of household

Models
======

1. One predictor
wells_one_pred_scale.stan: glm(switc ~ dist100, family=binomial(link="logit"))
wells_one_pred.stan      : glm(switc ~ dist, family=binomial(link="logit"))
y_x.stan                 : glm(y ~ x, family=binomial(link="logit"))

2. Multiple predictors with no interaction
wells_edu.stan     : glm(switc ~ dist100 + arsenic + educ4, 
			 family=binomial(link="logit"))
wells_two_pred.stan: glm(switc ~ dist100 + arsenic, family=binomial(link="logit"))

3. Multiple predictors with interaction
wells_all.stan         : glm(switc ~ dist100 + arsenic + educ4 + dist100:arsenic,
			     family=binomial(link="logit"))
wells_interactions.stan: glm(switc ~ dist100 + arsenic + dist1::arsenic, 
			     family=binomial(link="logit"))

4. Centering
wells_community.stan               : glm(switc ~ c_dist100 + c_arsenic 
                                                 + c_dist100:c_arsenic + assoc 
                                                 + educ4, 
					 family=binomial(link="logit"))
wells_interactions_center_educ.stan: glm(switc ~ c_dist100 + c_arsenic + c_educ4 
                                                 + c_dist100:c_arsenic 
                                                 + c_dist100:c_educ4 
                                                 + c_arsenic:c_educ4, 
					 family=binomial(link="logit"))
wells_interaction_center.stan      : glm(switc ~ c_dist100 + c_arsenic 
                                                 + c_dist100:c_arsenic, 
					 family=binomial(link="logit"))
wells_social.stan                  : glm(switc ~ c_dist100 + c_arsenic 
                                                 + c_dist100:c_arsenic + educ4, 
					 family=binomial(link="logit"))

5. Log transformations
wells_log_transform.stan : glm(switc ~ c_dist100 + c_log_arsenic + c_educ4 
                                       + c_dist100:c_log_arsenic + c_dist100:c_educ4 
                                       + c_log_arsenic:c_educ4, 
			       family=binomial(link="logit"))
wells_log_transform2.stan: glm(switc ~ dist100 + log(arsenic) + c_educ4 
                                       + dist100:log(arsenic) + dist100:educ4 
                                       + log(arsenic):educ4, 
			       family=binomial(link="logit"))

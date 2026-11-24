# Adversarial Privacy Attacks

Members: 3334 training records; non-members: 1111 holdout records (real, unseen patients). Membership inference AUC of 0.5 means the synthetic data reveals nothing about who was in the training set. Attribute inference reports the MEMBERSHIP ADVANTAGE -- accuracy on members minus accuracy on non-members; population-level inference (both above baseline, equally) is the intended use of released data, only member-specific advantage is leakage.

| run | MIA AUC (95% CI) | learned MIA AUC (95% CI) | empirical ε̂ lower bound | worst AIA membership advantage | anonymeter |
|---|---|---|---|---|---|
| aim40_eps10_seed0 | 0.5173 (0.4997-0.5373) | 0.5146 (0.496-0.5333) | 0.0254 | 0.028 | SO 0.0, link 0.0 |
| aim40_eps15_seed0 | 0.5166 (0.4963-0.5364) | 0.5133 (0.4925-0.5316) | 0.0 | 0.028 | SO 0.0, link 0.0 |
| aim40_eps1_seed0 | 0.5164 (0.4976-0.5356) | 0.5092 (0.4902-0.5273) | 0.014 | 0.028 | SO 0.0, link 0.0 |
| aim40_eps20_seed0 | 0.5172 (0.4976-0.5353) | 0.5123 (0.4939-0.531) | 0.025 | 0.028 | SO 0.0, link 0.0 |
| aim40_eps8_seed0 | 0.5172 (0.4972-0.5357) | 0.5129 (0.4926-0.5313) | 0.0 | 0.028 | SO 0.0, link 0.0 |
| aim50_eps1_seed0 | 0.5185 (0.499-0.5385) | 0.5134 (0.4927-0.5317) | 0.0248 | 0.028 | SO 0.005, link 0.0 |
| aim50_eps5_seed0 | 0.5183 (0.5-0.5385) | 0.5267 (0.5063-0.5449) | 0.0722 | 0.028 | SO 0.005, link 0.0 |
| ctgan_qt_seed0 | 0.5264 (0.5081-0.5455) | 0.5237 (0.506-0.5438) | 0.064 | 0.0032 | SO 0.0, link 0.0 |
| ctgan_seed0 | 0.5207 (0.5002-0.5379) | 0.5117 (0.4931-0.5299) | 0.0352 | 0.0232 | SO 0.0033, link 0.0 |
| ctgan_seed1 | 0.5194 (0.4992-0.5402) | 0.5118 (0.4905-0.5307) | 0.0235 | 0.0001 | SO 0.0, link 0.0 |
| ctgan_seed2 | 0.519 (0.5002-0.538) | 0.5099 (0.4908-0.5275) | 0.0246 | -0.0001 | SO 0.0017, link 0.0 |
| ddpm_g_seed0 | 0.5233 (0.5056-0.5425) | 0.5265 (0.5069-0.5454) | 0.1055 | 0.028 | SO 0.0, link 0.0 |
| ddpm_seed0 🚨 (worst: nearest-synthetic-distance) | 0.5319 (0.5111-0.5512) | 0.5284 (0.5088-0.5458) | 0.0566 | 0.0002 | SO 0.0, link 0.0 |
| ddpm_seed1 | 0.5236 (0.5039-0.5433) | 0.5211 (0.5014-0.5409) | 0.0493 | 0.0176 | SO 0.0, link 0.0 |
| ddpm_seed2 🚨 (worst: nearest-synthetic-distance) | 0.5427 (0.5253-0.5603) | 0.5417 (0.5231-0.5604) | 0.0868 | 0.0299 | SO 0.0, link 0.0 |
| dpctgan_eps10_seed0 | 0.5148 (0.4947-0.533) | 0.4949 (0.4742-0.5166) | 0.0019 | 0.028 | SO 0.0, link 0.0 |
| dpctgan_eps15_seed0 | 0.5045 (0.485-0.5234) | 0.4989 (0.4798-0.5194) | 0.0 | 0.028 | SO 0.0, link 0.0 |
| dpctgan_eps15_seed1 | 0.5174 (0.4981-0.5354) | 0.5154 (0.496-0.5339) | 0.0161 | 0.028 | SO 0.0, link 0.0 |
| dpctgan_eps15_seed2 | 0.5168 (0.4972-0.5357) | 0.5173 (0.4989-0.5349) | 0.008 | 0.0238 | SO 0.0, link 0.0 |
| dpctgan_eps1_seed0 | 0.5186 (0.4978-0.5366) | 0.5024 (0.483-0.5236) | 0.0044 | -0.0027 | SO 0.0, link 0.0 |
| dpctgan_eps20_seed0 | 0.5038 (0.4863-0.5234) | 0.4724 (0.453-0.491) | 0.0 | 0.0134 | SO 0.0, link 0.0 |
| dpctgan_eps5_seed0 | 0.5116 (0.4923-0.5315) | 0.4997 (0.4804-0.5204) | 0.0 | 0.028 | SO 0.0, link 0.0 |
| dpctgan_eps8_seed0 | 0.5152 (0.4969-0.5346) | 0.5077 (0.4885-0.5274) | 0.0269 | 0.028 | SO 0.0, link 0.0 |
| gaussian_copula_seed0 | 0.5266 (0.5091-0.5458) | 0.5284 (0.5098-0.548) | 0.0901 | 0.0299 | SO 0.0017, link 0.005 |
| gaussian_copula_seed1 | 0.5264 (0.507-0.5458) | 0.523 (0.5038-0.5431) | 0.0515 | 0.013 | SO 0.0, link 0.0 |
| gaussian_copula_seed2 | 0.5236 (0.5039-0.5421) | 0.5202 (0.4995-0.5384) | 0.1643 | 0.0125 | SO 0.005, link 0.0 |
| mst_eps0p5_seed0 | 0.5207 (0.5014-0.5407) | 0.5125 (0.4935-0.532) | 0.0062 | 0.0325 | SO 0.0, link 0.0 |
| mst_eps10_seed0 | 0.5213 (0.502-0.5405) | 0.5162 (0.4963-0.5362) | 0.0306 | 0.0226 | SO 0.0198, link 0.0 |
| mst_eps15_seed0 | 0.5215 (0.5003-0.541) | 0.5181 (0.4984-0.5379) | 0.0064 | 0.0136 | SO 0.0099, link 0.0 |
| mst_eps15_seed1 | 0.5283 (0.5087-0.547) | 0.5249 (0.5041-0.5447) | 0.0572 | 0.0104 | SO 0.0231, link 0.0 |
| mst_eps15_seed2 | 0.5279 (0.5079-0.5479) | 0.5268 (0.5068-0.5459) | 0.1167 | 0.0199 | SO 0.0099, link 0.0 |
| mst_eps1_seed0 | 0.5157 (0.495-0.5361) | 0.5134 (0.495-0.533) | 0.0 | -0.0044 | SO 0.005, link 0.0 |
| mst_eps20_seed0 | 0.5181 (0.4991-0.5381) | 0.5109 (0.4917-0.5296) | 0.0214 | 0.0244 | SO 0.0116, link 0.0 |
| mst_eps5_seed0 | 0.5231 (0.5042-0.5428) | 0.5124 (0.4936-0.5325) | 0.014 | 0.0193 | SO 0.0099, link 0.0 |
| mst_eps8_seed0 | 0.5215 (0.5009-0.5419) | 0.5141 (0.4935-0.533) | 0.0089 | 0.0196 | SO 0.0116, link 0.0 |
| patectgan_eps15_seed0 | 0.5228 (0.5046-0.5404) | 0.5225 (0.5028-0.5425) | 0.0845 | 0.0298 | SO 0.0, link 0.0 |
| patectgan_eps1_seed0 | 0.5074 (0.4908-0.5272) | 0.5148 (0.4952-0.5337) | 0.0 | -0.0019 | SO 0.0, link 0.0 |
| patectgan_eps5_seed0 | 0.5054 (0.4859-0.5233) | 0.5011 (0.4812-0.5211) | 0.0 | 0.0314 | SO 0.0, link 0.0 |
| tvae_cap256_seed0 🚨 (worst: nearest-synthetic-distance) | 0.5377 (0.5183-0.5589) | 0.5369 (0.5176-0.5558) | 0.2562 | 0.0299 | SO 0.0, link 0.0 |
| tvae_ep1000_seed0 🚨 (worst: nearest-synthetic-distance) | 0.5419 (0.5233-0.5629) | 0.5394 (0.5179-0.5574) | 0.1311 | 0.0302 | SO 0.0, link 0.0 |
| tvae_ind_seed0 🚨 (worst: nearest-synthetic-distance) | 0.5389 (0.522-0.557) | 0.5364 (0.5185-0.5574) | 0.1405 | 0.0328 | SO 0.0099, link 0.0 |
| tvae_qt_seed0 🚨 (worst: nearest-synthetic-distance) | 0.5315 (0.5102-0.5524) | 0.5225 (0.5014-0.5429) | 0.0614 | 0.0199 | SO 0.0, link 0.0 |
| tvae_qt_seed1 🚨 (worst: nearest-synthetic-distance) | 0.5313 (0.5114-0.5505) | 0.5286 (0.5078-0.5503) | 0.1078 | 0.0049 | SO 0.0, link 0.0 |
| tvae_qt_seed2 🚨 (worst: nearest-synthetic-distance) | 0.5352 (0.5143-0.5542) | 0.5331 (0.5108-0.5491) | 0.1077 | -0.0031 | SO 0.0, link 0.0 |
| tvae_seed0 🚨 (worst: nearest-synthetic-distance) | 0.5416 (0.5227-0.5635) | 0.5395 (0.5198-0.5579) | 0.1788 | 0.0335 | SO 0.0, link 0.0 |
| tvae_seed1 🚨 (worst: cv-logreg over (d1, d2, d1/d2)) | 0.5331 (0.513-0.5513) | 0.5334 (0.5138-0.5542) | 0.0771 | 0.0233 | SO 0.0017, link 0.0 |
| tvae_seed2 🚨 (worst: nearest-synthetic-distance) | 0.5392 (0.5203-0.5607) | 0.5373 (0.518-0.5564) | 0.1924 | 0.0286 | SO 0.0, link 0.0 |

## Who is at risk: membership inference by patient atypicality

WITHIN-STRATUM AUC: members and non-members both get the same atypicality score (distance to their 5th-nearest member, same reference set and encoder), non-members are binned by the member quartile cut points, and each stratum's AUC compares members-in-Qi against non-members-in-Qi only. 0.5 = no leakage on that stratum; elevated values indicate SELECTIVE leakage on that stratum (e.g. a model that memorizes its unusual patients shows it in Q4). Strata with fewer than 30 non-members are skipped ('-'). Cell format: AUC (n members / n non-members).

| run | Q1 typical | Q2 | Q3 | Q4 atypical |
|---|---|---|---|---|
| aim40_eps10_seed0 | 0.505 (834/259) | 0.5317 (833/265) | 0.4749 (834/277) | 0.5111 (833/310) |
| aim40_eps15_seed0 | 0.5033 (834/259) | 0.5324 (833/265) | 0.4727 (834/277) | 0.5107 (833/310) |
| aim40_eps1_seed0 | 0.5016 (834/259) | 0.5332 (833/265) | 0.4739 (834/277) | 0.5089 (833/310) |
| aim40_eps20_seed0 | 0.5108 (834/259) | 0.5326 (833/265) | 0.4731 (834/277) | 0.5098 (833/310) |
| aim40_eps8_seed0 | 0.5049 (834/259) | 0.5353 (833/265) | 0.4717 (834/277) | 0.511 (833/310) |
| aim50_eps1_seed0 | 0.5161 (834/259) | 0.5297 (833/265) | 0.4742 (834/277) | 0.515 (833/310) |
| aim50_eps5_seed0 | 0.5174 (834/259) | 0.5296 (833/265) | 0.4754 (834/277) | 0.5124 (833/310) |
| ctgan_qt_seed0 | 0.5135 (834/259) | 0.5323 (833/265) | 0.4842 (834/277) | 0.5474 (833/310) |
| ctgan_seed0 | 0.5313 (834/259) | 0.52 (833/265) | 0.471 (834/277) | 0.5205 (833/310) |
| ctgan_seed1 | 0.5065 (834/259) | 0.5196 (833/265) | 0.4723 (834/277) | 0.5106 (833/310) |
| ctgan_seed2 | 0.5325 (834/259) | 0.501 (833/265) | 0.4972 (834/277) | 0.5084 (833/310) |
| ddpm_g_seed0 | 0.5272 (834/259) | 0.5322 (833/265) | 0.4793 (834/277) | 0.5153 (833/310) |
| ddpm_seed0 | 0.5358 (834/259) | 0.5262 (833/265) | 0.511 (834/277) | 0.5407 (833/310) |
| ddpm_seed1 | 0.5137 (834/259) | 0.5124 (833/265) | 0.5119 (834/277) | 0.5236 (833/310) |
| ddpm_seed2 | 0.5295 (834/259) | 0.5566 (833/265) | 0.5627 (834/277) | 0.524 (833/310) |
| dpctgan_eps10_seed0 | 0.5452 (834/259) | 0.5051 (833/265) | 0.4942 (834/277) | 0.4888 (833/310) |
| dpctgan_eps15_seed0 | 0.4828 (834/259) | 0.5049 (833/265) | 0.4701 (834/277) | 0.4986 (833/310) |
| dpctgan_eps15_seed1 | 0.5066 (834/259) | 0.5067 (833/265) | 0.4857 (834/277) | 0.5156 (833/310) |
| dpctgan_eps15_seed2 | 0.4969 (834/259) | 0.5291 (833/265) | 0.4913 (834/277) | 0.5075 (833/310) |
| dpctgan_eps1_seed0 | 0.5061 (834/259) | 0.5264 (833/265) | 0.5033 (834/277) | 0.5067 (833/310) |
| dpctgan_eps20_seed0 | 0.5151 (834/259) | 0.4897 (833/265) | 0.4742 (834/277) | 0.515 (833/310) |
| dpctgan_eps5_seed0 | 0.522 (834/259) | 0.5111 (833/265) | 0.4714 (834/277) | 0.5012 (833/310) |
| dpctgan_eps8_seed0 | 0.526 (834/259) | 0.548 (833/265) | 0.4435 (834/277) | 0.5039 (833/310) |
| gaussian_copula_seed0 | 0.5519 (834/259) | 0.5247 (833/265) | 0.4799 (834/277) | 0.5382 (833/310) |
| gaussian_copula_seed1 | 0.5331 (834/259) | 0.5161 (833/265) | 0.5 (834/277) | 0.5356 (833/310) |
| gaussian_copula_seed2 | 0.5284 (834/259) | 0.5386 (833/265) | 0.4829 (834/277) | 0.5179 (833/310) |
| mst_eps0p5_seed0 | 0.5299 (834/259) | 0.5203 (833/265) | 0.4809 (834/277) | 0.5118 (833/310) |
| mst_eps10_seed0 | 0.5246 (834/259) | 0.5138 (833/265) | 0.4862 (834/277) | 0.5203 (833/310) |
| mst_eps15_seed0 | 0.5271 (834/259) | 0.5079 (833/265) | 0.4775 (834/277) | 0.5376 (833/310) |
| mst_eps15_seed1 | 0.5248 (834/259) | 0.5242 (833/265) | 0.4995 (834/277) | 0.5343 (833/310) |
| mst_eps15_seed2 | 0.5323 (834/259) | 0.5086 (833/265) | 0.4777 (834/277) | 0.5539 (833/310) |
| mst_eps1_seed0 | 0.5294 (834/259) | 0.5155 (833/265) | 0.4662 (834/277) | 0.5083 (833/310) |
| mst_eps20_seed0 | 0.5257 (834/259) | 0.5083 (833/265) | 0.4684 (834/277) | 0.521 (833/310) |
| mst_eps5_seed0 | 0.5188 (834/259) | 0.5282 (833/265) | 0.4803 (834/277) | 0.5245 (833/310) |
| mst_eps8_seed0 | 0.524 (834/259) | 0.5282 (833/265) | 0.4737 (834/277) | 0.5199 (833/310) |
| patectgan_eps15_seed0 | 0.5405 (834/259) | 0.5326 (833/265) | 0.4996 (834/277) | 0.508 (833/310) |
| patectgan_eps1_seed0 | 0.4917 (834/259) | 0.4983 (833/265) | 0.4711 (834/277) | 0.5202 (833/310) |
| patectgan_eps5_seed0 | 0.5046 (834/259) | 0.4998 (833/265) | 0.4958 (834/277) | 0.5057 (833/310) |
| tvae_cap256_seed0 | 0.5384 (834/259) | 0.5284 (833/265) | 0.5221 (834/277) | 0.5582 (833/310) |
| tvae_ep1000_seed0 | 0.5473 (834/259) | 0.5536 (833/265) | 0.5195 (834/277) | 0.5632 (833/310) |
| tvae_ind_seed0 | 0.538 (834/259) | 0.5153 (833/265) | 0.5529 (834/277) | 0.5453 (833/310) |
| tvae_qt_seed0 | 0.5186 (834/259) | 0.5221 (833/265) | 0.5285 (834/277) | 0.5258 (833/310) |
| tvae_qt_seed1 | 0.5215 (834/259) | 0.5178 (833/265) | 0.5248 (834/277) | 0.5392 (833/310) |
| tvae_qt_seed2 | 0.5241 (834/259) | 0.5461 (833/265) | 0.5207 (834/277) | 0.5428 (833/310) |
| tvae_seed0 | 0.5319 (834/259) | 0.5416 (833/265) | 0.5414 (834/277) | 0.5546 (833/310) |
| tvae_seed1 | 0.5195 (834/259) | 0.5228 (833/265) | 0.5401 (834/277) | 0.5257 (833/310) |
| tvae_seed2 | 0.5324 (834/259) | 0.5298 (833/265) | 0.5364 (834/277) | 0.5569 (833/310) |

## Attribute inference detail

| run | sensitive attribute | baseline | member acc | non-member acc | advantage |
|---|---|---|---|---|---|
| aim40_eps10_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.6653 | 0.6373 | +0.0280 |
| aim40_eps10_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.249 | 0.2502 | -0.0013 |
| aim40_eps10_seed0 | nyha_nyha_pET | 0.4032 | 0.3179 | 0.3132 | +0.0047 |
| aim40_eps15_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.6653 | 0.6373 | +0.0280 |
| aim40_eps15_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.2546 | 0.2637 | -0.0091 |
| aim40_eps15_seed0 | nyha_nyha_pET | 0.4032 | 0.3362 | 0.3438 | -0.0076 |
| aim40_eps1_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.6653 | 0.6373 | +0.0280 |
| aim40_eps1_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.2349 | 0.2214 | +0.0134 |
| aim40_eps1_seed0 | nyha_nyha_pET | 0.4032 | 0.314 | 0.2916 | +0.0224 |
| aim40_eps20_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.6653 | 0.6373 | +0.0280 |
| aim40_eps20_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.2328 | 0.2475 | -0.0148 |
| aim40_eps20_seed0 | nyha_nyha_pET | 0.4032 | 0.3218 | 0.3141 | +0.0077 |
| aim40_eps8_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.6653 | 0.6373 | +0.0280 |
| aim40_eps8_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.2579 | 0.2421 | +0.0158 |
| aim40_eps8_seed0 | nyha_nyha_pET | 0.4032 | 0.3044 | 0.3204 | -0.0160 |
| aim50_eps1_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.6653 | 0.6373 | +0.0280 |
| aim50_eps1_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.2528 | 0.2556 | -0.0028 |
| aim50_eps1_seed0 | nyha_nyha_pET | 0.4032 | 0.3137 | 0.3285 | -0.0148 |
| aim50_eps5_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.6653 | 0.6373 | +0.0280 |
| aim50_eps5_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.2654 | 0.2493 | +0.0161 |
| aim50_eps5_seed0 | nyha_nyha_pET | 0.4032 | 0.314 | 0.3168 | -0.0028 |
| ctgan_qt_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.569 | 0.5671 | +0.0019 |
| ctgan_qt_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.2388 | 0.243 | -0.0043 |
| ctgan_qt_seed0 | nyha_nyha_pET | 0.4032 | 0.365 | 0.3618 | +0.0032 |
| ctgan_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.5714 | 0.5482 | +0.0232 |
| ctgan_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.2313 | 0.2124 | +0.0188 |
| ctgan_seed0 | nyha_nyha_pET | 0.4032 | 0.341 | 0.3456 | -0.0046 |
| ctgan_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.578 | 0.5779 | +0.0001 |
| ctgan_seed1 | ckd_severity_from_calculated_egfr | 0.288 | 0.2235 | 0.2313 | -0.0079 |
| ctgan_seed1 | nyha_nyha_pET | 0.4032 | 0.3158 | 0.3312 | -0.0154 |
| ctgan_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.4805 | 0.4806 | -0.0001 |
| ctgan_seed2 | ckd_severity_from_calculated_egfr | 0.288 | 0.2382 | 0.2763 | -0.0382 |
| ctgan_seed2 | nyha_nyha_pET | 0.4032 | 0.3269 | 0.3366 | -0.0097 |
| ddpm_g_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.6653 | 0.6373 | +0.0280 |
| ddpm_g_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.2304 | 0.2124 | +0.0179 |
| ddpm_g_seed0 | nyha_nyha_pET | 0.4032 | 0.156 | 0.171 | -0.0150 |
| ddpm_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.4763 | 0.4968 | -0.0205 |
| ddpm_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.2283 | 0.2511 | -0.0229 |
| ddpm_seed0 | nyha_nyha_pET | 0.4032 | 0.1677 | 0.1674 | +0.0002 |
| ddpm_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.5216 | 0.5095 | +0.0121 |
| ddpm_seed1 | ckd_severity_from_calculated_egfr | 0.288 | 0.2597 | 0.2421 | +0.0176 |
| ddpm_seed1 | nyha_nyha_pET | 0.4032 | 0.1704 | 0.1719 | -0.0016 |
| ddpm_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.4931 | 0.5032 | -0.0100 |
| ddpm_seed2 | ckd_severity_from_calculated_egfr | 0.288 | 0.263 | 0.2331 | +0.0299 |
| ddpm_seed2 | nyha_nyha_pET | 0.4032 | 0.1638 | 0.1683 | -0.0045 |
| dpctgan_eps10_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.6653 | 0.6373 | +0.0280 |
| dpctgan_eps10_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.1614 | 0.1737 | -0.0123 |
| dpctgan_eps10_seed0 | nyha_nyha_pET | 0.4032 | 0.1494 | 0.1449 | +0.0045 |
| dpctgan_eps15_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.6653 | 0.6373 | +0.0280 |
| dpctgan_eps15_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.201 | 0.2151 | -0.0142 |
| dpctgan_eps15_seed0 | nyha_nyha_pET | 0.4032 | 0.0 | 0.0 | +0.0000 |
| dpctgan_eps15_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.6653 | 0.6373 | +0.0280 |
| dpctgan_eps15_seed1 | ckd_severity_from_calculated_egfr | 0.288 | 0.1614 | 0.1737 | -0.0123 |
| dpctgan_eps15_seed1 | nyha_nyha_pET | 0.4032 | 0.0 | 0.0 | +0.0000 |
| dpctgan_eps15_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.6467 | 0.6229 | +0.0238 |
| dpctgan_eps15_seed2 | ckd_severity_from_calculated_egfr | 0.288 | 0.2738 | 0.2502 | +0.0236 |
| dpctgan_eps15_seed2 | nyha_nyha_pET | 0.4032 | 0.3341 | 0.3357 | -0.0016 |
| dpctgan_eps1_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.3206 | 0.3492 | -0.0286 |
| dpctgan_eps1_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.174 | 0.189 | -0.0151 |
| dpctgan_eps1_seed0 | nyha_nyha_pET | 0.4032 | 0.108 | 0.1107 | -0.0027 |
| dpctgan_eps20_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.3206 | 0.3492 | -0.0286 |
| dpctgan_eps20_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.3014 | 0.288 | +0.0134 |
| dpctgan_eps20_seed0 | nyha_nyha_pET | 0.4032 | 0.108 | 0.1107 | -0.0027 |
| dpctgan_eps5_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.6653 | 0.6373 | +0.0280 |
| dpctgan_eps5_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.3014 | 0.288 | +0.0134 |
| dpctgan_eps5_seed0 | nyha_nyha_pET | 0.4032 | 0.0 | 0.0 | +0.0000 |
| dpctgan_eps8_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.6653 | 0.6373 | +0.0280 |
| dpctgan_eps8_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.2738 | 0.2502 | +0.0236 |
| dpctgan_eps8_seed0 | nyha_nyha_pET | 0.4032 | 0.1098 | 0.1107 | -0.0009 |
| gaussian_copula_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.5552 | 0.5554 | -0.0002 |
| gaussian_copula_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.2531 | 0.2232 | +0.0299 |
| gaussian_copula_seed0 | nyha_nyha_pET | 0.4032 | 0.3167 | 0.3312 | -0.0145 |
| gaussian_copula_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.554 | 0.541 | +0.0130 |
| gaussian_copula_seed1 | ckd_severity_from_calculated_egfr | 0.288 | 0.2636 | 0.2583 | +0.0053 |
| gaussian_copula_seed1 | nyha_nyha_pET | 0.4032 | 0.3257 | 0.3294 | -0.0037 |
| gaussian_copula_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.5522 | 0.5491 | +0.0031 |
| gaussian_copula_seed2 | ckd_severity_from_calculated_egfr | 0.288 | 0.2582 | 0.2457 | +0.0125 |
| gaussian_copula_seed2 | nyha_nyha_pET | 0.4032 | 0.3257 | 0.3168 | +0.0089 |
| mst_eps0p5_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.6311 | 0.5986 | +0.0325 |
| mst_eps0p5_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.1509 | 0.1845 | -0.0336 |
| mst_eps0p5_seed0 | nyha_nyha_pET | 0.4032 | 0.3509 | 0.333 | +0.0179 |
| mst_eps10_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.6041 | 0.5815 | +0.0226 |
| mst_eps10_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.2525 | 0.2493 | +0.0032 |
| mst_eps10_seed0 | nyha_nyha_pET | 0.4032 | 0.2259 | 0.234 | -0.0082 |
| mst_eps15_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.6167 | 0.6031 | +0.0136 |
| mst_eps15_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.2295 | 0.2367 | -0.0073 |
| mst_eps15_seed0 | nyha_nyha_pET | 0.4032 | 0.3275 | 0.3357 | -0.0082 |
| mst_eps15_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.5573 | 0.5545 | +0.0028 |
| mst_eps15_seed1 | ckd_severity_from_calculated_egfr | 0.288 | 0.2481 | 0.2376 | +0.0104 |
| mst_eps15_seed1 | nyha_nyha_pET | 0.4032 | 0.3068 | 0.3258 | -0.0190 |
| mst_eps15_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.596 | 0.5761 | +0.0199 |
| mst_eps15_seed2 | ckd_severity_from_calculated_egfr | 0.288 | 0.2412 | 0.2448 | -0.0037 |
| mst_eps15_seed2 | nyha_nyha_pET | 0.4032 | 0.2909 | 0.2979 | -0.0070 |
| mst_eps1_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.56 | 0.5644 | -0.0044 |
| mst_eps1_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.2202 | 0.2277 | -0.0076 |
| mst_eps1_seed0 | nyha_nyha_pET | 0.4032 | 0.3056 | 0.3258 | -0.0202 |
| mst_eps20_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.6023 | 0.5779 | +0.0244 |
| mst_eps20_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.2229 | 0.2223 | +0.0005 |
| mst_eps20_seed0 | nyha_nyha_pET | 0.4032 | 0.2603 | 0.2664 | -0.0061 |
| mst_eps5_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.5441 | 0.5248 | +0.0193 |
| mst_eps5_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.2409 | 0.2421 | -0.0013 |
| mst_eps5_seed0 | nyha_nyha_pET | 0.4032 | 0.2912 | 0.3051 | -0.0139 |
| mst_eps8_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.5741 | 0.5545 | +0.0196 |
| mst_eps8_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.2304 | 0.2295 | +0.0008 |
| mst_eps8_seed0 | nyha_nyha_pET | 0.4032 | 0.2609 | 0.261 | -0.0001 |
| patectgan_eps15_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.6635 | 0.6337 | +0.0298 |
| patectgan_eps15_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.2774 | 0.2574 | +0.0200 |
| patectgan_eps15_seed0 | nyha_nyha_pET | 0.4032 | 0.2624 | 0.252 | +0.0104 |
| patectgan_eps1_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.4637 | 0.468 | -0.0043 |
| patectgan_eps1_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.2439 | 0.2628 | -0.0190 |
| patectgan_eps1_seed0 | nyha_nyha_pET | 0.4032 | 0.3527 | 0.3546 | -0.0019 |
| patectgan_eps5_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.6599 | 0.6319 | +0.0280 |
| patectgan_eps5_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.2448 | 0.225 | +0.0197 |
| patectgan_eps5_seed0 | nyha_nyha_pET | 0.4032 | 0.3212 | 0.2898 | +0.0314 |
| tvae_cap256_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.5444 | 0.5248 | +0.0196 |
| tvae_cap256_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.2873 | 0.2574 | +0.0299 |
| tvae_cap256_seed0 | nyha_nyha_pET | 0.4032 | 0.3914 | 0.3879 | +0.0035 |
| tvae_ep1000_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.5441 | 0.5194 | +0.0247 |
| tvae_ep1000_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.2975 | 0.2673 | +0.0302 |
| tvae_ep1000_seed0 | nyha_nyha_pET | 0.4032 | 0.4013 | 0.4032 | -0.0019 |
| tvae_ind_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.5621 | 0.5293 | +0.0328 |
| tvae_ind_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.2876 | 0.279 | +0.0086 |
| tvae_ind_seed0 | nyha_nyha_pET | 0.4032 | 0.3974 | 0.3852 | +0.0122 |
| tvae_qt_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.5339 | 0.514 | +0.0199 |
| tvae_qt_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.2852 | 0.2799 | +0.0053 |
| tvae_qt_seed0 | nyha_nyha_pET | 0.4032 | 0.3995 | 0.4059 | -0.0064 |
| tvae_qt_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.5234 | 0.5185 | +0.0049 |
| tvae_qt_seed1 | ckd_severity_from_calculated_egfr | 0.288 | 0.2744 | 0.2745 | -0.0001 |
| tvae_qt_seed1 | nyha_nyha_pET | 0.4032 | 0.3875 | 0.4149 | -0.0274 |
| tvae_qt_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.5297 | 0.5383 | -0.0086 |
| tvae_qt_seed2 | ckd_severity_from_calculated_egfr | 0.288 | 0.2714 | 0.2745 | -0.0031 |
| tvae_qt_seed2 | nyha_nyha_pET | 0.4032 | 0.3731 | 0.405 | -0.0319 |
| tvae_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.5528 | 0.5266 | +0.0262 |
| tvae_seed0 | ckd_severity_from_calculated_egfr | 0.288 | 0.2855 | 0.252 | +0.0335 |
| tvae_seed0 | nyha_nyha_pET | 0.4032 | 0.3971 | 0.3969 | +0.0002 |
| tvae_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.5582 | 0.5446 | +0.0136 |
| tvae_seed1 | ckd_severity_from_calculated_egfr | 0.288 | 0.2987 | 0.2754 | +0.0233 |
| tvae_seed1 | nyha_nyha_pET | 0.4032 | 0.3935 | 0.4068 | -0.0133 |
| tvae_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.6373 | 0.5516 | 0.523 | +0.0286 |
| tvae_seed2 | ckd_severity_from_calculated_egfr | 0.288 | 0.2783 | 0.2808 | -0.0025 |
| tvae_seed2 | nyha_nyha_pET | 0.4032 | 0.3911 | 0.3843 | +0.0068 |

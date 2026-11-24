# Adversarial Privacy Attacks

Members: 4706 training records; non-members: 1568 holdout records (real, unseen patients). Membership inference AUC of 0.5 means the synthetic data reveals nothing about who was in the training set. Attribute inference reports the MEMBERSHIP ADVANTAGE -- accuracy on members minus accuracy on non-members; population-level inference (both above baseline, equally) is the intended use of released data, only member-specific advantage is leakage.

| run | MIA AUC (95% CI) | learned MIA AUC (95% CI) | empirical ε̂ lower bound | worst AIA membership advantage | anonymeter |
|---|---|---|---|---|---|
| ctgan_qt_seed0 | 0.4975 (0.4826-0.5132) | 0.4861 (0.4695-0.5032) | 0.0 | 0.0096 | SO 0.0, link 0.0 |
| ctgan_seed0 | 0.5076 (0.4909-0.5231) | 0.498 (0.4817-0.5144) | 0.0 | 0.0065 | SO 0.005, link 0.0 |
| ctgan_seed1 | 0.5024 (0.4859-0.5204) | 0.496 (0.4813-0.5139) | 0.0 | 0.0196 | SO 0.0, link 0.0 |
| ctgan_seed2 | 0.5027 (0.4867-0.52) | 0.4975 (0.4825-0.5144) | 0.0 | 0.023 | SO 0.005, link 0.0 |
| ddpm_g_seed0 | 0.5151 (0.4983-0.5332) | 0.5141 (0.4977-0.5281) | 0.0052 | 0.0006 | SO 0.0, link 0.0 |
| ddpm_seed0 | 0.5103 (0.4956-0.5266) | 0.4931 (0.4775-0.5097) | 0.0 | 0.0181 | SO 0.0, link 0.0 |
| ddpm_seed1 | 0.5146 (0.4977-0.531) | 0.509 (0.4913-0.526) | 0.0372 | 0.0006 | SO 0.0, link 0.0 |
| ddpm_seed2 | 0.5076 (0.49-0.5232) | 0.5003 (0.4837-0.5173) | 0.0918 | 0.0079 | SO 0.0033, link 0.0 |
| dpctgan_eps10_seed0 | 0.5064 (0.4909-0.5232) | 0.5081 (0.4929-0.5244) | 0.0 | 0.0006 | SO 0.0, link 0.0 |
| dpctgan_eps15_seed0 | 0.5011 (0.486-0.5172) | 0.5077 (0.4889-0.5255) | 0.0 | 0.0006 | SO 0.0, link 0.0 |
| dpctgan_eps15_seed1 | 0.5059 (0.4896-0.5218) | 0.5149 (0.4974-0.5297) | 0.0837 | 0.0056 | SO 0.0, link 0.0 |
| dpctgan_eps15_seed2 | 0.5074 (0.4914-0.5226) | 0.5025 (0.4872-0.5184) | 0.0 | 0.0075 | SO 0.0, link 0.0 |
| dpctgan_eps1_seed0 | 0.5141 (0.498-0.5301) | 0.5157 (0.5012-0.5319) | 0.0081 | 0.0103 | SO 0.0, link 0.0 |
| dpctgan_eps20_seed0 | 0.504 (0.4883-0.5211) | 0.5026 (0.4862-0.5191) | 0.193 | 0.0006 | SO 0.0, link 0.0 |
| dpctgan_eps5_seed0 | 0.5141 (0.4975-0.5312) | 0.5072 (0.4907-0.5225) | 0.0 | 0.0006 | SO 0.0, link 0.0 |
| dpctgan_eps8_seed0 | 0.5278 (0.5128-0.5462) | 0.5272 (0.5102-0.5416) | 0.0591 | 0.0224 | SO 0.0, link 0.0 |
| gaussian_copula_seed0 | 0.4988 (0.4839-0.516) | 0.5009 (0.4834-0.5187) | 0.0 | 0.0169 | SO 0.0, link 0.0 |
| gaussian_copula_seed1 | 0.4984 (0.4812-0.5148) | 0.4973 (0.4805-0.514) | 0.0 | 0.0169 | SO 0.0099, link 0.0 |
| gaussian_copula_seed2 | 0.4935 (0.4757-0.5093) | 0.4906 (0.4752-0.5079) | 0.0 | 0.0016 | SO 0.0, link 0.0 |
| mst_eps15_seed0 | 0.5115 (0.4948-0.5284) | 0.5016 (0.4867-0.5166) | 0.0 | 0.0152 | SO 0.005, link 0.0 |
| patectgan_eps15_seed0 | 0.5058 (0.4901-0.5209) | 0.5029 (0.4874-0.5188) | 0.0 | 0.0167 | SO 0.0, link 0.0 |
| patectgan_eps1_seed0 | 0.5042 (0.4854-0.5197) | 0.4962 (0.4809-0.5127) | 0.0026 | 0.0069 | SO 0.0, link 0.0 |
| patectgan_eps5_seed0 | 0.5064 (0.4906-0.5244) | 0.498 (0.4792-0.5158) | 0.0 | 0.0006 | SO 0.0, link 0.0 |
| tvae_cap256_seed0 | 0.5244 (0.5068-0.5423) | 0.5244 (0.5072-0.542) | 0.1397 | 0.0073 | SO 0.0, link 0.0 |
| tvae_ep1000_seed0 | 0.5186 (0.5029-0.5342) | 0.5131 (0.4968-0.5297) | 0.0136 | 0.0009 | SO 0.005, link 0.0 |
| tvae_ind_seed0 | 0.5207 (0.5037-0.5372) | 0.5149 (0.4971-0.5319) | 0.0308 | 0.0296 | SO 0.0, link 0.0 |
| tvae_qt_seed0 | 0.5201 (0.5039-0.5358) | 0.5197 (0.5046-0.5385) | 0.0305 | 0.0105 | SO 0.0, link 0.0 |
| tvae_qt_seed1 | 0.5172 (0.5003-0.5348) | 0.5129 (0.4954-0.5281) | 0.0305 | 0.0006 | SO 0.0, link 0.0 |
| tvae_qt_seed2 | 0.5206 (0.5047-0.5355) | 0.5178 (0.5024-0.5335) | 0.032 | 0.0041 | SO 0.005, link 0.0 |
| tvae_seed0 🚨 (worst: cv-logreg over (d1, d2, d1/d2)) | 0.5283 (0.5122-0.5424) | 0.5354 (0.5184-0.5503) | 0.0861 | 0.0171 | SO 0.0, link 0.0 |
| tvae_seed1 | 0.5187 (0.5015-0.5347) | 0.5176 (0.5014-0.5351) | 0.0287 | 0.0096 | SO 0.0, link 0.0 |
| tvae_seed2 | 0.5187 (0.5031-0.535) | 0.5154 (0.5001-0.5324) | 0.2098 | 0.0006 | SO 0.0, link 0.0 |

## Who is at risk: membership inference by patient atypicality

WITHIN-STRATUM AUC: members and non-members both get the same atypicality score (distance to their 5th-nearest member, same reference set and encoder), non-members are binned by the member quartile cut points, and each stratum's AUC compares members-in-Qi against non-members-in-Qi only. 0.5 = no leakage on that stratum; elevated values indicate SELECTIVE leakage on that stratum (e.g. a model that memorizes its unusual patients shows it in Q4). Strata with fewer than 30 non-members are skipped ('-'). Cell format: AUC (n members / n non-members).

| run | Q1 typical | Q2 | Q3 | Q4 atypical |
|---|---|---|---|---|
| ctgan_qt_seed0 | 0.4804 (1177/382) | 0.4862 (1176/367) | 0.4854 (1177/407) | 0.4885 (1176/412) |
| ctgan_seed0 | 0.4926 (1177/382) | 0.5248 (1176/367) | 0.4803 (1177/407) | 0.5067 (1176/412) |
| ctgan_seed1 | 0.4835 (1177/382) | 0.4813 (1176/367) | 0.4989 (1177/407) | 0.5107 (1176/412) |
| ctgan_seed2 | 0.5 (1177/382) | 0.4907 (1176/367) | 0.4959 (1177/407) | 0.4873 (1176/412) |
| ddpm_g_seed0 | 0.5219 (1177/382) | 0.5183 (1176/367) | 0.5093 (1177/407) | 0.4959 (1176/412) |
| ddpm_seed0 | 0.4997 (1177/382) | 0.511 (1176/367) | 0.5102 (1177/407) | 0.4981 (1176/412) |
| ddpm_seed1 | 0.5178 (1177/382) | 0.5027 (1176/367) | 0.5141 (1177/407) | 0.5002 (1176/412) |
| ddpm_seed2 | 0.5142 (1177/382) | 0.5151 (1176/367) | 0.4791 (1177/407) | 0.4984 (1176/412) |
| dpctgan_eps10_seed0 | 0.491 (1177/382) | 0.51 (1176/367) | 0.4964 (1177/407) | 0.5098 (1176/412) |
| dpctgan_eps15_seed0 | 0.4956 (1177/382) | 0.487 (1176/367) | 0.4777 (1177/407) | 0.5158 (1176/412) |
| dpctgan_eps15_seed1 | 0.4877 (1177/382) | 0.4981 (1176/367) | 0.4968 (1177/407) | 0.5196 (1176/412) |
| dpctgan_eps15_seed2 | 0.4604 (1177/382) | 0.5231 (1176/367) | 0.5158 (1177/407) | 0.5074 (1176/412) |
| dpctgan_eps1_seed0 | 0.5303 (1177/382) | 0.5116 (1176/367) | 0.4951 (1177/407) | 0.5091 (1176/412) |
| dpctgan_eps20_seed0 | 0.5001 (1177/382) | 0.4926 (1176/367) | 0.4953 (1177/407) | 0.5147 (1176/412) |
| dpctgan_eps5_seed0 | 0.4799 (1177/382) | 0.5204 (1176/367) | 0.5115 (1177/407) | 0.5296 (1176/412) |
| dpctgan_eps8_seed0 | 0.5317 (1177/382) | 0.5369 (1176/367) | 0.5353 (1177/407) | 0.4988 (1176/412) |
| gaussian_copula_seed0 | 0.4721 (1177/382) | 0.4836 (1176/367) | 0.5092 (1177/407) | 0.4855 (1176/412) |
| gaussian_copula_seed1 | 0.4783 (1177/382) | 0.4701 (1176/367) | 0.5071 (1177/407) | 0.5017 (1176/412) |
| gaussian_copula_seed2 | 0.4662 (1177/382) | 0.4699 (1176/367) | 0.4898 (1177/407) | 0.5036 (1176/412) |
| mst_eps15_seed0 | 0.4788 (1177/382) | 0.5233 (1176/367) | 0.4999 (1177/407) | 0.524 (1176/412) |
| patectgan_eps15_seed0 | 0.5205 (1177/382) | 0.4873 (1176/367) | 0.489 (1177/407) | 0.5043 (1176/412) |
| patectgan_eps1_seed0 | 0.5509 (1177/382) | 0.465 (1176/367) | 0.4941 (1177/407) | 0.4971 (1176/412) |
| patectgan_eps5_seed0 | 0.5163 (1177/382) | 0.4968 (1176/367) | 0.4861 (1177/407) | 0.4969 (1176/412) |
| tvae_cap256_seed0 | 0.536 (1177/382) | 0.5159 (1176/367) | 0.5363 (1177/407) | 0.5204 (1176/412) |
| tvae_ep1000_seed0 | 0.526 (1177/382) | 0.5074 (1176/367) | 0.5262 (1177/407) | 0.5219 (1176/412) |
| tvae_ind_seed0 | 0.5199 (1177/382) | 0.5271 (1176/367) | 0.5197 (1177/407) | 0.5246 (1176/412) |
| tvae_qt_seed0 | 0.5401 (1177/382) | 0.5071 (1176/367) | 0.5236 (1177/407) | 0.521 (1176/412) |
| tvae_qt_seed1 | 0.5257 (1177/382) | 0.5242 (1176/367) | 0.5075 (1177/407) | 0.5137 (1176/412) |
| tvae_qt_seed2 | 0.5188 (1177/382) | 0.5378 (1176/367) | 0.5257 (1177/407) | 0.4966 (1176/412) |
| tvae_seed0 | 0.5215 (1177/382) | 0.5165 (1176/367) | 0.569 (1177/407) | 0.5221 (1176/412) |
| tvae_seed1 | 0.5393 (1177/382) | 0.5104 (1176/367) | 0.5309 (1177/407) | 0.4973 (1176/412) |
| tvae_seed2 | 0.5513 (1177/382) | 0.4924 (1176/367) | 0.5249 (1177/407) | 0.5164 (1176/412) |

## Attribute inference detail

| run | sensitive attribute | baseline | member acc | non-member acc | advantage |
|---|---|---|---|---|---|
| ctgan_qt_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 1.0 | 0.9994 | +0.0006 |
| ctgan_qt_seed0 | ckd_severity_from_calculated_egfr | 0.3769 | 0.3183 | 0.3087 | +0.0096 |
| ctgan_qt_seed0 | nyha_nyha_pET | 0.3782 | 0.3657 | 0.3661 | -0.0004 |
| ctgan_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 1.0 | 0.9994 | +0.0006 |
| ctgan_seed0 | ckd_severity_from_calculated_egfr | 0.3769 | 0.2947 | 0.2883 | +0.0065 |
| ctgan_seed0 | nyha_nyha_pET | 0.3782 | 0.3668 | 0.3622 | +0.0045 |
| ctgan_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 1.0 | 0.9994 | +0.0006 |
| ctgan_seed1 | ckd_severity_from_calculated_egfr | 0.3769 | 0.3279 | 0.3335 | -0.0057 |
| ctgan_seed1 | nyha_nyha_pET | 0.3782 | 0.3372 | 0.3176 | +0.0196 |
| ctgan_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 1.0 | 0.9994 | +0.0006 |
| ctgan_seed2 | ckd_severity_from_calculated_egfr | 0.3769 | 0.3209 | 0.3093 | +0.0116 |
| ctgan_seed2 | nyha_nyha_pET | 0.3782 | 0.3279 | 0.3048 | +0.0230 |
| ddpm_g_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 1.0 | 0.9994 | +0.0006 |
| ddpm_g_seed0 | ckd_severity_from_calculated_egfr | 0.3769 | 0.3944 | 0.4018 | -0.0074 |
| ddpm_g_seed0 | nyha_nyha_pET | 0.3782 | 0.327 | 0.354 | -0.0269 |
| ddpm_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 1.0 | 0.9994 | +0.0006 |
| ddpm_seed0 | ckd_severity_from_calculated_egfr | 0.3769 | 0.3855 | 0.3673 | +0.0181 |
| ddpm_seed0 | nyha_nyha_pET | 0.3782 | 0.3355 | 0.3355 | +0.0001 |
| ddpm_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 1.0 | 0.9994 | +0.0006 |
| ddpm_seed1 | ckd_severity_from_calculated_egfr | 0.3769 | 0.3695 | 0.3731 | -0.0036 |
| ddpm_seed1 | nyha_nyha_pET | 0.3782 | 0.3164 | 0.3316 | -0.0152 |
| ddpm_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 1.0 | 0.9994 | +0.0006 |
| ddpm_seed2 | ckd_severity_from_calculated_egfr | 0.3769 | 0.3812 | 0.3737 | +0.0075 |
| ddpm_seed2 | nyha_nyha_pET | 0.3782 | 0.3338 | 0.3259 | +0.0079 |
| dpctgan_eps10_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 1.0 | 0.9994 | +0.0006 |
| dpctgan_eps10_seed0 | ckd_severity_from_calculated_egfr | 0.3769 | 0.3757 | 0.3769 | -0.0012 |
| dpctgan_eps10_seed0 | nyha_nyha_pET | 0.3782 | 0.3302 | 0.352 | -0.0218 |
| dpctgan_eps15_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 1.0 | 0.9994 | +0.0006 |
| dpctgan_eps15_seed0 | ckd_severity_from_calculated_egfr | 0.3769 | 0.1156 | 0.1193 | -0.0037 |
| dpctgan_eps15_seed0 | nyha_nyha_pET | 0.3782 | 0.3302 | 0.352 | -0.0218 |
| dpctgan_eps15_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 1.0 | 0.9994 | +0.0006 |
| dpctgan_eps15_seed1 | ckd_severity_from_calculated_egfr | 0.3769 | 0.1725 | 0.1811 | -0.0086 |
| dpctgan_eps15_seed1 | nyha_nyha_pET | 0.3782 | 0.3653 | 0.3597 | +0.0056 |
| dpctgan_eps15_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 1.0 | 0.9994 | +0.0006 |
| dpctgan_eps15_seed2 | ckd_severity_from_calculated_egfr | 0.3769 | 0.3621 | 0.3546 | +0.0075 |
| dpctgan_eps15_seed2 | nyha_nyha_pET | 0.3782 | 0.003 | 0.0019 | +0.0011 |
| dpctgan_eps1_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 1.0 | 0.9994 | +0.0006 |
| dpctgan_eps1_seed0 | ckd_severity_from_calculated_egfr | 0.3769 | 0.1685 | 0.1582 | +0.0103 |
| dpctgan_eps1_seed0 | nyha_nyha_pET | 0.3782 | 0.3302 | 0.352 | -0.0218 |
| dpctgan_eps20_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 1.0 | 0.9994 | +0.0006 |
| dpctgan_eps20_seed0 | ckd_severity_from_calculated_egfr | 0.3769 | 0.3729 | 0.3731 | -0.0002 |
| dpctgan_eps20_seed0 | nyha_nyha_pET | 0.3782 | 0.0 | 0.0 | +0.0000 |
| dpctgan_eps5_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 1.0 | 0.9994 | +0.0006 |
| dpctgan_eps5_seed0 | ckd_severity_from_calculated_egfr | 0.3769 | 0.0914 | 0.0976 | -0.0062 |
| dpctgan_eps5_seed0 | nyha_nyha_pET | 0.3782 | 0.3302 | 0.352 | -0.0218 |
| dpctgan_eps8_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 1.0 | 0.9994 | +0.0006 |
| dpctgan_eps8_seed0 | ckd_severity_from_calculated_egfr | 0.3769 | 0.3504 | 0.3431 | +0.0073 |
| dpctgan_eps8_seed0 | nyha_nyha_pET | 0.3782 | 0.3961 | 0.3737 | +0.0224 |
| gaussian_copula_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 1.0 | 0.9994 | +0.0006 |
| gaussian_copula_seed0 | ckd_severity_from_calculated_egfr | 0.3769 | 0.293 | 0.2761 | +0.0169 |
| gaussian_copula_seed0 | nyha_nyha_pET | 0.3782 | 0.3272 | 0.3501 | -0.0229 |
| gaussian_copula_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 1.0 | 0.9994 | +0.0006 |
| gaussian_copula_seed1 | ckd_severity_from_calculated_egfr | 0.3769 | 0.2992 | 0.2959 | +0.0033 |
| gaussian_copula_seed1 | nyha_nyha_pET | 0.3782 | 0.3485 | 0.3316 | +0.0169 |
| gaussian_copula_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 1.0 | 0.9994 | +0.0006 |
| gaussian_copula_seed2 | ckd_severity_from_calculated_egfr | 0.3769 | 0.3141 | 0.3278 | -0.0137 |
| gaussian_copula_seed2 | nyha_nyha_pET | 0.3782 | 0.3466 | 0.345 | +0.0016 |
| mst_eps15_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 1.0 | 0.9994 | +0.0006 |
| mst_eps15_seed0 | ckd_severity_from_calculated_egfr | 0.3769 | 0.2862 | 0.271 | +0.0152 |
| mst_eps15_seed0 | nyha_nyha_pET | 0.3782 | 0.3525 | 0.3508 | +0.0018 |
| patectgan_eps15_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 1.0 | 0.9994 | +0.0006 |
| patectgan_eps15_seed0 | ckd_severity_from_calculated_egfr | 0.3769 | 0.3755 | 0.3769 | -0.0014 |
| patectgan_eps15_seed0 | nyha_nyha_pET | 0.3782 | 0.3298 | 0.3131 | +0.0167 |
| patectgan_eps1_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 1.0 | 0.9994 | +0.0006 |
| patectgan_eps1_seed0 | ckd_severity_from_calculated_egfr | 0.3769 | 0.2856 | 0.2787 | +0.0069 |
| patectgan_eps1_seed0 | nyha_nyha_pET | 0.3782 | 0.35 | 0.3565 | -0.0065 |
| patectgan_eps5_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 1.0 | 0.9994 | +0.0006 |
| patectgan_eps5_seed0 | ckd_severity_from_calculated_egfr | 0.3769 | 0.3744 | 0.3763 | -0.0019 |
| patectgan_eps5_seed0 | nyha_nyha_pET | 0.3782 | 0.3068 | 0.3093 | -0.0025 |
| tvae_cap256_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 1.0 | 0.9994 | +0.0006 |
| tvae_cap256_seed0 | ckd_severity_from_calculated_egfr | 0.3769 | 0.3345 | 0.3272 | +0.0073 |
| tvae_cap256_seed0 | nyha_nyha_pET | 0.3782 | 0.3428 | 0.3367 | +0.0060 |
| tvae_ep1000_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 1.0 | 0.9994 | +0.0006 |
| tvae_ep1000_seed0 | ckd_severity_from_calculated_egfr | 0.3769 | 0.3753 | 0.3744 | +0.0009 |
| tvae_ep1000_seed0 | nyha_nyha_pET | 0.3782 | 0.3394 | 0.354 | -0.0146 |
| tvae_ind_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 1.0 | 0.9994 | +0.0006 |
| tvae_ind_seed0 | ckd_severity_from_calculated_egfr | 0.3769 | 0.3821 | 0.3776 | +0.0045 |
| tvae_ind_seed0 | nyha_nyha_pET | 0.3782 | 0.3708 | 0.3412 | +0.0296 |
| tvae_qt_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 1.0 | 0.9994 | +0.0006 |
| tvae_qt_seed0 | ckd_severity_from_calculated_egfr | 0.3769 | 0.3674 | 0.3629 | +0.0045 |
| tvae_qt_seed0 | nyha_nyha_pET | 0.3782 | 0.351 | 0.3406 | +0.0105 |
| tvae_qt_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 1.0 | 0.9994 | +0.0006 |
| tvae_qt_seed1 | ckd_severity_from_calculated_egfr | 0.3769 | 0.344 | 0.3533 | -0.0093 |
| tvae_qt_seed1 | nyha_nyha_pET | 0.3782 | 0.3468 | 0.3565 | -0.0097 |
| tvae_qt_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 1.0 | 0.9994 | +0.0006 |
| tvae_qt_seed2 | ckd_severity_from_calculated_egfr | 0.3769 | 0.3523 | 0.3482 | +0.0041 |
| tvae_qt_seed2 | nyha_nyha_pET | 0.3782 | 0.3536 | 0.375 | -0.0214 |
| tvae_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 1.0 | 0.9994 | +0.0006 |
| tvae_seed0 | ckd_severity_from_calculated_egfr | 0.3769 | 0.3755 | 0.3833 | -0.0078 |
| tvae_seed0 | nyha_nyha_pET | 0.3782 | 0.3564 | 0.3393 | +0.0171 |
| tvae_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 1.0 | 0.9994 | +0.0006 |
| tvae_seed1 | ckd_severity_from_calculated_egfr | 0.3769 | 0.3687 | 0.3591 | +0.0096 |
| tvae_seed1 | nyha_nyha_pET | 0.3782 | 0.3606 | 0.3705 | -0.0099 |
| tvae_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 1.0 | 0.9994 | +0.0006 |
| tvae_seed2 | ckd_severity_from_calculated_egfr | 0.3769 | 0.3432 | 0.352 | -0.0089 |
| tvae_seed2 | nyha_nyha_pET | 0.3782 | 0.3394 | 0.354 | -0.0146 |

# Adversarial Privacy Attacks

Members: 3520 training records; non-members: 1174 holdout records (real, unseen patients). Membership inference AUC of 0.5 means the synthetic data reveals nothing about who was in the training set. Attribute inference reports the MEMBERSHIP ADVANTAGE -- accuracy on members minus accuracy on non-members; population-level inference (both above baseline, equally) is the intended use of released data, only member-specific advantage is leakage.

| run | MIA AUC (95% CI) | learned MIA AUC (95% CI) | empirical ε̂ lower bound | worst AIA membership advantage | anonymeter |
|---|---|---|---|---|---|
| aim40_eps1_seed0 | 0.4984 (0.4793-0.5173) | 0.4722 (0.4518-0.49) | 0.1652 | 0.0037 | SO 0.0, link 0.0 |
| aim50_eps1_seed0 | 0.4988 (0.4796-0.5174) | 0.4773 (0.4593-0.498) | 0.0953 | 0.0234 | SO 0.0099, link 0.0 |
| ctgan_qt_seed0 | 0.5141 (0.4964-0.5316) | 0.5046 (0.4856-0.5228) | 0.0096 | 0.0072 | SO 0.0, link 0.0 |
| ctgan_seed0 | 0.496 (0.4759-0.5159) | 0.4921 (0.4753-0.5097) | 0.0 | 0.0227 | SO 0.0083, link 0.0 |
| ctgan_seed1 | 0.4991 (0.4802-0.5176) | 0.4648 (0.4459-0.4815) | 0.08 | 0.0 | SO 0.0, link 0.0 |
| ctgan_seed2 | 0.518 (0.4993-0.5361) | 0.5148 (0.4951-0.5323) | 0.0335 | 0.0264 | SO 0.0, link 0.0 |
| ddpm_g_seed0 | 0.5098 (0.4906-0.5282) | 0.4873 (0.4683-0.5064) | 0.0638 | 0.0328 | SO 0.0, link 0.0 |
| ddpm_seed0 | 0.5045 (0.4874-0.5235) | 0.4732 (0.4552-0.4912) | 0.0 | 0.0104 | SO 0.0, link 0.0 |
| ddpm_seed1 | 0.515 (0.497-0.5327) | 0.5019 (0.4838-0.5188) | 0.0 | 0.0 | SO 0.0, link 0.0 |
| ddpm_seed2 | 0.506 (0.4869-0.5222) | 0.4724 (0.4549-0.4905) | 0.0 | 0.0066 | SO 0.0, link 0.0 |
| dpctgan_eps10_seed0 | 0.4988 (0.4798-0.5181) | 0.4978 (0.475-0.5161) | 0.0 | 0.006 | SO 0.0, link 0.0 |
| dpctgan_eps15_seed0 | 0.4964 (0.478-0.5138) | 0.4729 (0.454-0.4918) | 0.0 | 0.0037 | SO 0.0, link 0.0 |
| dpctgan_eps15_seed1 | 0.5029 (0.4831-0.5214) | 0.4815 (0.4641-0.5012) | 0.0 | 0.0037 | SO 0.0, link 0.0 |
| dpctgan_eps15_seed2 | 0.4975 (0.4797-0.5151) | 0.4729 (0.4554-0.4921) | 0.0 | 0.0149 | SO 0.0, link 0.0 |
| dpctgan_eps1_seed0 | 0.51 (0.4893-0.5291) | 0.5057 (0.4887-0.5241) | 0.0042 | 0.0067 | SO 0.0, link 0.005 |
| dpctgan_eps20_seed0 | 0.5026 (0.4847-0.5218) | 0.5152 (0.4969-0.5326) | 0.0272 | 0.0057 | SO 0.0, link 0.0 |
| dpctgan_eps5_seed0 | 0.5014 (0.4838-0.5189) | 0.5038 (0.4855-0.5239) | 0.0 | 0.0037 | SO 0.0, link 0.0 |
| dpctgan_eps8_seed0 | 0.4996 (0.4806-0.5188) | 0.4835 (0.4665-0.503) | 0.0 | 0.0 | SO 0.0, link 0.0 |
| gaussian_copula_seed0 | 0.5076 (0.4882-0.5254) | 0.4817 (0.4635-0.503) | 0.0034 | 0.0 | SO 0.0099, link 0.0 |
| gaussian_copula_seed1 | 0.5092 (0.4905-0.5292) | 0.4807 (0.4633-0.5002) | 0.0 | 0.0153 | SO 0.0, link 0.0 |
| gaussian_copula_seed2 | 0.5082 (0.4896-0.5256) | 0.4914 (0.4735-0.5093) | 0.0091 | 0.0333 | SO 0.0, link 0.0 |
| mst_eps0p5_seed0 | 0.5003 (0.4818-0.5186) | 0.4774 (0.4595-0.4968) | 0.0 | 0.0231 | SO 0.005, link 0.0 |
| mst_eps10_seed0 | 0.5063 (0.4866-0.5262) | 0.4825 (0.4638-0.5005) | 0.0 | 0.0237 | SO 0.0149, link 0.0 |
| mst_eps15_seed0 | 0.5025 (0.4838-0.5212) | 0.4969 (0.4793-0.5157) | 0.0 | 0.0076 | SO 0.0198, link 0.0 |
| mst_eps15_seed1 | 0.5006 (0.4824-0.5213) | 0.4744 (0.4555-0.492) | 0.0 | 0.0095 | SO 0.0248, link 0.0 |
| mst_eps15_seed2 | 0.5078 (0.4888-0.5266) | 0.4993 (0.4807-0.5178) | 0.0 | 0.0 | SO 0.0281, link 0.0 |
| mst_eps1_seed0 | 0.5138 (0.4962-0.5318) | 0.4962 (0.4777-0.5153) | 0.0049 | 0.0 | SO 0.0198, link 0.0 |
| mst_eps20_seed0 | 0.5089 (0.4878-0.5262) | 0.5091 (0.4877-0.5283) | 0.0077 | 0.0083 | SO 0.0446, link 0.0 |
| mst_eps5_seed0 | 0.5026 (0.4841-0.5195) | 0.4792 (0.4614-0.4989) | 0.0 | 0.0151 | SO 0.0083, link 0.0 |
| mst_eps8_seed0 | 0.5178 (0.4999-0.5397) | 0.5022 (0.4838-0.5206) | 0.0368 | 0.004 | SO 0.0132, link 0.0 |
| patectgan_eps15_seed0 | 0.5035 (0.4836-0.524) | 0.4653 (0.4456-0.4829) | 0.0 | 0.0081 | SO 0.0, link 0.0 |
| patectgan_eps1_seed0 | 0.5072 (0.4889-0.5253) | 0.4832 (0.4636-0.5029) | 0.0 | 0.0079 | SO 0.0, link 0.0 |
| patectgan_eps5_seed0 | 0.5037 (0.4844-0.5237) | 0.5095 (0.4904-0.5303) | 0.0 | 0.0052 | SO 0.0, link 0.0 |
| tvae_cap256_seed0 | 0.5314 (0.5117-0.5478) | 0.521 (0.502-0.5404) | 0.2386 | 0.0104 | SO 0.0, link 0.0 |
| tvae_ep1000_seed0 | 0.5245 (0.5074-0.5446) | 0.5234 (0.5025-0.5421) | 0.033 | 0.0106 | SO 0.0, link 0.0 |
| tvae_ind_seed0 | 0.5216 (0.5031-0.5411) | 0.5077 (0.4886-0.5277) | 0.0406 | 0.0188 | SO 0.005, link 0.005 |
| tvae_qt_seed0 | 0.5221 (0.5044-0.5388) | 0.5089 (0.4897-0.5272) | 0.0365 | 0.0122 | SO 0.0, link 0.0 |
| tvae_qt_seed1 | 0.5221 (0.5038-0.5434) | 0.5098 (0.4911-0.5264) | 0.0245 | 0.0197 | SO 0.0, link 0.0 |
| tvae_qt_seed2 | 0.5224 (0.5028-0.5376) | 0.5091 (0.49-0.527) | 0.0339 | 0.0131 | SO 0.0, link 0.0 |
| tvae_seed0 | 0.5205 (0.5018-0.5395) | 0.5127 (0.4951-0.5322) | 0.0364 | 0.0362 | SO 0.0, link 0.0 |
| tvae_seed1 | 0.5252 (0.5067-0.5427) | 0.5154 (0.4956-0.5336) | 0.0163 | 0.0208 | SO 0.0033, link 0.0 |
| tvae_seed2 | 0.5157 (0.497-0.5348) | 0.5079 (0.489-0.5261) | 0.0 | 0.0244 | SO 0.0, link 0.0 |

## Who is at risk: membership inference by patient atypicality

Members are split into quartiles of atypicality (distance to their 5th-nearest fellow member); each quartile is attacked against all non-members. A model that leaks selectively on unusual patients shows an elevated Q4 AUC even when the overall AUC is at chance.

| run | Q1 typical | Q2 | Q3 | Q4 atypical |
|---|---|---|---|---|
| aim40_eps1_seed0 | 0.7124 | 0.5366 | 0.4679 | 0.2768 |
| aim50_eps1_seed0 | 0.7189 | 0.5361 | 0.4649 | 0.2752 |
| ctgan_qt_seed0 | 0.809 | 0.615 | 0.4404 | 0.1922 |
| ctgan_seed0 | 0.7322 | 0.6 | 0.4464 | 0.2053 |
| ctgan_seed1 | 0.7849 | 0.6082 | 0.4266 | 0.1768 |
| ctgan_seed2 | 0.8152 | 0.6197 | 0.4386 | 0.1986 |
| ddpm_g_seed0 | 0.7948 | 0.5876 | 0.4166 | 0.2401 |
| ddpm_seed0 | 0.7873 | 0.5765 | 0.4201 | 0.2342 |
| ddpm_seed1 | 0.7074 | 0.5738 | 0.4561 | 0.3226 |
| ddpm_seed2 | 0.7582 | 0.5701 | 0.4242 | 0.2717 |
| dpctgan_eps10_seed0 | 0.7176 | 0.5558 | 0.4515 | 0.2701 |
| dpctgan_eps15_seed0 | 0.7136 | 0.5521 | 0.4674 | 0.2525 |
| dpctgan_eps15_seed1 | 0.7576 | 0.5611 | 0.441 | 0.252 |
| dpctgan_eps15_seed2 | 0.6966 | 0.5544 | 0.48 | 0.2591 |
| dpctgan_eps1_seed0 | 0.7222 | 0.5816 | 0.4616 | 0.2746 |
| dpctgan_eps20_seed0 | 0.6652 | 0.556 | 0.4756 | 0.3137 |
| dpctgan_eps5_seed0 | 0.7599 | 0.5689 | 0.4384 | 0.2385 |
| dpctgan_eps8_seed0 | 0.6999 | 0.5695 | 0.4551 | 0.2737 |
| gaussian_copula_seed0 | 0.8227 | 0.6016 | 0.4223 | 0.1836 |
| gaussian_copula_seed1 | 0.8291 | 0.6001 | 0.4216 | 0.1858 |
| gaussian_copula_seed2 | 0.8217 | 0.6037 | 0.4211 | 0.1864 |
| mst_eps0p5_seed0 | 0.7827 | 0.5824 | 0.4397 | 0.1962 |
| mst_eps10_seed0 | 0.7808 | 0.5726 | 0.4337 | 0.238 |
| mst_eps15_seed0 | 0.7811 | 0.5743 | 0.4316 | 0.223 |
| mst_eps15_seed1 | 0.7765 | 0.5797 | 0.4288 | 0.2176 |
| mst_eps15_seed2 | 0.7891 | 0.5703 | 0.4232 | 0.2486 |
| mst_eps1_seed0 | 0.8144 | 0.5787 | 0.4239 | 0.2382 |
| mst_eps20_seed0 | 0.7856 | 0.5748 | 0.4345 | 0.2407 |
| mst_eps5_seed0 | 0.7715 | 0.5678 | 0.4259 | 0.2453 |
| mst_eps8_seed0 | 0.8034 | 0.5852 | 0.4437 | 0.239 |
| patectgan_eps15_seed0 | 0.7976 | 0.5764 | 0.4255 | 0.2147 |
| patectgan_eps1_seed0 | 0.6935 | 0.5716 | 0.4767 | 0.2873 |
| patectgan_eps5_seed0 | 0.6281 | 0.5501 | 0.474 | 0.3623 |
| tvae_cap256_seed0 | 0.8571 | 0.6328 | 0.4342 | 0.2014 |
| tvae_ep1000_seed0 | 0.8438 | 0.6335 | 0.4345 | 0.1863 |
| tvae_ind_seed0 | 0.8442 | 0.6312 | 0.4261 | 0.1849 |
| tvae_qt_seed0 | 0.85 | 0.6281 | 0.4292 | 0.181 |
| tvae_qt_seed1 | 0.8387 | 0.6302 | 0.4314 | 0.188 |
| tvae_qt_seed2 | 0.8405 | 0.6338 | 0.4272 | 0.1881 |
| tvae_seed0 | 0.8396 | 0.6286 | 0.4277 | 0.1863 |
| tvae_seed1 | 0.8476 | 0.6328 | 0.4277 | 0.1927 |
| tvae_seed2 | 0.8374 | 0.6155 | 0.4213 | 0.1886 |

## Attribute inference detail

| run | sensitive attribute | baseline | member acc | non-member acc | advantage |
|---|---|---|---|---|---|
| aim40_eps1_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 1.0 | 0.5719 | 0.5681 | +0.0037 |
| aim40_eps1_seed0 | ckd_severity_from_calculated_egfr | 0.2852 | 0.1736 | 0.1831 | -0.0096 |
| aim40_eps1_seed0 | nyha_nyha_pET | 0.7511 | 0.0 | 0.0 | +0.0000 |
| aim50_eps1_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 1.0 | 0.5719 | 0.5681 | +0.0037 |
| aim50_eps1_seed0 | ckd_severity_from_calculated_egfr | 0.2858 | 0.2176 | 0.1942 | +0.0234 |
| aim50_eps1_seed0 | nyha_nyha_pET | 0.7531 | 0.0 | 0.0 | +0.0000 |
| ctgan_qt_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5528 | 0.4679 | 0.4847 | -0.0168 |
| ctgan_qt_seed0 | ckd_severity_from_calculated_egfr | 0.3219 | 0.2142 | 0.207 | +0.0072 |
| ctgan_qt_seed0 | nyha_nyha_pET | 0.8909 | 0.0 | 0.0 | +0.0000 |
| ctgan_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6179 | 0.4452 | 0.4225 | +0.0227 |
| ctgan_seed0 | ckd_severity_from_calculated_egfr | 0.3199 | 0.1957 | 0.1857 | +0.0100 |
| ctgan_seed0 | nyha_nyha_pET | 0.8446 | 0.0 | 0.0 | +0.0000 |
| ctgan_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.6395 | 0.5131 | 0.5187 | -0.0057 |
| ctgan_seed1 | ckd_severity_from_calculated_egfr | 0.3989 | 0.2091 | 0.2206 | -0.0115 |
| ctgan_seed1 | nyha_nyha_pET | 0.7145 | 0.0 | 0.0 | +0.0000 |
| ctgan_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.5014 | 0.4966 | 0.4702 | +0.0264 |
| ctgan_seed2 | ckd_severity_from_calculated_egfr | 0.2994 | 0.1855 | 0.1789 | +0.0066 |
| ctgan_seed2 | nyha_nyha_pET | 0.871 | 0.0 | 0.0 | +0.0000 |
| ddpm_g_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.4259 | 0.3841 | 0.3756 | +0.0085 |
| ddpm_g_seed0 | ckd_severity_from_calculated_egfr | 0.2676 | 0.1895 | 0.1567 | +0.0328 |
| ddpm_g_seed0 | nyha_nyha_pET | 0.5281 | 0.0 | 0.0 | +0.0000 |
| ddpm_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.3639 | 0.3034 | 0.293 | +0.0104 |
| ddpm_seed0 | ckd_severity_from_calculated_egfr | 0.2625 | 0.1963 | 0.1925 | +0.0038 |
| ddpm_seed0 | nyha_nyha_pET | 0.527 | 0.0 | 0.0 | +0.0000 |
| ddpm_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.4088 | 0.3929 | 0.4003 | -0.0074 |
| ddpm_seed1 | ckd_severity_from_calculated_egfr | 0.2557 | 0.1756 | 0.1772 | -0.0016 |
| ddpm_seed1 | nyha_nyha_pET | 0.5878 | 0.0 | 0.0 | +0.0000 |
| ddpm_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.4321 | 0.3585 | 0.3739 | -0.0154 |
| ddpm_seed2 | ckd_severity_from_calculated_egfr | 0.3588 | 0.1855 | 0.1789 | +0.0066 |
| ddpm_seed2 | nyha_nyha_pET | 0.5662 | 0.0 | 0.0 | +0.0000 |
| dpctgan_eps10_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9639 | 0.5724 | 0.5664 | +0.0060 |
| dpctgan_eps10_seed0 | ckd_severity_from_calculated_egfr | 0.9864 | 0.1026 | 0.121 | -0.0184 |
| dpctgan_eps10_seed0 | nyha_nyha_pET | 1.0 | 0.0 | 0.0 | +0.0000 |
| dpctgan_eps15_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9986 | 0.5719 | 0.5681 | +0.0037 |
| dpctgan_eps15_seed0 | ckd_severity_from_calculated_egfr | 0.994 | 0.2884 | 0.2888 | -0.0004 |
| dpctgan_eps15_seed0 | nyha_nyha_pET | 1.0 | 0.0 | 0.0 | +0.0000 |
| dpctgan_eps15_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 1.0 | 0.5719 | 0.5681 | +0.0037 |
| dpctgan_eps15_seed1 | ckd_severity_from_calculated_egfr | 0.9898 | 0.2003 | 0.2121 | -0.0118 |
| dpctgan_eps15_seed1 | nyha_nyha_pET | 1.0 | 0.0 | 0.0 | +0.0000 |
| dpctgan_eps15_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.973 | 0.5611 | 0.5613 | -0.0002 |
| dpctgan_eps15_seed2 | ckd_severity_from_calculated_egfr | 0.9849 | 0.1818 | 0.167 | +0.0149 |
| dpctgan_eps15_seed2 | nyha_nyha_pET | 1.0 | 0.0 | 0.0 | +0.0000 |
| dpctgan_eps1_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9983 | 0.5713 | 0.5673 | +0.0040 |
| dpctgan_eps1_seed0 | ckd_severity_from_calculated_egfr | 0.9486 | 0.2827 | 0.276 | +0.0067 |
| dpctgan_eps1_seed0 | nyha_nyha_pET | 0.6358 | 0.0 | 0.0 | +0.0000 |
| dpctgan_eps20_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9955 | 0.573 | 0.5673 | +0.0057 |
| dpctgan_eps20_seed0 | ckd_severity_from_calculated_egfr | 0.979 | 0.179 | 0.1874 | -0.0084 |
| dpctgan_eps20_seed0 | nyha_nyha_pET | 0.6088 | 0.0 | 0.0 | +0.0000 |
| dpctgan_eps5_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 1.0 | 0.5719 | 0.5681 | +0.0037 |
| dpctgan_eps5_seed0 | ckd_severity_from_calculated_egfr | 0.9983 | 0.2895 | 0.2905 | -0.0010 |
| dpctgan_eps5_seed0 | nyha_nyha_pET | 1.0 | 0.0 | 0.0 | +0.0000 |
| dpctgan_eps8_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.7455 | 0.4355 | 0.4438 | -0.0083 |
| dpctgan_eps8_seed0 | ckd_severity_from_calculated_egfr | 0.9938 | 0.177 | 0.1789 | -0.0019 |
| dpctgan_eps8_seed0 | nyha_nyha_pET | 1.0 | 0.0 | 0.0 | +0.0000 |
| gaussian_copula_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.556 | 0.4991 | 0.5 | -0.0009 |
| gaussian_copula_seed0 | ckd_severity_from_calculated_egfr | 0.3082 | 0.2114 | 0.2147 | -0.0033 |
| gaussian_copula_seed0 | nyha_nyha_pET | 0.7676 | 0.0 | 0.0 | +0.0000 |
| gaussian_copula_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.5824 | 0.5085 | 0.4932 | +0.0153 |
| gaussian_copula_seed1 | ckd_severity_from_calculated_egfr | 0.2932 | 0.2085 | 0.2257 | -0.0172 |
| gaussian_copula_seed1 | nyha_nyha_pET | 0.7676 | 0.0 | 0.0 | +0.0000 |
| gaussian_copula_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.5616 | 0.5108 | 0.4813 | +0.0295 |
| gaussian_copula_seed2 | ckd_severity_from_calculated_egfr | 0.3003 | 0.2207 | 0.1874 | +0.0333 |
| gaussian_copula_seed2 | nyha_nyha_pET | 0.7602 | 0.0 | 0.0 | +0.0000 |
| mst_eps0p5_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6599 | 0.5602 | 0.563 | -0.0028 |
| mst_eps0p5_seed0 | ckd_severity_from_calculated_egfr | 0.2793 | 0.2003 | 0.1772 | +0.0231 |
| mst_eps0p5_seed0 | nyha_nyha_pET | 0.5915 | 0.0 | 0.0 | +0.0000 |
| mst_eps10_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5724 | 0.4526 | 0.4634 | -0.0108 |
| mst_eps10_seed0 | ckd_severity_from_calculated_egfr | 0.2938 | 0.1727 | 0.1491 | +0.0237 |
| mst_eps10_seed0 | nyha_nyha_pET | 0.7565 | 0.0 | 0.0 | +0.0000 |
| mst_eps15_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.573 | 0.4599 | 0.4523 | +0.0076 |
| mst_eps15_seed0 | ckd_severity_from_calculated_egfr | 0.2889 | 0.2159 | 0.2215 | -0.0056 |
| mst_eps15_seed0 | nyha_nyha_pET | 0.7622 | 0.0 | 0.0 | +0.0000 |
| mst_eps15_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.573 | 0.4935 | 0.4898 | +0.0037 |
| mst_eps15_seed1 | ckd_severity_from_calculated_egfr | 0.2881 | 0.1781 | 0.1687 | +0.0095 |
| mst_eps15_seed1 | nyha_nyha_pET | 0.7628 | 0.0 | 0.0 | +0.0000 |
| mst_eps15_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.5693 | 0.4205 | 0.4327 | -0.0123 |
| mst_eps15_seed2 | ckd_severity_from_calculated_egfr | 0.2895 | 0.1815 | 0.2002 | -0.0186 |
| mst_eps15_seed2 | nyha_nyha_pET | 0.7628 | 0.0 | 0.0 | +0.0000 |
| mst_eps1_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5656 | 0.4966 | 0.5009 | -0.0043 |
| mst_eps1_seed0 | ckd_severity_from_calculated_egfr | 0.2807 | 0.1398 | 0.1576 | -0.0178 |
| mst_eps1_seed0 | nyha_nyha_pET | 0.7304 | 0.0 | 0.0 | +0.0000 |
| mst_eps20_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5716 | 0.4415 | 0.4387 | +0.0028 |
| mst_eps20_seed0 | ckd_severity_from_calculated_egfr | 0.2909 | 0.1787 | 0.1704 | +0.0083 |
| mst_eps20_seed0 | nyha_nyha_pET | 0.7597 | 0.0 | 0.0 | +0.0000 |
| mst_eps5_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5733 | 0.5312 | 0.5162 | +0.0151 |
| mst_eps5_seed0 | ckd_severity_from_calculated_egfr | 0.2847 | 0.1776 | 0.1874 | -0.0098 |
| mst_eps5_seed0 | nyha_nyha_pET | 0.7722 | 0.0 | 0.0 | +0.0000 |
| mst_eps8_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5776 | 0.5338 | 0.5298 | +0.0040 |
| mst_eps8_seed0 | ckd_severity_from_calculated_egfr | 0.2906 | 0.1764 | 0.1993 | -0.0229 |
| mst_eps8_seed0 | nyha_nyha_pET | 0.7548 | 0.0 | 0.0 | +0.0000 |
| patectgan_eps15_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9926 | 0.573 | 0.5673 | +0.0057 |
| patectgan_eps15_seed0 | ckd_severity_from_calculated_egfr | 0.3537 | 0.2381 | 0.23 | +0.0081 |
| patectgan_eps15_seed0 | nyha_nyha_pET | 0.7205 | 0.0 | 0.0 | +0.0000 |
| patectgan_eps1_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5423 | 0.444 | 0.4361 | +0.0079 |
| patectgan_eps1_seed0 | ckd_severity_from_calculated_egfr | 0.2119 | 0.1747 | 0.1831 | -0.0084 |
| patectgan_eps1_seed0 | nyha_nyha_pET | 0.6614 | 0.0 | 0.0 | +0.0000 |
| patectgan_eps5_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9835 | 0.5719 | 0.5698 | +0.0020 |
| patectgan_eps5_seed0 | ckd_severity_from_calculated_egfr | 0.3239 | 0.2216 | 0.2164 | +0.0052 |
| patectgan_eps5_seed0 | nyha_nyha_pET | 0.6514 | 0.0 | 0.0 | +0.0000 |
| tvae_cap256_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.4929 | 0.5415 | 0.5375 | +0.0040 |
| tvae_cap256_seed0 | ckd_severity_from_calculated_egfr | 0.3196 | 0.2514 | 0.2411 | +0.0104 |
| tvae_cap256_seed0 | nyha_nyha_pET | 0.9634 | 0.0 | 0.0 | +0.0000 |
| tvae_ep1000_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5426 | 0.5426 | 0.5664 | -0.0238 |
| tvae_ep1000_seed0 | ckd_severity_from_calculated_egfr | 0.4327 | 0.2449 | 0.2342 | +0.0106 |
| tvae_ep1000_seed0 | nyha_nyha_pET | 0.9574 | 0.0 | 0.0 | +0.0000 |
| tvae_ind_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5159 | 0.546 | 0.5273 | +0.0188 |
| tvae_ind_seed0 | ckd_severity_from_calculated_egfr | 0.3489 | 0.2253 | 0.2521 | -0.0268 |
| tvae_ind_seed0 | nyha_nyha_pET | 0.9812 | 0.0 | 0.0 | +0.0000 |
| tvae_qt_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5159 | 0.5369 | 0.5247 | +0.0122 |
| tvae_qt_seed0 | ckd_severity_from_calculated_egfr | 0.2841 | 0.2534 | 0.2513 | +0.0021 |
| tvae_qt_seed0 | nyha_nyha_pET | 0.9199 | 0.0 | 0.0 | +0.0000 |
| tvae_qt_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.5202 | 0.5196 | 0.5392 | -0.0196 |
| tvae_qt_seed1 | ckd_severity_from_calculated_egfr | 0.2668 | 0.248 | 0.2283 | +0.0197 |
| tvae_qt_seed1 | nyha_nyha_pET | 0.9355 | 0.0 | 0.0 | +0.0000 |
| tvae_qt_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.5017 | 0.5114 | 0.4983 | +0.0131 |
| tvae_qt_seed2 | ckd_severity_from_calculated_egfr | 0.2872 | 0.2452 | 0.2589 | -0.0138 |
| tvae_qt_seed2 | nyha_nyha_pET | 0.9378 | 0.0 | 0.0 | +0.0000 |
| tvae_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5259 | 0.5284 | 0.5298 | -0.0014 |
| tvae_seed0 | ckd_severity_from_calculated_egfr | 0.3392 | 0.2628 | 0.2266 | +0.0362 |
| tvae_seed0 | nyha_nyha_pET | 0.9631 | 0.0 | 0.0 | +0.0000 |
| tvae_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.5344 | 0.5574 | 0.5366 | +0.0208 |
| tvae_seed1 | ckd_severity_from_calculated_egfr | 0.3051 | 0.2537 | 0.2368 | +0.0169 |
| tvae_seed1 | nyha_nyha_pET | 0.9619 | 0.0 | 0.0 | +0.0000 |
| tvae_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.5165 | 0.5355 | 0.5111 | +0.0244 |
| tvae_seed2 | ckd_severity_from_calculated_egfr | 0.331 | 0.2378 | 0.2428 | -0.0050 |
| tvae_seed2 | nyha_nyha_pET | 0.977 | 0.0 | 0.0 | +0.0000 |

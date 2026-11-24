# Adversarial Privacy Attacks

Members: 3520 training records; non-members: 1174 holdout records (real, unseen patients). Membership inference AUC of 0.5 means the synthetic data reveals nothing about who was in the training set. Attribute inference reports the MEMBERSHIP ADVANTAGE -- accuracy on members minus accuracy on non-members; population-level inference (both above baseline, equally) is the intended use of released data, only member-specific advantage is leakage.

| run | MIA AUC (95% CI) | learned MIA AUC (95% CI) | empirical ε̂ lower bound | worst AIA membership advantage | anonymeter |
|---|---|---|---|---|---|
| aim40_eps1_seed0 | 0.4984 (0.4793-0.5173) | 0.4722 (0.4518-0.49) | 0.1652 | 0.0037 | SO 0.0, link 0.0 |
| aim50_eps1_seed0 | 0.4988 (0.4796-0.5174) | 0.4773 (0.4593-0.498) | 0.0953 | 0.05 | SO 0.0071, link 0.0 |
| ctgan_qt_seed0 | 0.5141 (0.4964-0.5316) | 0.5046 (0.4856-0.5228) | 0.0096 | 0.0117 | SO 0.0, link 0.0 |
| ctgan_seed0 | 0.496 (0.4759-0.5159) | 0.4921 (0.4753-0.5097) | 0.0 | 0.0227 | SO 0.0, link 0.0 |
| ctgan_seed1 | 0.4991 (0.4802-0.5176) | 0.4648 (0.4459-0.4815) | 0.08 | 0.0216 | SO 0.0099, link 0.0 |
| ctgan_seed2 | 0.518 (0.4993-0.5361) | 0.5148 (0.4951-0.5323) | 0.0335 | 0.0407 | SO 0.0066, link 0.005 |
| ddpm_g_seed0 | 0.5098 (0.4906-0.5282) | 0.4873 (0.4683-0.5064) | 0.0638 | 0.0328 | SO 0.0, link 0.0 |
| ddpm_seed0 | 0.5045 (0.4874-0.5235) | 0.4732 (0.4552-0.4912) | 0.0 | 0.0221 | SO 0.0, link 0.0 |
| ddpm_seed1 | 0.515 (0.497-0.5327) | 0.5019 (0.4838-0.5188) | 0.0 | 0.0041 | SO 0.0, link 0.0 |
| ddpm_seed2 | 0.506 (0.4869-0.5222) | 0.4724 (0.4549-0.4905) | 0.0 | 0.0334 | SO 0.0, link 0.0 |
| dpctgan_eps10_seed0 | 0.4988 (0.4798-0.5181) | 0.4978 (0.475-0.5161) | 0.0 | 0.0339 | SO 0.0, link 0.0 |
| dpctgan_eps15_seed0 | 0.4964 (0.478-0.5138) | 0.4729 (0.454-0.4918) | 0.0 | 0.0339 | SO 0.0, link 0.0 |
| dpctgan_eps15_seed1 | 0.5029 (0.4831-0.5214) | 0.4815 (0.4641-0.5012) | 0.0 | 0.0339 | SO 0.0, link 0.0 |
| dpctgan_eps15_seed2 | 0.4975 (0.4797-0.5151) | 0.4729 (0.4554-0.4921) | 0.0 | 0.0339 | SO 0.0, link 0.0 |
| dpctgan_eps1_seed0 | 0.51 (0.4893-0.5291) | 0.5057 (0.4887-0.5241) | 0.0042 | 0.0037 | SO 0.0, link 0.0 |
| dpctgan_eps20_seed0 | 0.5026 (0.4847-0.5218) | 0.5152 (0.4969-0.5326) | 0.0272 | 0.0037 | SO 0.0, link 0.0 |
| dpctgan_eps5_seed0 | 0.5014 (0.4838-0.5189) | 0.5038 (0.4855-0.5239) | 0.0 | 0.0339 | SO 0.0, link 0.0 |
| dpctgan_eps8_seed0 | 0.4996 (0.4806-0.5188) | 0.4835 (0.4665-0.503) | 0.0 | 0.0339 | SO 0.0, link 0.0 |
| gaussian_copula_seed0 | 0.5076 (0.4882-0.5254) | 0.4817 (0.4635-0.503) | 0.0034 | 0.0162 | SO 0.0, link 0.0 |
| gaussian_copula_seed1 | 0.5092 (0.4905-0.5292) | 0.4807 (0.4633-0.5002) | 0.0 | 0.0279 | SO 0.0, link 0.0 |
| gaussian_copula_seed2 | 0.5082 (0.4896-0.5256) | 0.4914 (0.4735-0.5093) | 0.0091 | 0.039 | SO 0.0017, link 0.0 |
| mst_eps0p5_seed0 | 0.5003 (0.4818-0.5186) | 0.4774 (0.4595-0.4968) | 0.0 | 0.0154 | SO 0.005, link 0.0 |
| mst_eps10_seed0 | 0.5063 (0.4866-0.5262) | 0.4825 (0.4638-0.5005) | 0.0 | 0.0297 | SO 0.0149, link 0.0 |
| mst_eps15_seed0 | 0.5025 (0.4838-0.5212) | 0.4969 (0.4793-0.5157) | 0.0 | 0.0125 | SO 0.0182, link 0.0 |
| mst_eps15_seed1 | 0.5006 (0.4824-0.5213) | 0.4744 (0.4555-0.492) | 0.0 | 0.0277 | SO 0.0215, link 0.0 |
| mst_eps15_seed2 | 0.5078 (0.4888-0.5266) | 0.4993 (0.4807-0.5178) | 0.0 | 0.0415 | SO 0.0249, link 0.0 |
| mst_eps1_seed0 | 0.5138 (0.4962-0.5318) | 0.4962 (0.4777-0.5153) | 0.0049 | 0.0017 | SO 0.0198, link 0.0 |
| mst_eps20_seed0 | 0.5089 (0.4878-0.5262) | 0.5091 (0.4877-0.5283) | 0.0077 | 0.0268 | SO 0.0529, link 0.0 |
| mst_eps5_seed0 | 0.5026 (0.4841-0.5195) | 0.4792 (0.4614-0.4989) | 0.0 | 0.0259 | SO 0.0083, link 0.0 |
| mst_eps8_seed0 | 0.5178 (0.4999-0.5397) | 0.5022 (0.4838-0.5206) | 0.0368 | 0.0254 | SO 0.0182, link 0.0 |
| patectgan_eps15_seed0 | 0.5035 (0.4836-0.524) | 0.4653 (0.4456-0.4829) | 0.0 | 0.0284 | SO 0.005, link 0.0 |
| patectgan_eps1_seed0 | 0.5072 (0.4889-0.5253) | 0.4832 (0.4636-0.5029) | 0.0 | 0.0133 | SO 0.0, link 0.0 |
| patectgan_eps5_seed0 | 0.5037 (0.4844-0.5237) | 0.5095 (0.4904-0.5303) | 0.0 | 0.0037 | SO 0.0033, link 0.0 |
| tvae_cap256_seed0 | 0.5314 (0.5117-0.5478) | 0.521 (0.502-0.5404) | 0.2386 | 0.0282 | SO 0.0017, link 0.0 |
| tvae_ep1000_seed0 | 0.5245 (0.5074-0.5446) | 0.5234 (0.5025-0.5421) | 0.033 | 0.026 | SO 0.0033, link 0.0 |
| tvae_ind_seed0 | 0.5216 (0.5031-0.5411) | 0.5077 (0.4886-0.5277) | 0.0406 | 0.0348 | SO 0.0066, link 0.0 |
| tvae_qt_seed0 | 0.5221 (0.5044-0.5388) | 0.5089 (0.4897-0.5272) | 0.0365 | 0.0396 | SO 0.0, link 0.0 |
| tvae_qt_seed1 | 0.5221 (0.5038-0.5434) | 0.5098 (0.4911-0.5264) | 0.0245 | 0.0302 | SO 0.0, link 0.0 |
| tvae_qt_seed2 | 0.5224 (0.5028-0.5376) | 0.5091 (0.49-0.527) | 0.0339 | 0.0285 | SO 0.0, link 0.0 |
| tvae_seed0 | 0.5205 (0.5018-0.5395) | 0.5127 (0.4951-0.5322) | 0.0364 | 0.0311 | SO 0.005, link 0.0 |
| tvae_seed1 | 0.5252 (0.5067-0.5427) | 0.5154 (0.4956-0.5336) | 0.0163 | 0.0302 | SO 0.0, link 0.0 |
| tvae_seed2 | 0.5157 (0.497-0.5348) | 0.5079 (0.489-0.5261) | 0.0 | 0.0285 | SO 0.0, link 0.0 |

## Who is at risk: membership inference by patient atypicality

WITHIN-STRATUM AUC: members and non-members both get the same atypicality score (distance to their 5th-nearest member, same reference set and encoder), non-members are binned by the member quartile cut points, and each stratum's AUC compares members-in-Qi against non-members-in-Qi only. 0.5 = no leakage on that stratum; elevated values indicate SELECTIVE leakage on that stratum (e.g. a model that memorizes its unusual patients shows it in Q4). Strata with fewer than 30 non-members are skipped ('-'). Cell format: AUC (n members / n non-members).

| run | Q1 typical | Q2 | Q3 | Q4 atypical |
|---|---|---|---|---|
| aim40_eps1_seed0 | 0.5187 (880/262) | 0.4801 (880/309) | 0.5031 (880/297) | 0.4671 (880/306) |
| aim50_eps1_seed0 | 0.5198 (880/262) | 0.4783 (880/309) | 0.5004 (880/297) | 0.4725 (880/306) |
| ctgan_qt_seed0 | 0.5051 (880/262) | 0.5244 (880/309) | 0.5332 (880/297) | 0.4566 (880/306) |
| ctgan_seed0 | 0.4793 (880/262) | 0.5006 (880/309) | 0.495 (880/297) | 0.4555 (880/306) |
| ctgan_seed1 | 0.5046 (880/262) | 0.5065 (880/309) | 0.4724 (880/297) | 0.4529 (880/306) |
| ctgan_seed2 | 0.5083 (880/262) | 0.5285 (880/309) | 0.5072 (880/297) | 0.4767 (880/306) |
| ddpm_g_seed0 | 0.5163 (880/262) | 0.504 (880/309) | 0.4956 (880/297) | 0.4785 (880/306) |
| ddpm_seed0 | 0.5143 (880/262) | 0.4913 (880/309) | 0.502 (880/297) | 0.4609 (880/306) |
| ddpm_seed1 | 0.5086 (880/262) | 0.5237 (880/309) | 0.5174 (880/297) | 0.4757 (880/306) |
| ddpm_seed2 | 0.5187 (880/262) | 0.48 (880/309) | 0.5201 (880/297) | 0.4698 (880/306) |
| dpctgan_eps10_seed0 | 0.4968 (880/262) | 0.4847 (880/309) | 0.5067 (880/297) | 0.4738 (880/306) |
| dpctgan_eps15_seed0 | 0.5123 (880/262) | 0.4731 (880/309) | 0.4903 (880/297) | 0.4673 (880/306) |
| dpctgan_eps15_seed1 | 0.51 (880/262) | 0.4775 (880/309) | 0.4951 (880/297) | 0.4868 (880/306) |
| dpctgan_eps15_seed2 | 0.5146 (880/262) | 0.482 (880/309) | 0.4905 (880/297) | 0.4731 (880/306) |
| dpctgan_eps1_seed0 | 0.5154 (880/262) | 0.516 (880/309) | 0.5085 (880/297) | 0.4656 (880/306) |
| dpctgan_eps20_seed0 | 0.5005 (880/262) | 0.5134 (880/309) | 0.4992 (880/297) | 0.4774 (880/306) |
| dpctgan_eps5_seed0 | 0.4868 (880/262) | 0.4828 (880/309) | 0.5062 (880/297) | 0.4665 (880/306) |
| dpctgan_eps8_seed0 | 0.5201 (880/262) | 0.4683 (880/309) | 0.5045 (880/297) | 0.4728 (880/306) |
| gaussian_copula_seed0 | 0.516 (880/262) | 0.4976 (880/309) | 0.4982 (880/297) | 0.4731 (880/306) |
| gaussian_copula_seed1 | 0.4993 (880/262) | 0.5063 (880/309) | 0.4967 (880/297) | 0.4656 (880/306) |
| gaussian_copula_seed2 | 0.5142 (880/262) | 0.5085 (880/309) | 0.5044 (880/297) | 0.4599 (880/306) |
| mst_eps0p5_seed0 | 0.4905 (880/262) | 0.471 (880/309) | 0.5123 (880/297) | 0.4711 (880/306) |
| mst_eps10_seed0 | 0.4927 (880/262) | 0.4702 (880/309) | 0.5221 (880/297) | 0.4962 (880/306) |
| mst_eps15_seed0 | 0.4819 (880/262) | 0.4966 (880/309) | 0.5117 (880/297) | 0.4874 (880/306) |
| mst_eps15_seed1 | 0.4821 (880/262) | 0.4938 (880/309) | 0.5076 (880/297) | 0.48 (880/306) |
| mst_eps15_seed2 | 0.4946 (880/262) | 0.478 (880/309) | 0.5091 (880/297) | 0.5067 (880/306) |
| mst_eps1_seed0 | 0.5252 (880/262) | 0.5144 (880/309) | 0.5098 (880/297) | 0.4877 (880/306) |
| mst_eps20_seed0 | 0.5004 (880/262) | 0.473 (880/309) | 0.5228 (880/297) | 0.4936 (880/306) |
| mst_eps5_seed0 | 0.4821 (880/262) | 0.4637 (880/309) | 0.5071 (880/297) | 0.5062 (880/306) |
| mst_eps8_seed0 | 0.5072 (880/262) | 0.5219 (880/309) | 0.529 (880/297) | 0.4953 (880/306) |
| patectgan_eps15_seed0 | 0.5008 (880/262) | 0.515 (880/309) | 0.4938 (880/297) | 0.4488 (880/306) |
| patectgan_eps1_seed0 | 0.5152 (880/262) | 0.4961 (880/309) | 0.5143 (880/297) | 0.4769 (880/306) |
| patectgan_eps5_seed0 | 0.5213 (880/262) | 0.5214 (880/309) | 0.4965 (880/297) | 0.4629 (880/306) |
| tvae_cap256_seed0 | 0.5292 (880/262) | 0.5305 (880/309) | 0.5538 (880/297) | 0.5296 (880/306) |
| tvae_ep1000_seed0 | 0.5013 (880/262) | 0.5219 (880/309) | 0.5586 (880/297) | 0.4985 (880/306) |
| tvae_ind_seed0 | 0.4915 (880/262) | 0.5248 (880/309) | 0.552 (880/297) | 0.493 (880/306) |
| tvae_qt_seed0 | 0.5059 (880/262) | 0.5259 (880/309) | 0.5518 (880/297) | 0.4849 (880/306) |
| tvae_qt_seed1 | 0.4885 (880/262) | 0.5183 (880/309) | 0.5519 (880/297) | 0.4989 (880/306) |
| tvae_qt_seed2 | 0.4964 (880/262) | 0.5253 (880/309) | 0.5392 (880/297) | 0.5107 (880/306) |
| tvae_seed0 | 0.4851 (880/262) | 0.515 (880/309) | 0.5474 (880/297) | 0.4965 (880/306) |
| tvae_seed1 | 0.4996 (880/262) | 0.5185 (880/309) | 0.5595 (880/297) | 0.5074 (880/306) |
| tvae_seed2 | 0.4921 (880/262) | 0.5098 (880/309) | 0.5312 (880/297) | 0.4957 (880/306) |

## Attribute inference detail

| run | sensitive attribute | baseline | member acc | non-member acc | advantage |
|---|---|---|---|---|---|
| aim40_eps1_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.5719 | 0.5681 | +0.0037 |
| aim40_eps1_seed0 | ckd_severity_from_calculated_egfr | 0.2905 | 0.1932 | 0.1951 | -0.0019 |
| aim40_eps1_seed0 | nyha_nyha_pET | 0.7266 | 0.5969 | 0.5963 | +0.0006 |
| aim50_eps1_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.5719 | 0.5681 | +0.0037 |
| aim50_eps1_seed0 | ckd_severity_from_calculated_egfr | 0.2905 | 0.2139 | 0.1959 | +0.0180 |
| aim50_eps1_seed0 | nyha_nyha_pET | 0.7266 | 0.631 | 0.5809 | +0.0500 |
| ctgan_qt_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.4653 | 0.4847 | -0.0193 |
| ctgan_qt_seed0 | ckd_severity_from_calculated_egfr | 0.2905 | 0.2131 | 0.2053 | +0.0078 |
| ctgan_qt_seed0 | nyha_nyha_pET | 0.7266 | 0.6659 | 0.6542 | +0.0117 |
| ctgan_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.4452 | 0.4225 | +0.0227 |
| ctgan_seed0 | ckd_severity_from_calculated_egfr | 0.2905 | 0.1957 | 0.1857 | +0.0100 |
| ctgan_seed0 | nyha_nyha_pET | 0.7266 | 0.6457 | 0.6269 | +0.0188 |
| ctgan_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.4946 | 0.506 | -0.0114 |
| ctgan_seed1 | ckd_severity_from_calculated_egfr | 0.2905 | 0.2151 | 0.2257 | -0.0107 |
| ctgan_seed1 | nyha_nyha_pET | 0.7266 | 0.5957 | 0.5741 | +0.0216 |
| ctgan_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.4966 | 0.4702 | +0.0264 |
| ctgan_seed2 | ckd_severity_from_calculated_egfr | 0.2905 | 0.1855 | 0.1789 | +0.0066 |
| ctgan_seed2 | nyha_nyha_pET | 0.7266 | 0.6753 | 0.6346 | +0.0407 |
| ddpm_g_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.3841 | 0.3756 | +0.0085 |
| ddpm_g_seed0 | ckd_severity_from_calculated_egfr | 0.2905 | 0.1895 | 0.1567 | +0.0328 |
| ddpm_g_seed0 | nyha_nyha_pET | 0.7266 | 0.2983 | 0.2785 | +0.0198 |
| ddpm_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.3034 | 0.293 | +0.0104 |
| ddpm_seed0 | ckd_severity_from_calculated_egfr | 0.2905 | 0.1963 | 0.1925 | +0.0038 |
| ddpm_seed0 | nyha_nyha_pET | 0.7266 | 0.4284 | 0.4063 | +0.0221 |
| ddpm_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.3929 | 0.4003 | -0.0074 |
| ddpm_seed1 | ckd_severity_from_calculated_egfr | 0.2905 | 0.1756 | 0.1772 | -0.0016 |
| ddpm_seed1 | nyha_nyha_pET | 0.7266 | 0.3057 | 0.3015 | +0.0041 |
| ddpm_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.369 | 0.3356 | +0.0334 |
| ddpm_seed2 | ckd_severity_from_calculated_egfr | 0.2905 | 0.1628 | 0.1508 | +0.0120 |
| ddpm_seed2 | nyha_nyha_pET | 0.7266 | 0.5219 | 0.5068 | +0.0151 |
| dpctgan_eps10_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.5645 | 0.5664 | -0.0020 |
| dpctgan_eps10_seed0 | ckd_severity_from_calculated_egfr | 0.2905 | 0.1028 | 0.121 | -0.0181 |
| dpctgan_eps10_seed0 | nyha_nyha_pET | 0.7266 | 0.7605 | 0.7266 | +0.0339 |
| dpctgan_eps15_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.5719 | 0.5681 | +0.0037 |
| dpctgan_eps15_seed0 | ckd_severity_from_calculated_egfr | 0.2905 | 0.2895 | 0.2905 | -0.0010 |
| dpctgan_eps15_seed0 | nyha_nyha_pET | 0.7266 | 0.7605 | 0.7266 | +0.0339 |
| dpctgan_eps15_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.5719 | 0.5681 | +0.0037 |
| dpctgan_eps15_seed1 | ckd_severity_from_calculated_egfr | 0.2905 | 0.2003 | 0.2121 | -0.0118 |
| dpctgan_eps15_seed1 | nyha_nyha_pET | 0.7266 | 0.7605 | 0.7266 | +0.0339 |
| dpctgan_eps15_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.5719 | 0.5681 | +0.0037 |
| dpctgan_eps15_seed2 | ckd_severity_from_calculated_egfr | 0.2905 | 0.1909 | 0.1618 | +0.0291 |
| dpctgan_eps15_seed2 | nyha_nyha_pET | 0.7266 | 0.7605 | 0.7266 | +0.0339 |
| dpctgan_eps1_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.5719 | 0.5681 | +0.0037 |
| dpctgan_eps1_seed0 | ckd_severity_from_calculated_egfr | 0.2905 | 0.2886 | 0.2879 | +0.0007 |
| dpctgan_eps1_seed0 | nyha_nyha_pET | 0.7266 | 0.0935 | 0.109 | -0.0156 |
| dpctgan_eps20_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.5719 | 0.5681 | +0.0037 |
| dpctgan_eps20_seed0 | ckd_severity_from_calculated_egfr | 0.2905 | 0.1778 | 0.1772 | +0.0007 |
| dpctgan_eps20_seed0 | nyha_nyha_pET | 0.7266 | 0.0832 | 0.1031 | -0.0198 |
| dpctgan_eps5_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.5719 | 0.5681 | +0.0037 |
| dpctgan_eps5_seed0 | ckd_severity_from_calculated_egfr | 0.2905 | 0.2895 | 0.2905 | -0.0010 |
| dpctgan_eps5_seed0 | nyha_nyha_pET | 0.7266 | 0.7605 | 0.7266 | +0.0339 |
| dpctgan_eps8_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.4037 | 0.4114 | -0.0077 |
| dpctgan_eps8_seed0 | ckd_severity_from_calculated_egfr | 0.2905 | 0.177 | 0.1789 | -0.0019 |
| dpctgan_eps8_seed0 | nyha_nyha_pET | 0.7266 | 0.7605 | 0.7266 | +0.0339 |
| gaussian_copula_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.4838 | 0.5332 | -0.0494 |
| gaussian_copula_seed0 | ckd_severity_from_calculated_egfr | 0.2905 | 0.2105 | 0.2027 | +0.0078 |
| gaussian_copula_seed0 | nyha_nyha_pET | 0.7266 | 0.5878 | 0.5716 | +0.0162 |
| gaussian_copula_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.5256 | 0.5273 | -0.0017 |
| gaussian_copula_seed1 | ckd_severity_from_calculated_egfr | 0.2905 | 0.2159 | 0.2019 | +0.0140 |
| gaussian_copula_seed1 | nyha_nyha_pET | 0.7266 | 0.5952 | 0.5673 | +0.0279 |
| gaussian_copula_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.5057 | 0.5043 | +0.0014 |
| gaussian_copula_seed2 | ckd_severity_from_calculated_egfr | 0.2905 | 0.2051 | 0.2078 | -0.0027 |
| gaussian_copula_seed2 | nyha_nyha_pET | 0.7266 | 0.6045 | 0.5656 | +0.0390 |
| mst_eps0p5_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.5565 | 0.5613 | -0.0048 |
| mst_eps0p5_seed0 | ckd_severity_from_calculated_egfr | 0.2905 | 0.1974 | 0.1823 | +0.0152 |
| mst_eps0p5_seed0 | nyha_nyha_pET | 0.7266 | 0.5955 | 0.5801 | +0.0154 |
| mst_eps10_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.4466 | 0.4421 | +0.0045 |
| mst_eps10_seed0 | ckd_severity_from_calculated_egfr | 0.2905 | 0.1719 | 0.161 | +0.0109 |
| mst_eps10_seed0 | nyha_nyha_pET | 0.7266 | 0.7256 | 0.6959 | +0.0297 |
| mst_eps15_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.4562 | 0.4438 | +0.0125 |
| mst_eps15_seed0 | ckd_severity_from_calculated_egfr | 0.2905 | 0.2151 | 0.2274 | -0.0124 |
| mst_eps15_seed0 | nyha_nyha_pET | 0.7266 | 0.6687 | 0.6576 | +0.0112 |
| mst_eps15_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.4767 | 0.4736 | +0.0031 |
| mst_eps15_seed1 | ckd_severity_from_calculated_egfr | 0.2905 | 0.177 | 0.1661 | +0.0109 |
| mst_eps15_seed1 | nyha_nyha_pET | 0.7266 | 0.7278 | 0.7002 | +0.0277 |
| mst_eps15_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.4173 | 0.4225 | -0.0052 |
| mst_eps15_seed2 | ckd_severity_from_calculated_egfr | 0.2905 | 0.1841 | 0.2087 | -0.0246 |
| mst_eps15_seed2 | nyha_nyha_pET | 0.7266 | 0.6582 | 0.6167 | +0.0415 |
| mst_eps1_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.5074 | 0.5111 | -0.0037 |
| mst_eps1_seed0 | ckd_severity_from_calculated_egfr | 0.2905 | 0.1435 | 0.155 | -0.0116 |
| mst_eps1_seed0 | nyha_nyha_pET | 0.7266 | 0.5332 | 0.5315 | +0.0017 |
| mst_eps20_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.442 | 0.4506 | -0.0086 |
| mst_eps20_seed0 | ckd_severity_from_calculated_egfr | 0.2905 | 0.1855 | 0.1763 | +0.0092 |
| mst_eps20_seed0 | nyha_nyha_pET | 0.7266 | 0.7125 | 0.6857 | +0.0268 |
| mst_eps5_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.5193 | 0.5179 | +0.0014 |
| mst_eps5_seed0 | ckd_severity_from_calculated_egfr | 0.2905 | 0.1793 | 0.1934 | -0.0141 |
| mst_eps5_seed0 | nyha_nyha_pET | 0.7266 | 0.6869 | 0.661 | +0.0259 |
| mst_eps8_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.5298 | 0.5315 | -0.0017 |
| mst_eps8_seed0 | ckd_severity_from_calculated_egfr | 0.2905 | 0.1747 | 0.1968 | -0.0220 |
| mst_eps8_seed0 | nyha_nyha_pET | 0.7266 | 0.7136 | 0.6882 | +0.0254 |
| patectgan_eps15_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.5699 | 0.5656 | +0.0043 |
| patectgan_eps15_seed0 | ckd_severity_from_calculated_egfr | 0.2905 | 0.2389 | 0.2249 | +0.0140 |
| patectgan_eps15_seed0 | nyha_nyha_pET | 0.7266 | 0.5591 | 0.5307 | +0.0284 |
| patectgan_eps1_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.4409 | 0.4276 | +0.0133 |
| patectgan_eps1_seed0 | ckd_severity_from_calculated_egfr | 0.2905 | 0.1716 | 0.1891 | -0.0175 |
| patectgan_eps1_seed0 | nyha_nyha_pET | 0.7266 | 0.2057 | 0.2428 | -0.0371 |
| patectgan_eps5_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.5693 | 0.5767 | -0.0073 |
| patectgan_eps5_seed0 | ckd_severity_from_calculated_egfr | 0.2905 | 0.2224 | 0.2198 | +0.0027 |
| patectgan_eps5_seed0 | nyha_nyha_pET | 0.7266 | 0.4986 | 0.4949 | +0.0037 |
| tvae_cap256_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.5415 | 0.5375 | +0.0040 |
| tvae_cap256_seed0 | ckd_severity_from_calculated_egfr | 0.2905 | 0.2514 | 0.2411 | +0.0104 |
| tvae_cap256_seed0 | nyha_nyha_pET | 0.7266 | 0.7412 | 0.7129 | +0.0282 |
| tvae_ep1000_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.5364 | 0.5247 | +0.0117 |
| tvae_ep1000_seed0 | ckd_severity_from_calculated_egfr | 0.2905 | 0.2378 | 0.2479 | -0.0101 |
| tvae_ep1000_seed0 | nyha_nyha_pET | 0.7266 | 0.7244 | 0.6985 | +0.0260 |
| tvae_ind_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.546 | 0.5273 | +0.0188 |
| tvae_ind_seed0 | ckd_severity_from_calculated_egfr | 0.2905 | 0.2253 | 0.2521 | -0.0268 |
| tvae_ind_seed0 | nyha_nyha_pET | 0.7266 | 0.7486 | 0.7138 | +0.0348 |
| tvae_qt_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.5199 | 0.5179 | +0.0020 |
| tvae_qt_seed0 | ckd_severity_from_calculated_egfr | 0.2905 | 0.2477 | 0.247 | +0.0007 |
| tvae_qt_seed0 | nyha_nyha_pET | 0.7266 | 0.7347 | 0.6951 | +0.0396 |
| tvae_qt_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.5278 | 0.5162 | +0.0117 |
| tvae_qt_seed1 | ckd_severity_from_calculated_egfr | 0.2905 | 0.2355 | 0.2462 | -0.0107 |
| tvae_qt_seed1 | nyha_nyha_pET | 0.7266 | 0.7338 | 0.7036 | +0.0302 |
| tvae_qt_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.5273 | 0.5077 | +0.0196 |
| tvae_qt_seed2 | ckd_severity_from_calculated_egfr | 0.2905 | 0.256 | 0.2402 | +0.0158 |
| tvae_qt_seed2 | nyha_nyha_pET | 0.7266 | 0.7415 | 0.7129 | +0.0285 |
| tvae_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.5233 | 0.5324 | -0.0091 |
| tvae_seed0 | ckd_severity_from_calculated_egfr | 0.2905 | 0.2645 | 0.2504 | +0.0141 |
| tvae_seed0 | nyha_nyha_pET | 0.7266 | 0.7398 | 0.7087 | +0.0311 |
| tvae_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.5301 | 0.54 | -0.0099 |
| tvae_seed1 | ckd_severity_from_calculated_egfr | 0.2905 | 0.254 | 0.2419 | +0.0121 |
| tvae_seed1 | nyha_nyha_pET | 0.7266 | 0.7449 | 0.7147 | +0.0302 |
| tvae_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.5681 | 0.5355 | 0.5111 | +0.0244 |
| tvae_seed2 | ckd_severity_from_calculated_egfr | 0.2905 | 0.2378 | 0.2428 | -0.0050 |
| tvae_seed2 | nyha_nyha_pET | 0.7266 | 0.7526 | 0.724 | +0.0285 |

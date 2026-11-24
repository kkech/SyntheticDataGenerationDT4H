# Adversarial Privacy Attacks

Members: 1549 training records; non-members: 516 holdout records (real, unseen patients). Membership inference AUC of 0.5 means the synthetic data reveals nothing about who was in the training set. Attribute inference reports the MEMBERSHIP ADVANTAGE -- accuracy on members minus accuracy on non-members; population-level inference (both above baseline, equally) is the intended use of released data, only member-specific advantage is leakage.

| run | MIA AUC (95% CI) | learned MIA AUC (95% CI) | empirical ε̂ lower bound | worst AIA membership advantage | anonymeter |
|---|---|---|---|---|---|
| aim40_eps1_seed0 | 0.4841 (0.4535-0.5134) | 0.5089 (0.4813-0.5384) | 0.0 | -0.0197 | SO 0.0, link 0.0 |
| aim40_eps5_seed0 | 0.4832 (0.4566-0.5112) | 0.5081 (0.4796-0.5386) | 0.0 | 0.0089 | SO 0.0, link 0.0 |
| aim40_eps8_seed0 | 0.4807 (0.4552-0.5073) | 0.5012 (0.4686-0.5314) | 0.0 | 0.0249 | SO 0.0, link 0.0 |
| aim50_eps1_seed0 | 0.4856 (0.4569-0.5139) | 0.4779 (0.4503-0.5045) | 0.0 | 0.0502 | SO 0.0, link 0.0 |
| aim50_eps5_seed0 | 0.4793 (0.4539-0.508) | 0.5088 (0.4831-0.537) | 0.0 | 0.0475 | SO 0.0, link 0.0 |
| aim50_eps8_seed0 | 0.4831 (0.4535-0.5108) | 0.5056 (0.4772-0.5331) | 0.0 | -0.001 | SO 0.0, link 0.0 |
| ctgan_qt_seed0 | 0.5018 (0.4723-0.5316) | 0.4642 (0.4381-0.4934) | 0.0 | 0.0373 | SO 0.005, link 0.0 |
| ctgan_seed0 | 0.4853 (0.4585-0.5151) | 0.5 (0.474-0.5258) | 0.0 | 0.0339 | SO 0.0033, link 0.0 |
| ctgan_seed1 | 0.4959 (0.4682-0.5248) | 0.449 (0.4208-0.4745) | 0.0 | 0.0134 | SO 0.0099, link 0.0099 |
| ctgan_seed2 | 0.4984 (0.4733-0.5287) | 0.4934 (0.4643-0.5218) | 0.0 | 0.012 | SO 0.005, link 0.0 |
| ddpm_g_seed0 | 0.5175 (0.4885-0.5455) | 0.4918 (0.4649-0.5211) | 0.1672 | 0.0496 | SO 0.0, link 0.0 |
| ddpm_seed0 | 0.501 (0.4704-0.5284) | 0.4951 (0.4676-0.5257) | 0.0 | 0.0108 | SO 0.0, link 0.0 |
| ddpm_seed1 🚨 (worst: nearest-synthetic-distance) | 0.528 (0.5011-0.5568) | 0.5181 (0.4889-0.5485) | 0.0687 | 0.0261 | SO 0.0, link 0.0 |
| ddpm_seed2 | 0.5109 (0.482-0.5376) | 0.4935 (0.4611-0.5219) | 0.0 | 0.008 | SO 0.0, link 0.0 |
| dpctgan_eps10_seed0 | 0.4869 (0.4583-0.5137) | 0.4886 (0.4626-0.517) | 0.0 | 0.007 | SO 0.0, link 0.0 |
| dpctgan_eps15_seed0 | 0.4912 (0.4644-0.5189) | 0.5075 (0.48-0.5366) | 0.0 | 0.0147 | SO 0.005, link 0.0 |
| dpctgan_eps15_seed1 | 0.4919 (0.4657-0.5204) | 0.4969 (0.4738-0.5273) | 0.0 | 0.0339 | SO 0.0, link 0.0 |
| dpctgan_eps15_seed2 | 0.4882 (0.4604-0.5151) | 0.5005 (0.4725-0.5282) | 0.0 | 0.0 | SO 0.0, link 0.0 |
| dpctgan_eps1_seed0 | 0.4812 (0.4512-0.511) | 0.4801 (0.451-0.5091) | 0.0 | 0.0075 | SO 0.0, link 0.0 |
| dpctgan_eps20_seed0 | 0.4834 (0.4552-0.5119) | 0.5112 (0.4836-0.5382) | 0.0 | 0.0225 | SO 0.0, link 0.0 |
| dpctgan_eps5_seed0 | 0.4794 (0.4499-0.5067) | 0.4826 (0.4542-0.508) | 0.0 | 0.0339 | SO 0.0, link 0.0 |
| dpctgan_eps8_seed0 | 0.4919 (0.4622-0.5187) | 0.5054 (0.4754-0.5363) | 0.0 | 0.0339 | SO 0.0, link 0.0 |
| gaussian_copula_seed0 | 0.5139 (0.485-0.5435) | 0.4965 (0.4685-0.5274) | 0.084 | 0.0132 | SO 0.0083, link 0.0 |
| gaussian_copula_seed1 | 0.5126 (0.4852-0.5397) | 0.5073 (0.4794-0.5369) | 0.0 | 0.0436 | SO 0.0132, link 0.0 |
| gaussian_copula_seed2 🚨 (worst: cv-logreg over (d1, d2, d1/d2)) | 0.5097 (0.484-0.536) | 0.5362 (0.5063-0.5633) | 0.072 | 0.027 | SO 0.0083, link 0.0 |
| mst_eps10_seed0 | 0.5148 (0.4842-0.5412) | 0.5 (0.4722-0.5291) | 0.0 | 0.0191 | SO 0.005, link 0.0 |
| mst_eps15_seed0 | 0.5104 (0.4848-0.5397) | 0.4616 (0.4349-0.4889) | 0.0 | 0.0134 | SO 0.0, link 0.0 |
| mst_eps15_seed1 | 0.5155 (0.4867-0.5434) | 0.5061 (0.4753-0.5337) | 0.0 | 0.0179 | SO 0.0066, link 0.0 |
| mst_eps15_seed2 | 0.5072 (0.4784-0.5386) | 0.4754 (0.4458-0.505) | 0.0 | 0.0373 | SO 0.0199, link 0.0 |
| mst_eps1_seed0 | 0.4866 (0.4578-0.5144) | 0.4672 (0.4374-0.4965) | 0.0 | 0.0036 | SO 0.0, link 0.0 |
| mst_eps20_seed0 | 0.5063 (0.475-0.5385) | 0.4709 (0.4367-0.4983) | 0.0 | 0.021 | SO 0.0, link 0.0 |
| mst_eps5_seed0 | 0.5042 (0.4773-0.5318) | 0.4687 (0.438-0.497) | 0.0 | 0.0158 | SO 0.0, link 0.0 |
| mst_eps8_seed0 | 0.5152 (0.4852-0.5405) | 0.5206 (0.4956-0.5489) | 0.0169 | 0.0184 | SO 0.005, link 0.0 |
| patectgan_eps15_seed0 | 0.5094 (0.4798-0.5377) | 0.4723 (0.4438-0.4975) | 0.0108 | 0.0404 | SO 0.0, link 0.0 |
| patectgan_eps1_seed0 | 0.4918 (0.4624-0.5186) | 0.4823 (0.4547-0.5138) | 0.0 | 0.0509 | SO 0.005, link 0.0 |
| patectgan_eps5_seed0 | 0.4957 (0.4665-0.5216) | 0.494 (0.4646-0.5242) | 0.0 | 0.0036 | SO 0.0, link 0.0 |
| tvae_cap256_seed0 🚨 (worst: cv-logreg over (d1, d2, d1/d2)) | 0.5476 (0.5196-0.5776) | 0.5612 (0.5311-0.5905) | 0.208 | 0.0203 | SO 0.0, link 0.0 |
| tvae_ep1000_seed0 🚨 (worst: nearest-synthetic-distance) | 0.5412 (0.5131-0.5697) | 0.5385 (0.5077-0.5662) | 0.0749 | 0.0281 | SO 0.0066, link 0.0 |
| tvae_ind_seed0 🚨 (worst: nearest-synthetic-distance) | 0.5318 (0.5036-0.5598) | 0.5165 (0.4883-0.5459) | 0.0363 | 0.0217 | SO 0.0017, link 0.0 |
| tvae_qt_seed0 🚨 (worst: nearest-synthetic-distance) | 0.5353 (0.5088-0.5637) | 0.5341 (0.5037-0.5596) | 0.0371 | 0.0196 | SO 0.0, link 0.0 |
| tvae_qt_seed1 | 0.5208 (0.4922-0.5497) | 0.5075 (0.4797-0.5348) | 0.148 | 0.0217 | SO 0.0, link 0.0 |
| tvae_qt_seed2 🚨 (worst: nearest-synthetic-distance) | 0.5301 (0.5008-0.5588) | 0.5259 (0.499-0.5546) | 0.2008 | 0.0281 | SO 0.0, link 0.0 |
| tvae_seed0 🚨 (worst: nearest-synthetic-distance) | 0.5273 (0.498-0.5562) | 0.5268 (0.4943-0.5551) | 0.145 | 0.0289 | SO 0.0066, link 0.0 |
| tvae_seed1 🚨 (worst: cv-logreg over (d1, d2, d1/d2)) | 0.5266 (0.4977-0.5542) | 0.5273 (0.499-0.5554) | 0.0399 | 0.0242 | SO 0.0, link 0.0 |
| tvae_seed2 🚨 (worst: nearest-synthetic-distance) | 0.5276 (0.4978-0.5543) | 0.5239 (0.4962-0.5505) | 0.145 | 0.0203 | SO 0.005, link 0.0 |

## Who is at risk: membership inference by patient atypicality

WITHIN-STRATUM AUC: members and non-members both get the same atypicality score (distance to their 5th-nearest member, same reference set and encoder), non-members are binned by the member quartile cut points, and each stratum's AUC compares members-in-Qi against non-members-in-Qi only. 0.5 = no leakage on that stratum; elevated values indicate SELECTIVE leakage on that stratum (e.g. a model that memorizes its unusual patients shows it in Q4). Strata with fewer than 30 non-members are skipped ('-'). Cell format: AUC (n members / n non-members).

| run | Q1 typical | Q2 | Q3 | Q4 atypical |
|---|---|---|---|---|
| aim40_eps1_seed0 | 0.5151 (388/136) | 0.4709 (387/115) | 0.4563 (387/131) | 0.4776 (387/134) |
| aim40_eps5_seed0 | 0.5117 (388/136) | 0.4658 (387/115) | 0.4584 (387/131) | 0.4794 (387/134) |
| aim40_eps8_seed0 | 0.5112 (388/136) | 0.4575 (387/115) | 0.4566 (387/131) | 0.4778 (387/134) |
| aim50_eps1_seed0 | 0.5085 (388/136) | 0.4737 (387/115) | 0.4592 (387/131) | 0.4839 (387/134) |
| aim50_eps5_seed0 | 0.5065 (388/136) | 0.4563 (387/115) | 0.4618 (387/131) | 0.4739 (387/134) |
| aim50_eps8_seed0 | 0.5093 (388/136) | 0.465 (387/115) | 0.4656 (387/131) | 0.4748 (387/134) |
| ctgan_qt_seed0 | 0.5579 (388/136) | 0.4622 (387/115) | 0.4917 (387/131) | 0.4791 (387/134) |
| ctgan_seed0 | 0.5047 (388/136) | 0.429 (387/115) | 0.5071 (387/131) | 0.4635 (387/134) |
| ctgan_seed1 | 0.5246 (388/136) | 0.4435 (387/115) | 0.4986 (387/131) | 0.4921 (387/134) |
| ctgan_seed2 | 0.5327 (388/136) | 0.4642 (387/115) | 0.4881 (387/131) | 0.4933 (387/134) |
| ddpm_g_seed0 | 0.5275 (388/136) | 0.5126 (387/115) | 0.5171 (387/131) | 0.5131 (387/134) |
| ddpm_seed0 | 0.4815 (388/136) | 0.5065 (387/115) | 0.5174 (387/131) | 0.5052 (387/134) |
| ddpm_seed1 | 0.5385 (388/136) | 0.4765 (387/115) | 0.5457 (387/131) | 0.5708 (387/134) |
| ddpm_seed2 | 0.4957 (388/136) | 0.4679 (387/115) | 0.5805 (387/131) | 0.5067 (387/134) |
| dpctgan_eps10_seed0 | 0.5334 (388/136) | 0.4408 (387/115) | 0.4961 (387/131) | 0.4604 (387/134) |
| dpctgan_eps15_seed0 | 0.5508 (388/136) | 0.4388 (387/115) | 0.5013 (387/131) | 0.4574 (387/134) |
| dpctgan_eps15_seed1 | 0.5331 (388/136) | 0.4541 (387/115) | 0.4905 (387/131) | 0.4676 (387/134) |
| dpctgan_eps15_seed2 | 0.5317 (388/136) | 0.4628 (387/115) | 0.487 (387/131) | 0.4531 (387/134) |
| dpctgan_eps1_seed0 | 0.4933 (388/136) | 0.4453 (387/115) | 0.4924 (387/131) | 0.4621 (387/134) |
| dpctgan_eps20_seed0 | 0.5228 (388/136) | 0.466 (387/115) | 0.4714 (387/131) | 0.4621 (387/134) |
| dpctgan_eps5_seed0 | 0.5031 (388/136) | 0.4271 (387/115) | 0.4902 (387/131) | 0.4753 (387/134) |
| dpctgan_eps8_seed0 | 0.5239 (388/136) | 0.4941 (387/115) | 0.4792 (387/131) | 0.4689 (387/134) |
| gaussian_copula_seed0 | 0.5831 (388/136) | 0.4877 (387/115) | 0.5016 (387/131) | 0.4964 (387/134) |
| gaussian_copula_seed1 | 0.529 (388/136) | 0.5014 (387/115) | 0.5188 (387/131) | 0.4991 (387/134) |
| gaussian_copula_seed2 | 0.5701 (388/136) | 0.5012 (387/115) | 0.4847 (387/131) | 0.5167 (387/134) |
| mst_eps10_seed0 | 0.5419 (388/136) | 0.4685 (387/115) | 0.5129 (387/131) | 0.5447 (387/134) |
| mst_eps15_seed0 | 0.5332 (388/136) | 0.4758 (387/115) | 0.5141 (387/131) | 0.513 (387/134) |
| mst_eps15_seed1 | 0.5525 (388/136) | 0.5084 (387/115) | 0.5001 (387/131) | 0.4943 (387/134) |
| mst_eps15_seed2 | 0.5157 (388/136) | 0.4518 (387/115) | 0.5081 (387/131) | 0.5436 (387/134) |
| mst_eps1_seed0 | 0.5279 (388/136) | 0.4276 (387/115) | 0.4944 (387/131) | 0.4812 (387/134) |
| mst_eps20_seed0 | 0.5166 (388/136) | 0.457 (387/115) | 0.5232 (387/131) | 0.5013 (387/134) |
| mst_eps5_seed0 | 0.5458 (388/136) | 0.4263 (387/115) | 0.5017 (387/131) | 0.5302 (387/134) |
| mst_eps8_seed0 | 0.5503 (388/136) | 0.4954 (387/115) | 0.5002 (387/131) | 0.5353 (387/134) |
| patectgan_eps15_seed0 | 0.5524 (388/136) | 0.524 (387/115) | 0.4947 (387/131) | 0.4753 (387/134) |
| patectgan_eps1_seed0 | 0.5374 (388/136) | 0.4579 (387/115) | 0.5056 (387/131) | 0.4777 (387/134) |
| patectgan_eps5_seed0 | 0.5257 (388/136) | 0.4971 (387/115) | 0.503 (387/131) | 0.4548 (387/134) |
| tvae_cap256_seed0 🚨 | 0.5935 (388/136) | 0.5697 (387/115) | 0.552 (387/131) | 0.6158 (387/134) |
| tvae_ep1000_seed0 | 0.5856 (388/136) | 0.5609 (387/115) | 0.5782 (387/131) | 0.5658 (387/134) |
| tvae_ind_seed0 | 0.5416 (388/136) | 0.5376 (387/115) | 0.5594 (387/131) | 0.5664 (387/134) |
| tvae_qt_seed0 | 0.5766 (388/136) | 0.5513 (387/115) | 0.5432 (387/131) | 0.5732 (387/134) |
| tvae_qt_seed1 | 0.5282 (388/136) | 0.522 (387/115) | 0.5248 (387/131) | 0.5784 (387/134) |
| tvae_qt_seed2 | 0.5486 (388/136) | 0.5511 (387/115) | 0.5374 (387/131) | 0.5702 (387/134) |
| tvae_seed0 | 0.543 (388/136) | 0.5552 (387/115) | 0.5282 (387/131) | 0.5727 (387/134) |
| tvae_seed1 | 0.5425 (388/136) | 0.5042 (387/115) | 0.5672 (387/131) | 0.5657 (387/134) |
| tvae_seed2 | 0.5659 (388/136) | 0.5092 (387/115) | 0.549 (387/131) | 0.5686 (387/134) |

## Attribute inference detail

| run | sensitive attribute | baseline | member acc | non-member acc | advantage |
|---|---|---|---|---|---|
| aim40_eps1_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5184 | 0.5523 | -0.0339 |
| aim40_eps1_seed0 | ckd_severity_from_calculated_egfr | 0.3585 | 0.2421 | 0.2694 | -0.0273 |
| aim40_eps1_seed0 | nyha_nyha_pET | 0.5407 | 0.4203 | 0.4399 | -0.0197 |
| aim40_eps5_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5184 | 0.5523 | -0.0339 |
| aim40_eps5_seed0 | ckd_severity_from_calculated_egfr | 0.3585 | 0.2976 | 0.2888 | +0.0089 |
| aim40_eps5_seed0 | nyha_nyha_pET | 0.5407 | 0.4861 | 0.4903 | -0.0042 |
| aim40_eps8_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5184 | 0.5523 | -0.0339 |
| aim40_eps8_seed0 | ckd_severity_from_calculated_egfr | 0.3585 | 0.2214 | 0.2209 | +0.0005 |
| aim40_eps8_seed0 | nyha_nyha_pET | 0.5407 | 0.4745 | 0.4496 | +0.0249 |
| aim50_eps1_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5184 | 0.5523 | -0.0339 |
| aim50_eps1_seed0 | ckd_severity_from_calculated_egfr | 0.3585 | 0.2828 | 0.2326 | +0.0502 |
| aim50_eps1_seed0 | nyha_nyha_pET | 0.5407 | 0.4622 | 0.4302 | +0.0320 |
| aim50_eps5_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5184 | 0.5523 | -0.0339 |
| aim50_eps5_seed0 | ckd_severity_from_calculated_egfr | 0.3585 | 0.2408 | 0.2597 | -0.0189 |
| aim50_eps5_seed0 | nyha_nyha_pET | 0.5407 | 0.499 | 0.4516 | +0.0475 |
| aim50_eps8_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5184 | 0.5523 | -0.0339 |
| aim50_eps8_seed0 | ckd_severity_from_calculated_egfr | 0.3585 | 0.2744 | 0.281 | -0.0066 |
| aim50_eps8_seed0 | nyha_nyha_pET | 0.5407 | 0.4835 | 0.4845 | -0.0010 |
| ctgan_qt_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5223 | 0.5058 | +0.0165 |
| ctgan_qt_seed0 | ckd_severity_from_calculated_egfr | 0.3585 | 0.2544 | 0.2171 | +0.0373 |
| ctgan_qt_seed0 | nyha_nyha_pET | 0.5407 | 0.4822 | 0.5271 | -0.0449 |
| ctgan_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5416 | 0.5078 | +0.0339 |
| ctgan_seed0 | ckd_severity_from_calculated_egfr | 0.3585 | 0.2647 | 0.25 | +0.0147 |
| ctgan_seed0 | nyha_nyha_pET | 0.5407 | 0.4467 | 0.436 | +0.0107 |
| ctgan_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5223 | 0.5329 | -0.0107 |
| ctgan_seed1 | ckd_severity_from_calculated_egfr | 0.3585 | 0.2886 | 0.2752 | +0.0134 |
| ctgan_seed1 | nyha_nyha_pET | 0.5407 | 0.4364 | 0.4651 | -0.0287 |
| ctgan_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.4784 | 0.4709 | +0.0074 |
| ctgan_seed2 | ckd_severity_from_calculated_egfr | 0.3585 | 0.2615 | 0.2539 | +0.0076 |
| ctgan_seed2 | nyha_nyha_pET | 0.5407 | 0.45 | 0.438 | +0.0120 |
| ddpm_g_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5416 | 0.5136 | +0.0281 |
| ddpm_g_seed0 | ckd_severity_from_calculated_egfr | 0.3585 | 0.2473 | 0.1977 | +0.0496 |
| ddpm_g_seed0 | nyha_nyha_pET | 0.5407 | 0.2653 | 0.2558 | +0.0095 |
| ddpm_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5203 | 0.5136 | +0.0068 |
| ddpm_seed0 | ckd_severity_from_calculated_egfr | 0.3585 | 0.2414 | 0.2306 | +0.0108 |
| ddpm_seed0 | nyha_nyha_pET | 0.5407 | 0.2841 | 0.2907 | -0.0066 |
| ddpm_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.532 | 0.5058 | +0.0261 |
| ddpm_seed1 | ckd_severity_from_calculated_egfr | 0.3585 | 0.2511 | 0.2597 | -0.0086 |
| ddpm_seed1 | nyha_nyha_pET | 0.5407 | 0.2886 | 0.3178 | -0.0293 |
| ddpm_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5507 | 0.5426 | +0.0080 |
| ddpm_seed2 | ckd_severity_from_calculated_egfr | 0.3585 | 0.2731 | 0.2713 | +0.0018 |
| ddpm_seed2 | nyha_nyha_pET | 0.5407 | 0.2795 | 0.2791 | +0.0005 |
| dpctgan_eps10_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5184 | 0.5523 | -0.0339 |
| dpctgan_eps10_seed0 | ckd_severity_from_calculated_egfr | 0.3585 | 0.2376 | 0.2306 | +0.0070 |
| dpctgan_eps10_seed0 | nyha_nyha_pET | 0.5407 | 0.0349 | 0.0349 | -0.0000 |
| dpctgan_eps15_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5184 | 0.5523 | -0.0339 |
| dpctgan_eps15_seed0 | ckd_severity_from_calculated_egfr | 0.3585 | 0.2376 | 0.2306 | +0.0070 |
| dpctgan_eps15_seed0 | nyha_nyha_pET | 0.5407 | 0.1717 | 0.157 | +0.0147 |
| dpctgan_eps15_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.4816 | 0.4477 | +0.0339 |
| dpctgan_eps15_seed1 | ckd_severity_from_calculated_egfr | 0.3585 | 0.2376 | 0.2306 | +0.0070 |
| dpctgan_eps15_seed1 | nyha_nyha_pET | 0.5407 | 0.4164 | 0.4128 | +0.0036 |
| dpctgan_eps15_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5184 | 0.5523 | -0.0339 |
| dpctgan_eps15_seed2 | ckd_severity_from_calculated_egfr | 0.3585 | 0.1336 | 0.1531 | -0.0195 |
| dpctgan_eps15_seed2 | nyha_nyha_pET | 0.5407 | 0.0 | 0.0 | +0.0000 |
| dpctgan_eps1_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5197 | 0.5349 | -0.0152 |
| dpctgan_eps1_seed0 | ckd_severity_from_calculated_egfr | 0.3585 | 0.366 | 0.3585 | +0.0075 |
| dpctgan_eps1_seed0 | nyha_nyha_pET | 0.5407 | 0.0549 | 0.0581 | -0.0033 |
| dpctgan_eps20_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5184 | 0.5523 | -0.0339 |
| dpctgan_eps20_seed0 | ckd_severity_from_calculated_egfr | 0.3585 | 0.2105 | 0.188 | +0.0225 |
| dpctgan_eps20_seed0 | nyha_nyha_pET | 0.5407 | 0.0 | 0.0 | +0.0000 |
| dpctgan_eps5_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.4816 | 0.4477 | +0.0339 |
| dpctgan_eps5_seed0 | ckd_severity_from_calculated_egfr | 0.3585 | 0.366 | 0.3585 | +0.0075 |
| dpctgan_eps5_seed0 | nyha_nyha_pET | 0.5407 | 0.4106 | 0.4244 | -0.0138 |
| dpctgan_eps8_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.4816 | 0.4477 | +0.0339 |
| dpctgan_eps8_seed0 | ckd_severity_from_calculated_egfr | 0.3585 | 0.2376 | 0.2306 | +0.0070 |
| dpctgan_eps8_seed0 | nyha_nyha_pET | 0.5407 | 0.0 | 0.0 | +0.0000 |
| gaussian_copula_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5384 | 0.5252 | +0.0132 |
| gaussian_copula_seed0 | ckd_severity_from_calculated_egfr | 0.3585 | 0.2653 | 0.2558 | +0.0095 |
| gaussian_copula_seed0 | nyha_nyha_pET | 0.5407 | 0.4906 | 0.4903 | +0.0003 |
| gaussian_copula_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5449 | 0.5601 | -0.0152 |
| gaussian_copula_seed1 | ckd_severity_from_calculated_egfr | 0.3585 | 0.2776 | 0.2461 | +0.0315 |
| gaussian_copula_seed1 | nyha_nyha_pET | 0.5407 | 0.4913 | 0.4477 | +0.0436 |
| gaussian_copula_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5429 | 0.5291 | +0.0139 |
| gaussian_copula_seed2 | ckd_severity_from_calculated_egfr | 0.3585 | 0.2576 | 0.2306 | +0.0270 |
| gaussian_copula_seed2 | nyha_nyha_pET | 0.5407 | 0.4745 | 0.5097 | -0.0352 |
| mst_eps10_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5771 | 0.593 | -0.0159 |
| mst_eps10_seed0 | ckd_severity_from_calculated_egfr | 0.3585 | 0.2511 | 0.2326 | +0.0186 |
| mst_eps10_seed0 | nyha_nyha_pET | 0.5407 | 0.4203 | 0.4012 | +0.0191 |
| mst_eps15_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.6172 | 0.6473 | -0.0301 |
| mst_eps15_seed0 | ckd_severity_from_calculated_egfr | 0.3585 | 0.2402 | 0.2267 | +0.0134 |
| mst_eps15_seed0 | nyha_nyha_pET | 0.5407 | 0.3725 | 0.3915 | -0.0190 |
| mst_eps15_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.6262 | 0.6667 | -0.0405 |
| mst_eps15_seed1 | ckd_severity_from_calculated_egfr | 0.3585 | 0.2795 | 0.2616 | +0.0179 |
| mst_eps15_seed1 | nyha_nyha_pET | 0.5407 | 0.306 | 0.3372 | -0.0312 |
| mst_eps15_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.6527 | 0.6822 | -0.0295 |
| mst_eps15_seed2 | ckd_severity_from_calculated_egfr | 0.3585 | 0.2718 | 0.2345 | +0.0373 |
| mst_eps15_seed2 | nyha_nyha_pET | 0.5407 | 0.4564 | 0.4477 | +0.0087 |
| mst_eps1_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5016 | 0.4981 | +0.0036 |
| mst_eps1_seed0 | ckd_severity_from_calculated_egfr | 0.3585 | 0.235 | 0.2345 | +0.0005 |
| mst_eps1_seed0 | nyha_nyha_pET | 0.5407 | 0.4622 | 0.4729 | -0.0106 |
| mst_eps20_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.6411 | 0.6822 | -0.0411 |
| mst_eps20_seed0 | ckd_severity_from_calculated_egfr | 0.3585 | 0.2621 | 0.25 | +0.0121 |
| mst_eps20_seed0 | nyha_nyha_pET | 0.5407 | 0.4668 | 0.4457 | +0.0210 |
| mst_eps5_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5513 | 0.5988 | -0.0475 |
| mst_eps5_seed0 | ckd_severity_from_calculated_egfr | 0.3585 | 0.244 | 0.2461 | -0.0021 |
| mst_eps5_seed0 | nyha_nyha_pET | 0.5407 | 0.5178 | 0.5019 | +0.0158 |
| mst_eps8_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5507 | 0.5795 | -0.0288 |
| mst_eps8_seed0 | ckd_severity_from_calculated_egfr | 0.3585 | 0.2576 | 0.2733 | -0.0157 |
| mst_eps8_seed0 | nyha_nyha_pET | 0.5407 | 0.4835 | 0.4651 | +0.0184 |
| patectgan_eps15_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5972 | 0.562 | +0.0351 |
| patectgan_eps15_seed0 | ckd_severity_from_calculated_egfr | 0.3585 | 0.3267 | 0.3217 | +0.0050 |
| patectgan_eps15_seed0 | nyha_nyha_pET | 0.5407 | 0.4241 | 0.3837 | +0.0404 |
| patectgan_eps1_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5029 | 0.4922 | +0.0107 |
| patectgan_eps1_seed0 | ckd_severity_from_calculated_egfr | 0.3585 | 0.2524 | 0.2016 | +0.0509 |
| patectgan_eps1_seed0 | nyha_nyha_pET | 0.5407 | 0.419 | 0.4186 | +0.0004 |
| patectgan_eps5_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5165 | 0.562 | -0.0456 |
| patectgan_eps5_seed0 | ckd_severity_from_calculated_egfr | 0.3585 | 0.275 | 0.2868 | -0.0118 |
| patectgan_eps5_seed0 | nyha_nyha_pET | 0.5407 | 0.3641 | 0.3605 | +0.0036 |
| tvae_cap256_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.6075 | 0.5872 | +0.0203 |
| tvae_cap256_seed0 | ckd_severity_from_calculated_egfr | 0.3585 | 0.2957 | 0.3314 | -0.0357 |
| tvae_cap256_seed0 | nyha_nyha_pET | 0.5407 | 0.5268 | 0.5581 | -0.0313 |
| tvae_ep1000_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5855 | 0.593 | -0.0075 |
| tvae_ep1000_seed0 | ckd_severity_from_calculated_egfr | 0.3585 | 0.3363 | 0.3391 | -0.0028 |
| tvae_ep1000_seed0 | nyha_nyha_pET | 0.5407 | 0.5397 | 0.5116 | +0.0281 |
| tvae_ind_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5881 | 0.5756 | +0.0125 |
| tvae_ind_seed0 | ckd_severity_from_calculated_egfr | 0.3585 | 0.3357 | 0.314 | +0.0217 |
| tvae_ind_seed0 | nyha_nyha_pET | 0.5407 | 0.541 | 0.5252 | +0.0158 |
| tvae_qt_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5855 | 0.5659 | +0.0196 |
| tvae_qt_seed0 | ckd_severity_from_calculated_egfr | 0.3585 | 0.2918 | 0.3256 | -0.0338 |
| tvae_qt_seed0 | nyha_nyha_pET | 0.5407 | 0.541 | 0.5446 | -0.0036 |
| tvae_qt_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5985 | 0.5795 | +0.0190 |
| tvae_qt_seed1 | ckd_severity_from_calculated_egfr | 0.3585 | 0.3338 | 0.312 | +0.0217 |
| tvae_qt_seed1 | nyha_nyha_pET | 0.5407 | 0.5152 | 0.5174 | -0.0023 |
| tvae_qt_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.6191 | 0.6027 | +0.0164 |
| tvae_qt_seed2 | ckd_severity_from_calculated_egfr | 0.3585 | 0.326 | 0.3198 | +0.0062 |
| tvae_qt_seed2 | nyha_nyha_pET | 0.5407 | 0.5145 | 0.4864 | +0.0281 |
| tvae_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5836 | 0.5601 | +0.0235 |
| tvae_seed0 | ckd_severity_from_calculated_egfr | 0.3585 | 0.3351 | 0.3062 | +0.0289 |
| tvae_seed0 | nyha_nyha_pET | 0.5407 | 0.5268 | 0.5174 | +0.0093 |
| tvae_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5707 | 0.5465 | +0.0242 |
| tvae_seed1 | ckd_severity_from_calculated_egfr | 0.3585 | 0.3234 | 0.3488 | -0.0254 |
| tvae_seed1 | nyha_nyha_pET | 0.5407 | 0.5307 | 0.5271 | +0.0035 |
| tvae_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.5523 | 0.5959 | 0.5756 | +0.0203 |
| tvae_seed2 | ckd_severity_from_calculated_egfr | 0.3585 | 0.3196 | 0.3275 | -0.0080 |
| tvae_seed2 | nyha_nyha_pET | 0.5407 | 0.5365 | 0.5194 | +0.0171 |

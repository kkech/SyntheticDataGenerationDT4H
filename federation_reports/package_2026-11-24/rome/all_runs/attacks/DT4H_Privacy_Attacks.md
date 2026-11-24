# Adversarial Privacy Attacks

Members: 3773 training records; non-members: 1258 holdout records (real, unseen patients). Membership inference AUC of 0.5 means the synthetic data reveals nothing about who was in the training set. Attribute inference reports the MEMBERSHIP ADVANTAGE -- accuracy on members minus accuracy on non-members; population-level inference (both above baseline, equally) is the intended use of released data, only member-specific advantage is leakage.

| run | MIA AUC (95% CI) | learned MIA AUC (95% CI) | empirical ε̂ lower bound | worst AIA membership advantage | anonymeter |
|---|---|---|---|---|---|
| aim40_eps1_seed0 | 0.4786 (0.46-0.4964) | 0.5172 (0.5001-0.5355) | 0.0 | 0.0091 | SO 0.005, link 0.0 |
| aim40_eps5_seed0 | 0.4779 (0.4576-0.4953) | 0.5147 (0.4962-0.5314) | 0.0 | 0.0141 | SO 0.0, link 0.0 |
| aim50_eps1_seed0 | 0.4807 (0.4617-0.4992) | 0.5138 (0.4943-0.5334) | 0.0148 | 0.0184 | SO 0.0, link 0.0 |
| ctgan_qt_seed0 | 0.4817 (0.4629-0.5004) | 0.5074 (0.4873-0.5278) | 0.0038 | 0.0118 | SO 0.0, link 0.0 |
| ctgan_seed0 | 0.4841 (0.4654-0.5007) | 0.5102 (0.4899-0.5275) | 0.0 | 0.0038 | SO 0.0066, link 0.0 |
| ctgan_seed1 | 0.4874 (0.4675-0.5055) | 0.5115 (0.491-0.5297) | 0.0227 | 0.0186 | SO 0.0, link 0.0 |
| ctgan_seed2 | 0.4891 (0.4714-0.507) | 0.4949 (0.4767-0.5132) | 0.0 | 0.0221 | SO 0.005, link 0.0 |
| ddpm_g_seed0 | 0.4753 (0.4596-0.4914) | 0.5135 (0.4966-0.5321) | 0.0091 | -0.0023 | SO 0.0, link 0.0 |
| ddpm_seed0 | 0.4772 (0.4597-0.4964) | 0.512 (0.4939-0.5318) | 0.0024 | -0.0016 | SO 0.0, link 0.0 |
| ddpm_seed1 | 0.4843 (0.4648-0.5028) | 0.5169 (0.4977-0.5342) | 0.0 | -0.0036 | SO 0.0, link 0.0 |
| ddpm_seed2 | 0.4821 (0.4629-0.5005) | 0.515 (0.4959-0.5344) | 0.006 | 0.0022 | SO 0.0, link 0.0 |
| dpctgan_eps10_seed0 | 0.4872 (0.4684-0.5044) | 0.5053 (0.4879-0.524) | 0.0 | 0.0036 | SO 0.0, link 0.0 |
| dpctgan_eps15_seed0 | 0.4838 (0.4645-0.5033) | 0.5111 (0.4935-0.529) | 0.0096 | 0.0135 | SO 0.0, link 0.0 |
| dpctgan_eps15_seed1 | 0.4791 (0.4603-0.4981) | 0.509 (0.4903-0.5271) | 0.0 | -0.0027 | SO 0.0, link 0.0 |
| dpctgan_eps15_seed2 | 0.4747 (0.4575-0.495) | 0.5195 (0.5006-0.5368) | 0.0125 | 0.0133 | SO 0.0, link 0.0 |
| dpctgan_eps1_seed0 | 0.4847 (0.4682-0.5012) | 0.5099 (0.4917-0.5284) | 0.0 | 0.0173 | SO 0.0, link 0.0 |
| dpctgan_eps20_seed0 | 0.4796 (0.4618-0.4968) | 0.5053 (0.4882-0.5233) | 0.0 | 0.013 | SO 0.0, link 0.0 |
| dpctgan_eps5_seed0 | 0.4794 (0.4619-0.4986) | 0.5103 (0.4925-0.5273) | 0.0 | 0.0218 | SO 0.0, link 0.0 |
| dpctgan_eps8_seed0 | 0.4885 (0.4712-0.5062) | 0.506 (0.4895-0.5258) | 0.0 | 0.0123 | SO 0.0, link 0.0 |
| gaussian_copula_seed0 | 0.4834 (0.4642-0.5014) | 0.5079 (0.4893-0.5263) | 0.0 | 0.007 | SO 0.005, link 0.0 |
| gaussian_copula_seed1 | 0.479 (0.462-0.4974) | 0.5122 (0.494-0.5291) | 0.0 | 0.03 | SO 0.0, link 0.0 |
| gaussian_copula_seed2 | 0.4824 (0.4644-0.5) | 0.5076 (0.4901-0.526) | 0.0 | 0.0046 | SO 0.0, link 0.0 |
| mst_eps0p5_seed0 | 0.478 (0.4609-0.4957) | 0.5161 (0.4995-0.5344) | 0.0 | -0.0021 | SO 0.0, link 0.0 |
| mst_eps10_seed0 | 0.4865 (0.4684-0.5059) | 0.5079 (0.4902-0.5254) | 0.0 | 0.0104 | SO 0.0, link 0.0 |
| mst_eps15_seed0 | 0.4866 (0.4672-0.5056) | 0.5129 (0.4946-0.5291) | 0.0543 | 0.0027 | SO 0.0, link 0.0 |
| mst_eps15_seed1 | 0.4893 (0.4713-0.5057) | 0.5069 (0.4894-0.5261) | 0.0 | 0.0059 | SO 0.0233, link 0.0 |
| mst_eps15_seed2 | 0.4857 (0.4659-0.5045) | 0.498 (0.4796-0.5148) | 0.0 | 0.0115 | SO 0.0281, link 0.0 |
| mst_eps1_seed0 | 0.4761 (0.4576-0.4941) | 0.5149 (0.4967-0.5331) | 0.0 | -0.0026 | SO 0.0, link 0.0 |
| mst_eps20_seed0 | 0.487 (0.4706-0.507) | 0.4986 (0.4799-0.5175) | 0.0 | 0.0128 | SO 0.0249, link 0.0 |
| mst_eps5_seed0 | 0.4874 (0.4697-0.505) | 0.4972 (0.4768-0.5178) | 0.0 | 0.0308 | SO 0.005, link 0.0 |
| mst_eps8_seed0 | 0.477 (0.4592-0.4941) | 0.5171 (0.5003-0.5357) | 0.0103 | 0.0377 | SO 0.0066, link 0.0 |
| patectgan_eps15_seed0 | 0.4904 (0.4734-0.51) | 0.4931 (0.4746-0.5122) | 0.0 | 0.0266 | SO 0.0, link 0.0 |
| patectgan_eps1_seed0 | 0.4851 (0.4655-0.5025) | 0.5009 (0.4821-0.5188) | 0.1437 | 0.0446 | SO 0.0, link 0.005 |
| patectgan_eps5_seed0 | 0.4882 (0.4704-0.5061) | 0.5011 (0.4827-0.5178) | 0.0 | 0.0125 | SO 0.0, link 0.0 |
| tvae_cap256_seed0 | 0.5026 (0.4846-0.5201) | 0.4862 (0.4697-0.5053) | 0.0 | 0.0396 | SO 0.0, link 0.0 |
| tvae_ep1000_seed0 | 0.5012 (0.4822-0.5209) | 0.5021 (0.4838-0.5201) | 0.0 | 0.0224 | SO 0.0, link 0.0 |
| tvae_ind_seed0 | 0.4924 (0.4737-0.5108) | 0.4935 (0.4741-0.5121) | 0.0 | 0.0322 | SO 0.005, link 0.0 |
| tvae_qt_seed0 | 0.4948 (0.4751-0.5113) | 0.4977 (0.4776-0.5152) | 0.0 | 0.024 | SO 0.0, link 0.0 |
| tvae_qt_seed1 | 0.4941 (0.4758-0.514) | 0.506 (0.4863-0.5237) | 0.0 | 0.0353 | SO 0.0, link 0.0 |
| tvae_qt_seed2 | 0.4941 (0.4739-0.5125) | 0.4826 (0.467-0.5015) | 0.0 | 0.0256 | SO 0.0, link 0.0 |
| tvae_seed0 | 0.4941 (0.4759-0.513) | 0.4863 (0.4694-0.5049) | 0.0 | 0.0163 | SO 0.0, link 0.0 |
| tvae_seed1 | 0.4944 (0.4744-0.5115) | 0.4701 (0.4518-0.4861) | 0.0 | 0.0174 | SO 0.0, link 0.0 |
| tvae_seed2 | 0.4947 (0.4755-0.5144) | 0.489 (0.4707-0.5085) | 0.0 | 0.0174 | SO 0.0, link 0.0 |

## Who is at risk: membership inference by patient atypicality

WITHIN-STRATUM AUC: members and non-members both get the same atypicality score (distance to their 5th-nearest member, same reference set and encoder), non-members are binned by the member quartile cut points, and each stratum's AUC compares members-in-Qi against non-members-in-Qi only. 0.5 = no leakage on that stratum; elevated values indicate SELECTIVE leakage on that stratum (e.g. a model that memorizes its unusual patients shows it in Q4). Strata with fewer than 30 non-members are skipped ('-'). Cell format: AUC (n members / n non-members).

| run | Q1 typical | Q2 | Q3 | Q4 atypical |
|---|---|---|---|---|
| aim40_eps1_seed0 | 0.4932 (944/349) | 0.4839 (943/327) | 0.4703 (943/283) | 0.503 (943/299) |
| aim40_eps5_seed0 | 0.496 (944/349) | 0.4785 (943/327) | 0.4687 (943/283) | 0.4994 (943/299) |
| aim50_eps1_seed0 | 0.5075 (944/349) | 0.4801 (943/327) | 0.4812 (943/283) | 0.5001 (943/299) |
| ctgan_qt_seed0 | 0.4965 (944/349) | 0.4737 (943/327) | 0.507 (943/283) | 0.5157 (943/299) |
| ctgan_seed0 | 0.5303 (944/349) | 0.4873 (943/327) | 0.4989 (943/283) | 0.4997 (943/299) |
| ctgan_seed1 | 0.4956 (944/349) | 0.4941 (943/327) | 0.5175 (943/283) | 0.5171 (943/299) |
| ctgan_seed2 | 0.5318 (944/349) | 0.4782 (943/327) | 0.5116 (943/283) | 0.5094 (943/299) |
| ddpm_g_seed0 | 0.4901 (944/349) | 0.4749 (943/327) | 0.4938 (943/283) | 0.4937 (943/299) |
| ddpm_seed0 | 0.4981 (944/349) | 0.4808 (943/327) | 0.4807 (943/283) | 0.4917 (943/299) |
| ddpm_seed1 | 0.5016 (944/349) | 0.4801 (943/327) | 0.5214 (943/283) | 0.4871 (943/299) |
| ddpm_seed2 | 0.5037 (944/349) | 0.4921 (943/327) | 0.4736 (943/283) | 0.5113 (943/299) |
| dpctgan_eps10_seed0 | 0.504 (944/349) | 0.5156 (943/327) | 0.4895 (943/283) | 0.4947 (943/299) |
| dpctgan_eps15_seed0 | 0.4893 (944/349) | 0.5089 (943/327) | 0.4942 (943/283) | 0.4986 (943/299) |
| dpctgan_eps15_seed1 | 0.4792 (944/349) | 0.4862 (943/327) | 0.491 (943/283) | 0.4982 (943/299) |
| dpctgan_eps15_seed2 | 0.5004 (944/349) | 0.4742 (943/327) | 0.4698 (943/283) | 0.4838 (943/299) |
| dpctgan_eps1_seed0 | 0.5143 (944/349) | 0.4806 (943/327) | 0.4927 (943/283) | 0.4959 (943/299) |
| dpctgan_eps20_seed0 | 0.4905 (944/349) | 0.4908 (943/327) | 0.475 (943/283) | 0.4876 (943/299) |
| dpctgan_eps5_seed0 | 0.5092 (944/349) | 0.4964 (943/327) | 0.4755 (943/283) | 0.4854 (943/299) |
| dpctgan_eps8_seed0 | 0.5015 (944/349) | 0.5027 (943/327) | 0.4882 (943/283) | 0.4833 (943/299) |
| gaussian_copula_seed0 | 0.5161 (944/349) | 0.4961 (943/327) | 0.4896 (943/283) | 0.4947 (943/299) |
| gaussian_copula_seed1 | 0.5016 (944/349) | 0.4697 (943/327) | 0.5006 (943/283) | 0.4868 (943/299) |
| gaussian_copula_seed2 | 0.4967 (944/349) | 0.5015 (943/327) | 0.496 (943/283) | 0.4842 (943/299) |
| mst_eps0p5_seed0 | 0.4965 (944/349) | 0.4817 (943/327) | 0.4946 (943/283) | 0.4809 (943/299) |
| mst_eps10_seed0 | 0.4651 (944/349) | 0.4896 (943/327) | 0.5451 (943/283) | 0.4848 (943/299) |
| mst_eps15_seed0 | 0.4874 (944/349) | 0.4798 (943/327) | 0.5426 (943/283) | 0.4803 (943/299) |
| mst_eps15_seed1 | 0.4895 (944/349) | 0.4773 (943/327) | 0.5468 (943/283) | 0.4946 (943/299) |
| mst_eps15_seed2 | 0.4849 (944/349) | 0.4715 (943/327) | 0.5449 (943/283) | 0.4871 (943/299) |
| mst_eps1_seed0 | 0.4935 (944/349) | 0.4775 (943/327) | 0.5016 (943/283) | 0.4784 (943/299) |
| mst_eps20_seed0 | 0.4792 (944/349) | 0.483 (943/327) | 0.5392 (943/283) | 0.4943 (943/299) |
| mst_eps5_seed0 | 0.4678 (944/349) | 0.4931 (943/327) | 0.529 (943/283) | 0.4969 (943/299) |
| mst_eps8_seed0 | 0.4727 (944/349) | 0.4743 (943/327) | 0.5192 (943/283) | 0.4797 (943/299) |
| patectgan_eps15_seed0 | 0.5134 (944/349) | 0.4882 (943/327) | 0.5086 (943/283) | 0.478 (943/299) |
| patectgan_eps1_seed0 | 0.4879 (944/349) | 0.4998 (943/327) | 0.5119 (943/283) | 0.4834 (943/299) |
| patectgan_eps5_seed0 | 0.5139 (944/349) | 0.4822 (943/327) | 0.4995 (943/283) | 0.4818 (943/299) |
| tvae_cap256_seed0 | 0.5316 (944/349) | 0.5159 (943/327) | 0.565 (943/283) | 0.5174 (943/299) |
| tvae_ep1000_seed0 | 0.5245 (944/349) | 0.5337 (943/327) | 0.5596 (943/283) | 0.4974 (943/299) |
| tvae_ind_seed0 | 0.503 (944/349) | 0.5068 (943/327) | 0.5569 (943/283) | 0.4995 (943/299) |
| tvae_qt_seed0 | 0.5039 (944/349) | 0.5182 (943/327) | 0.5552 (943/283) | 0.4934 (943/299) |
| tvae_qt_seed1 | 0.501 (944/349) | 0.5464 (943/327) | 0.5226 (943/283) | 0.4814 (943/299) |
| tvae_qt_seed2 | 0.5063 (944/349) | 0.5063 (943/327) | 0.544 (943/283) | 0.4989 (943/299) |
| tvae_seed0 | 0.511 (944/349) | 0.5167 (943/327) | 0.5396 (943/283) | 0.4889 (943/299) |
| tvae_seed1 | 0.5332 (944/349) | 0.5154 (943/327) | 0.5282 (943/283) | 0.5129 (943/299) |
| tvae_seed2 | 0.5044 (944/349) | 0.5391 (943/327) | 0.5238 (943/283) | 0.5037 (943/299) |

## Attribute inference detail

| run | sensitive attribute | baseline | member acc | non-member acc | advantage |
|---|---|---|---|---|---|
| aim40_eps1_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9711 | 0.9738 | -0.0027 |
| aim40_eps1_seed0 | ckd_severity_from_calculated_egfr | 0.4173 | 0.3043 | 0.2989 | +0.0054 |
| aim40_eps1_seed0 | nyha_nyha_pET | 0.4738 | 0.4042 | 0.3951 | +0.0091 |
| aim40_eps5_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9711 | 0.9738 | -0.0027 |
| aim40_eps5_seed0 | ckd_severity_from_calculated_egfr | 0.4173 | 0.2677 | 0.2599 | +0.0078 |
| aim40_eps5_seed0 | nyha_nyha_pET | 0.4738 | 0.3711 | 0.3569 | +0.0141 |
| aim50_eps1_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9711 | 0.9738 | -0.0027 |
| aim50_eps1_seed0 | ckd_severity_from_calculated_egfr | 0.4173 | 0.2534 | 0.2544 | -0.0010 |
| aim50_eps1_seed0 | nyha_nyha_pET | 0.4738 | 0.3506 | 0.3323 | +0.0184 |
| ctgan_qt_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.8969 | 0.9078 | -0.0109 |
| ctgan_qt_seed0 | ckd_severity_from_calculated_egfr | 0.4173 | 0.2677 | 0.2957 | -0.0280 |
| ctgan_qt_seed0 | nyha_nyha_pET | 0.4738 | 0.3726 | 0.3609 | +0.0118 |
| ctgan_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9014 | 0.9126 | -0.0112 |
| ctgan_seed0 | ckd_severity_from_calculated_egfr | 0.4173 | 0.3265 | 0.3227 | +0.0038 |
| ctgan_seed0 | nyha_nyha_pET | 0.4738 | 0.3167 | 0.3188 | -0.0020 |
| ctgan_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.7721 | 0.7687 | +0.0034 |
| ctgan_seed1 | ckd_severity_from_calculated_egfr | 0.4173 | 0.2656 | 0.2576 | +0.0080 |
| ctgan_seed1 | nyha_nyha_pET | 0.4738 | 0.3501 | 0.3315 | +0.0186 |
| ctgan_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.864 | 0.8625 | +0.0016 |
| ctgan_seed2 | ckd_severity_from_calculated_egfr | 0.4173 | 0.2613 | 0.2711 | -0.0097 |
| ctgan_seed2 | nyha_nyha_pET | 0.4738 | 0.4299 | 0.4078 | +0.0221 |
| ddpm_g_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9642 | 0.9674 | -0.0032 |
| ddpm_g_seed0 | ckd_severity_from_calculated_egfr | 0.4173 | 0.3284 | 0.3307 | -0.0023 |
| ddpm_g_seed0 | nyha_nyha_pET | 0.4738 | 0.216 | 0.2329 | -0.0169 |
| ddpm_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9634 | 0.965 | -0.0016 |
| ddpm_seed0 | ckd_severity_from_calculated_egfr | 0.4173 | 0.344 | 0.3514 | -0.0073 |
| ddpm_seed0 | nyha_nyha_pET | 0.4738 | 0.2187 | 0.2234 | -0.0047 |
| ddpm_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9475 | 0.9595 | -0.0119 |
| ddpm_seed1 | ckd_severity_from_calculated_egfr | 0.4173 | 0.3567 | 0.3609 | -0.0041 |
| ddpm_seed1 | nyha_nyha_pET | 0.4738 | 0.2356 | 0.2393 | -0.0036 |
| ddpm_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9584 | 0.9626 | -0.0043 |
| ddpm_seed2 | ckd_severity_from_calculated_egfr | 0.4173 | 0.3408 | 0.3386 | +0.0022 |
| ddpm_seed2 | nyha_nyha_pET | 0.4738 | 0.2346 | 0.2361 | -0.0015 |
| dpctgan_eps10_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9711 | 0.9738 | -0.0027 |
| dpctgan_eps10_seed0 | ckd_severity_from_calculated_egfr | 0.4173 | 0.4209 | 0.4173 | +0.0036 |
| dpctgan_eps10_seed0 | nyha_nyha_pET | 0.4738 | 0.1561 | 0.1693 | -0.0132 |
| dpctgan_eps15_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9698 | 0.973 | -0.0032 |
| dpctgan_eps15_seed0 | ckd_severity_from_calculated_egfr | 0.4173 | 0.1376 | 0.124 | +0.0135 |
| dpctgan_eps15_seed0 | nyha_nyha_pET | 0.4738 | 0.1391 | 0.1479 | -0.0087 |
| dpctgan_eps15_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9711 | 0.9738 | -0.0027 |
| dpctgan_eps15_seed1 | ckd_severity_from_calculated_egfr | 0.4173 | 0.2293 | 0.2504 | -0.0211 |
| dpctgan_eps15_seed1 | nyha_nyha_pET | 0.4738 | 0.1391 | 0.1479 | -0.0087 |
| dpctgan_eps15_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9711 | 0.9738 | -0.0027 |
| dpctgan_eps15_seed2 | ckd_severity_from_calculated_egfr | 0.4173 | 0.1413 | 0.128 | +0.0133 |
| dpctgan_eps15_seed2 | nyha_nyha_pET | 0.4738 | 0.3825 | 0.3879 | -0.0055 |
| dpctgan_eps1_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.0289 | 0.0262 | +0.0027 |
| dpctgan_eps1_seed0 | ckd_severity_from_calculated_egfr | 0.4173 | 0.4177 | 0.4141 | +0.0036 |
| dpctgan_eps1_seed0 | nyha_nyha_pET | 0.4738 | 0.2836 | 0.2663 | +0.0173 |
| dpctgan_eps20_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9711 | 0.9738 | -0.0027 |
| dpctgan_eps20_seed0 | ckd_severity_from_calculated_egfr | 0.4173 | 0.141 | 0.128 | +0.0130 |
| dpctgan_eps20_seed0 | nyha_nyha_pET | 0.4738 | 0.1391 | 0.1479 | -0.0087 |
| dpctgan_eps5_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9711 | 0.9738 | -0.0027 |
| dpctgan_eps5_seed0 | ckd_severity_from_calculated_egfr | 0.4173 | 0.1413 | 0.128 | +0.0133 |
| dpctgan_eps5_seed0 | nyha_nyha_pET | 0.4738 | 0.4535 | 0.4316 | +0.0218 |
| dpctgan_eps8_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9711 | 0.9738 | -0.0027 |
| dpctgan_eps8_seed0 | ckd_severity_from_calculated_egfr | 0.4173 | 0.2311 | 0.2528 | -0.0217 |
| dpctgan_eps8_seed0 | nyha_nyha_pET | 0.4738 | 0.4447 | 0.4324 | +0.0123 |
| gaussian_copula_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9396 | 0.9483 | -0.0088 |
| gaussian_copula_seed0 | ckd_severity_from_calculated_egfr | 0.4173 | 0.2998 | 0.2933 | +0.0064 |
| gaussian_copula_seed0 | nyha_nyha_pET | 0.4738 | 0.3512 | 0.3442 | +0.0070 |
| gaussian_copula_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9512 | 0.9571 | -0.0058 |
| gaussian_copula_seed1 | ckd_severity_from_calculated_egfr | 0.4173 | 0.3196 | 0.31 | +0.0096 |
| gaussian_copula_seed1 | nyha_nyha_pET | 0.4738 | 0.3798 | 0.3498 | +0.0300 |
| gaussian_copula_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9406 | 0.9459 | -0.0053 |
| gaussian_copula_seed2 | ckd_severity_from_calculated_egfr | 0.4173 | 0.317 | 0.3299 | -0.0129 |
| gaussian_copula_seed2 | nyha_nyha_pET | 0.4738 | 0.3742 | 0.3696 | +0.0046 |
| mst_eps0p5_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9669 | 0.969 | -0.0021 |
| mst_eps0p5_seed0 | ckd_severity_from_calculated_egfr | 0.4173 | 0.312 | 0.3331 | -0.0211 |
| mst_eps0p5_seed0 | nyha_nyha_pET | 0.4738 | 0.3729 | 0.3879 | -0.0150 |
| mst_eps10_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9565 | 0.961 | -0.0045 |
| mst_eps10_seed0 | ckd_severity_from_calculated_egfr | 0.4173 | 0.1717 | 0.1669 | +0.0048 |
| mst_eps10_seed0 | nyha_nyha_pET | 0.4738 | 0.4007 | 0.3903 | +0.0104 |
| mst_eps15_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9459 | 0.9499 | -0.0040 |
| mst_eps15_seed0 | ckd_severity_from_calculated_egfr | 0.4173 | 0.2149 | 0.2122 | +0.0027 |
| mst_eps15_seed0 | nyha_nyha_pET | 0.4738 | 0.388 | 0.3887 | -0.0007 |
| mst_eps15_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9536 | 0.9547 | -0.0011 |
| mst_eps15_seed1 | ckd_severity_from_calculated_egfr | 0.4173 | 0.2067 | 0.221 | -0.0143 |
| mst_eps15_seed1 | nyha_nyha_pET | 0.4738 | 0.4066 | 0.4006 | +0.0059 |
| mst_eps15_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9549 | 0.9579 | -0.0029 |
| mst_eps15_seed2 | ckd_severity_from_calculated_egfr | 0.4173 | 0.2327 | 0.2266 | +0.0062 |
| mst_eps15_seed2 | nyha_nyha_pET | 0.4738 | 0.4241 | 0.4126 | +0.0115 |
| mst_eps1_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9539 | 0.9579 | -0.0040 |
| mst_eps1_seed0 | ckd_severity_from_calculated_egfr | 0.4173 | 0.3138 | 0.3235 | -0.0097 |
| mst_eps1_seed0 | nyha_nyha_pET | 0.4738 | 0.3742 | 0.3768 | -0.0026 |
| mst_eps20_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9703 | 0.9722 | -0.0019 |
| mst_eps20_seed0 | ckd_severity_from_calculated_egfr | 0.4173 | 0.171 | 0.1582 | +0.0128 |
| mst_eps20_seed0 | nyha_nyha_pET | 0.4738 | 0.3886 | 0.403 | -0.0145 |
| mst_eps5_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9669 | 0.969 | -0.0021 |
| mst_eps5_seed0 | ckd_severity_from_calculated_egfr | 0.4173 | 0.3098 | 0.279 | +0.0308 |
| mst_eps5_seed0 | nyha_nyha_pET | 0.4738 | 0.397 | 0.4078 | -0.0108 |
| mst_eps8_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9534 | 0.9523 | +0.0010 |
| mst_eps8_seed0 | ckd_severity_from_calculated_egfr | 0.4173 | 0.3334 | 0.2957 | +0.0377 |
| mst_eps8_seed0 | nyha_nyha_pET | 0.4738 | 0.3878 | 0.376 | +0.0118 |
| patectgan_eps15_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9711 | 0.9738 | -0.0027 |
| patectgan_eps15_seed0 | ckd_severity_from_calculated_egfr | 0.4173 | 0.3138 | 0.3299 | -0.0161 |
| patectgan_eps15_seed0 | nyha_nyha_pET | 0.4738 | 0.4129 | 0.3863 | +0.0266 |
| patectgan_eps1_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.8349 | 0.8577 | -0.0228 |
| patectgan_eps1_seed0 | ckd_severity_from_calculated_egfr | 0.4173 | 0.2878 | 0.2432 | +0.0446 |
| patectgan_eps1_seed0 | nyha_nyha_pET | 0.4738 | 0.3875 | 0.3776 | +0.0099 |
| patectgan_eps5_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9647 | 0.9634 | +0.0013 |
| patectgan_eps5_seed0 | ckd_severity_from_calculated_egfr | 0.4173 | 0.3856 | 0.3792 | +0.0065 |
| patectgan_eps5_seed0 | nyha_nyha_pET | 0.4738 | 0.3536 | 0.341 | +0.0125 |
| tvae_cap256_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9382 | 0.9404 | -0.0021 |
| tvae_cap256_seed0 | ckd_severity_from_calculated_egfr | 0.4173 | 0.3586 | 0.3307 | +0.0279 |
| tvae_cap256_seed0 | nyha_nyha_pET | 0.4738 | 0.4752 | 0.4356 | +0.0396 |
| tvae_ep1000_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9536 | 0.9539 | -0.0003 |
| tvae_ep1000_seed0 | ckd_severity_from_calculated_egfr | 0.4173 | 0.3252 | 0.3458 | -0.0206 |
| tvae_ep1000_seed0 | nyha_nyha_pET | 0.4738 | 0.4636 | 0.4412 | +0.0224 |
| tvae_ind_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9425 | 0.9444 | -0.0019 |
| tvae_ind_seed0 | ckd_severity_from_calculated_egfr | 0.4173 | 0.3459 | 0.3561 | -0.0102 |
| tvae_ind_seed0 | nyha_nyha_pET | 0.4738 | 0.4797 | 0.4475 | +0.0322 |
| tvae_qt_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9478 | 0.9436 | +0.0042 |
| tvae_qt_seed0 | ckd_severity_from_calculated_egfr | 0.4173 | 0.3329 | 0.3529 | -0.0200 |
| tvae_qt_seed0 | nyha_nyha_pET | 0.4738 | 0.4882 | 0.4642 | +0.0240 |
| tvae_qt_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9385 | 0.9356 | +0.0029 |
| tvae_qt_seed1 | ckd_severity_from_calculated_egfr | 0.4173 | 0.3597 | 0.3243 | +0.0353 |
| tvae_qt_seed1 | nyha_nyha_pET | 0.4738 | 0.4824 | 0.4587 | +0.0237 |
| tvae_qt_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.929 | 0.9308 | -0.0019 |
| tvae_qt_seed2 | ckd_severity_from_calculated_egfr | 0.4173 | 0.3305 | 0.3172 | +0.0133 |
| tvae_qt_seed2 | nyha_nyha_pET | 0.4738 | 0.4842 | 0.4587 | +0.0256 |
| tvae_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9425 | 0.9483 | -0.0058 |
| tvae_seed0 | ckd_severity_from_calculated_egfr | 0.4173 | 0.3522 | 0.3577 | -0.0055 |
| tvae_seed0 | nyha_nyha_pET | 0.4738 | 0.4869 | 0.4706 | +0.0163 |
| tvae_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9385 | 0.9491 | -0.0106 |
| tvae_seed1 | ckd_severity_from_calculated_egfr | 0.4173 | 0.3382 | 0.3569 | -0.0187 |
| tvae_seed1 | nyha_nyha_pET | 0.4738 | 0.4856 | 0.4682 | +0.0174 |
| tvae_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.9738 | 0.9502 | 0.9531 | -0.0029 |
| tvae_seed2 | ckd_severity_from_calculated_egfr | 0.4173 | 0.3541 | 0.3474 | +0.0067 |
| tvae_seed2 | nyha_nyha_pET | 0.4738 | 0.4848 | 0.4674 | +0.0174 |

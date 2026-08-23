# Adversarial Privacy Attacks

Members: 3520 training records; non-members: 1174 holdout records (real, unseen patients). Membership inference AUC of 0.5 means the synthetic data reveals nothing about who was in the training set. Attribute inference reports the MEMBERSHIP ADVANTAGE -- accuracy on members minus accuracy on non-members; population-level inference (both above baseline, equally) is the intended use of released data, only member-specific advantage is leakage.

| run | MIA AUC (95% CI) | worst AIA membership advantage | anonymeter |
|---|---|---|---|
| aim50_eps1_seed0 | 0.4995 (0.4809-0.5182) | 0.0037 | SO 0.0071, link - |
| aim50_eps5_seed0 | 0.499 (0.4802-0.5199) | 0.0037 | SO 0.0, link - |
| ctgan_seed0 | 0.4786 (0.4609-0.4976) | 0.0227 | SO 0.0, link - |
| ctgan_seed1 | 0.4973 (0.4784-0.5147) | 0.0 | skipped |
| ctgan_seed2 | 0.5149 (0.4957-0.531) | 0.0264 | SO 0.0, link - |
| dpctgan_eps10_seed0 | 0.497 (0.4784-0.5166) | 0.0 | skipped |
| dpctgan_eps15_seed0 | 0.5003 (0.4815-0.5191) | 0.0066 | SO 0.0, link - |
| dpctgan_eps15_seed1 | 0.5004 (0.4812-0.5214) | 0.0 | skipped |
| dpctgan_eps15_seed2 | 0.5007 (0.4811-0.5196) | 0.0319 | skipped |
| dpctgan_eps1_seed0 | 0.4997 (0.4817-0.5189) | 0.0 | skipped |
| dpctgan_eps20_seed0 | 0.4983 (0.4802-0.5166) | 0.0319 | SO 0.0, link - |
| dpctgan_eps5_seed0 | 0.5001 (0.483-0.5202) | 0.0037 | SO 0.0, link - |
| dpctgan_eps8_seed0 | 0.4985 (0.4803-0.5184) | 0.0037 | SO 0.0, link - |
| gaussian_copula_seed0 | 0.4934 (0.4772-0.5113) | 0.0 | skipped |
| gaussian_copula_seed1 | 0.5 (0.4801-0.5178) | 0.0153 | skipped |
| gaussian_copula_seed2 | 0.498 (0.4791-0.5174) | 0.0333 | SO 0.0, link - |
| mst_eps10_seed0 | 0.5011 (0.4816-0.5214) | 0.0 | SO 0.0066, link - |
| mst_eps15_seed0 | 0.4924 (0.4711-0.5112) | 0.0043 | skipped |
| mst_eps15_seed1 | 0.4997 (0.4823-0.5192) | 0.0105 | SO 0.0364, link - |
| mst_eps15_seed2 | 0.5055 (0.4868-0.525) | 0.0071 | SO 0.0132, link - |
| mst_eps1_seed0 | 0.5047 (0.4839-0.5228) | 0.0234 | SO 0.005, link - |
| mst_eps20_seed0 | 0.498 (0.4792-0.5164) | 0.0034 | SO 0.0281, link - |
| mst_eps5_seed0 | 0.4893 (0.4703-0.5082) | 0.0 | SO 0.0198, link - |
| mst_eps8_seed0 | 0.4967 (0.4755-0.5158) | 0.0068 | skipped |
| tvae_seed0 | 0.5195 (0.4994-0.5384) | 0.0362 | skipped |
| tvae_seed1 | 0.5137 (0.4949-0.5316) | 0.0208 | SO 0.0033, link - |
| tvae_seed2 | 0.5041 (0.4841-0.5207) | 0.0244 | skipped |

## Attribute inference detail

| run | sensitive attribute | baseline | member acc | non-member acc | advantage |
|---|---|---|---|---|---|
| aim50_eps1_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 1.0 | 0.5719 | 0.5681 | +0.0037 |
| aim50_eps1_seed0 | ckd_severity_from_calculated_egfr | 0.3798 | 0.225 | 0.2462 | -0.0212 |
| aim50_eps1_seed0 | nyha_nyha_pET | 0.7653 | 0.0 | 0.0 | +0.0000 |
| aim50_eps5_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 1.0 | 0.5719 | 0.5681 | +0.0037 |
| aim50_eps5_seed0 | ckd_severity_from_calculated_egfr | 0.2875 | 0.1688 | 0.1823 | -0.0135 |
| aim50_eps5_seed0 | nyha_nyha_pET | 0.7639 | 0.0 | 0.0 | +0.0000 |
| ctgan_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6179 | 0.4452 | 0.4225 | +0.0227 |
| ctgan_seed0 | ckd_severity_from_calculated_egfr | 0.3199 | 0.1957 | 0.1857 | +0.0100 |
| ctgan_seed0 | nyha_nyha_pET | 0.8446 | 0.0 | 0.0 | +0.0000 |
| ctgan_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.6395 | 0.5131 | 0.5187 | -0.0057 |
| ctgan_seed1 | ckd_severity_from_calculated_egfr | 0.3989 | 0.2091 | 0.2206 | -0.0115 |
| ctgan_seed1 | nyha_nyha_pET | 0.7145 | 0.0 | 0.0 | +0.0000 |
| ctgan_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.5014 | 0.4966 | 0.4702 | +0.0264 |
| ctgan_seed2 | ckd_severity_from_calculated_egfr | 0.2994 | 0.1855 | 0.1789 | +0.0066 |
| ctgan_seed2 | nyha_nyha_pET | 0.871 | 0.0 | 0.0 | +0.0000 |
| dpctgan_eps10_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9974 | 0.4037 | 0.4114 | -0.0077 |
| dpctgan_eps10_seed0 | ckd_severity_from_calculated_egfr | 1.0 | 0.2895 | 0.2905 | -0.0010 |
| dpctgan_eps10_seed0 | nyha_nyha_pET | 1.0 | 0.0 | 0.0 | +0.0000 |
| dpctgan_eps15_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9966 | 0.569 | 0.5681 | +0.0009 |
| dpctgan_eps15_seed0 | ckd_severity_from_calculated_egfr | 0.9795 | 0.0491 | 0.0426 | +0.0066 |
| dpctgan_eps15_seed0 | nyha_nyha_pET | 0.9142 | 0.0 | 0.0 | +0.0000 |
| dpctgan_eps15_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.9991 | 0.4037 | 0.4114 | -0.0077 |
| dpctgan_eps15_seed1 | ckd_severity_from_calculated_egfr | 0.9955 | 0.2895 | 0.2905 | -0.0010 |
| dpctgan_eps15_seed1 | nyha_nyha_pET | 1.0 | 0.0 | 0.0 | +0.0000 |
| dpctgan_eps15_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 0.0 | 0.0 | +0.0000 |
| dpctgan_eps15_seed2 | ckd_severity_from_calculated_egfr | 0.998 | 0.1869 | 0.155 | +0.0319 |
| dpctgan_eps15_seed2 | nyha_nyha_pET | 1.0 | 0.0 | 0.0 | +0.0000 |
| dpctgan_eps1_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6756 | 0.1838 | 0.1882 | -0.0044 |
| dpctgan_eps1_seed0 | ckd_severity_from_calculated_egfr | 0.9977 | 0.2895 | 0.2905 | -0.0010 |
| dpctgan_eps1_seed0 | nyha_nyha_pET | 1.0 | 0.0 | 0.0 | +0.0000 |
| dpctgan_eps20_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9938 | 0.0472 | 0.04 | +0.0071 |
| dpctgan_eps20_seed0 | ckd_severity_from_calculated_egfr | 0.9974 | 0.1869 | 0.155 | +0.0319 |
| dpctgan_eps20_seed0 | nyha_nyha_pET | 1.0 | 0.0 | 0.0 | +0.0000 |
| dpctgan_eps5_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 0.5719 | 0.5681 | +0.0037 |
| dpctgan_eps5_seed0 | ckd_severity_from_calculated_egfr | 0.9977 | 0.177 | 0.1797 | -0.0027 |
| dpctgan_eps5_seed0 | nyha_nyha_pET | 0.7384 | 0.0 | 0.0 | +0.0000 |
| dpctgan_eps8_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.9994 | 0.5719 | 0.5681 | +0.0037 |
| dpctgan_eps8_seed0 | ckd_severity_from_calculated_egfr | 0.9858 | 0.2011 | 0.2129 | -0.0118 |
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
| mst_eps10_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5741 | 0.4719 | 0.4855 | -0.0136 |
| mst_eps10_seed0 | ckd_severity_from_calculated_egfr | 0.2929 | 0.1881 | 0.1882 | -0.0002 |
| mst_eps10_seed0 | nyha_nyha_pET | 0.7551 | 0.0 | 0.0 | +0.0000 |
| mst_eps15_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5722 | 0.5307 | 0.5264 | +0.0043 |
| mst_eps15_seed0 | ckd_severity_from_calculated_egfr | 0.2872 | 0.1625 | 0.1712 | -0.0087 |
| mst_eps15_seed0 | nyha_nyha_pET | 0.7591 | 0.0 | 0.0 | +0.0000 |
| mst_eps15_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.575 | 0.4739 | 0.4634 | +0.0105 |
| mst_eps15_seed1 | ckd_severity_from_calculated_egfr | 0.2932 | 0.1787 | 0.2078 | -0.0291 |
| mst_eps15_seed1 | nyha_nyha_pET | 0.754 | 0.0 | 0.0 | +0.0000 |
| mst_eps15_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.5739 | 0.4526 | 0.4455 | +0.0071 |
| mst_eps15_seed2 | ckd_severity_from_calculated_egfr | 0.2872 | 0.1787 | 0.1874 | -0.0087 |
| mst_eps15_seed2 | nyha_nyha_pET | 0.7616 | 0.0 | 0.0 | +0.0000 |
| mst_eps1_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5719 | 0.5616 | 0.5622 | -0.0005 |
| mst_eps1_seed0 | ckd_severity_from_calculated_egfr | 0.2918 | 0.2824 | 0.2589 | +0.0234 |
| mst_eps1_seed0 | nyha_nyha_pET | 0.7156 | 0.0 | 0.0 | +0.0000 |
| mst_eps20_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5744 | 0.4608 | 0.4574 | +0.0034 |
| mst_eps20_seed0 | ckd_severity_from_calculated_egfr | 0.2884 | 0.177 | 0.1831 | -0.0061 |
| mst_eps20_seed0 | nyha_nyha_pET | 0.7577 | 0.0 | 0.0 | +0.0000 |
| mst_eps5_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5702 | 0.4889 | 0.4915 | -0.0026 |
| mst_eps5_seed0 | ckd_severity_from_calculated_egfr | 0.2972 | 0.1943 | 0.1985 | -0.0041 |
| mst_eps5_seed0 | nyha_nyha_pET | 0.7537 | 0.0 | 0.0 | +0.0000 |
| mst_eps8_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5784 | 0.4898 | 0.483 | +0.0068 |
| mst_eps8_seed0 | ckd_severity_from_calculated_egfr | 0.2906 | 0.1639 | 0.184 | -0.0201 |
| mst_eps8_seed0 | nyha_nyha_pET | 0.758 | 0.0 | 0.0 | +0.0000 |
| tvae_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.5259 | 0.5284 | 0.5298 | -0.0014 |
| tvae_seed0 | ckd_severity_from_calculated_egfr | 0.3392 | 0.2628 | 0.2266 | +0.0362 |
| tvae_seed0 | nyha_nyha_pET | 0.9631 | 0.0 | 0.0 | +0.0000 |
| tvae_seed1 | cause_of_death_isAllCause_f5a_w5a_first | 0.5344 | 0.5574 | 0.5366 | +0.0208 |
| tvae_seed1 | ckd_severity_from_calculated_egfr | 0.3051 | 0.2537 | 0.2368 | +0.0169 |
| tvae_seed1 | nyha_nyha_pET | 0.9619 | 0.0 | 0.0 | +0.0000 |
| tvae_seed2 | cause_of_death_isAllCause_f5a_w5a_first | 0.5165 | 0.5355 | 0.5111 | +0.0244 |
| tvae_seed2 | ckd_severity_from_calculated_egfr | 0.331 | 0.2378 | 0.2428 | -0.0050 |
| tvae_seed2 | nyha_nyha_pET | 0.977 | 0.0 | 0.0 | +0.0000 |

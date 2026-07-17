# Project Alpha Feature Importance Analysis

## Summary

- Total features: 80
- Common features: 80
- RandomForest-only features: 0
- LightGBM-only features: 0
- Low-importance candidates: 20
- High-disagreement features: 8

## Top Consensus Features

| consensus_rank | feature | consensus_score | random_forest_rank | lightgbm_rank | family | position | side |
|---|---|---|---|---|---|---|---|
| 1 | home_core_DF | 0.027794 | 1 | 1 | Core player strength | DF | Home |
| 2 | home_core_FW | 0.025839 | 2 | 2 | Core player strength | FW | Home |
| 3 | away_core_DF | 0.025399 | 5 | 3 | Core player strength | DF | Away |
| 4 | away_momentum_FW | 0.025345 | 6 | 4 | Momentum | FW | Away |
| 5 | home_team_momentum | 0.025211 | 3 | 5 | Momentum | Team-level | Home |
| 6 | away_momentum_MF | 0.024940 | 8 | 6 | Momentum | MF | Away |
| 7 | home_momentum_FW | 0.024682 | 4 | 7 | Momentum | FW | Home |
| 8 | away_core_FW | 0.023806 | 10 | 8 | Core player strength | FW | Away |
| 9 | home_core_MF | 0.023552 | 12 | 9 | Core player strength | MF | Home |
| 10 | away_core_MF | 0.023418 | 9 | 12 | Core player strength | MF | Away |
| 11 | home_momentum_MF | 0.023365 | 15 | 10 | Momentum | MF | Home |
| 12 | away_team_momentum | 0.023215 | 14 | 11 | Momentum | Team-level | Away |
| 13 | home_team_core_score | 0.023011 | 7 | 13 | Core player strength | Team-level | Home |
| 14 | away_team_core_score | 0.022040 | 16 | 14 | Core player strength | Team-level | Away |
| 15 | away_momentum_DF | 0.022022 | 13 | 15 | Momentum | DF | Away |
| 16 | home_momentum_DF | 0.021606 | 11 | 16 | Momentum | DF | Home |
| 17 | momentum_FWRatio | 0.020778 | 18 | 17 | Momentum | FW | Ratio |
| 18 | core_MFDiff | 0.020034 | 27 | 18 | Core player strength | MF | Difference |
| 19 | core_FWDiff | 0.020028 | 25 | 19 | Core player strength | FW | Difference |
| 20 | team_momentumRatio | 0.019918 | 17 | 20 | Momentum | Team-level | Ratio |
| 21 | core_DFDiff | 0.019535 | 26 | 21 | Core player strength | DF | Difference |
| 22 | team_core_scoreRatio | 0.019379 | 21 | 23 | Core player strength | Team-level | Ratio |
| 23 | momentum_DFDiff | 0.019340 | 24 | 22 | Momentum | DF | Difference |
| 24 | momentum_FWDiff | 0.019290 | 19 | 26 | Momentum | FW | Difference |
| 25 | team_core_scoreDiff | 0.019288 | 20 | 25 | Core player strength | Team-level | Difference |
| 26 | team_momentumDiff | 0.019257 | 23 | 24 | Momentum | Team-level | Difference |
| 27 | momentum_MFDiff | 0.018900 | 28 | 27 | Momentum | MF | Difference |
| 28 | core_MFRatio | 0.018807 | 22 | 28 | Core player strength | MF | Ratio |
| 29 | momentum_DFRatio | 0.018516 | 29 | 29 | Momentum | DF | Ratio |
| 30 | core_DFRatio | 0.018143 | 31 | 30 | Core player strength | DF | Ratio |

## Feature Family Summary

| family | features | random_forest_importance | lightgbm_importance | consensus_importance | average_rank |
|---|---|---|---|---|---|
| Core player strength | 24 | 0.405307 | 0.422745 | 0.413328 | 26.500000 |
| Momentum | 20 | 0.370974 | 0.387892 | 0.378897 | 22.500000 |
| Availability | 20 | 0.104772 | 0.086116 | 0.094932 | 63.050000 |
| Starter availability | 12 | 0.068877 | 0.055730 | 0.062027 | 61.750000 |
| Team stability | 4 | 0.050070 | 0.047517 | 0.048759 | 38.000000 |

## Position Summary

| position | features | random_forest_importance | lightgbm_importance | consensus_importance |
|---|---|---|---|---|
| Team-level | 20 | 0.270238 | 0.261690 | 0.265497 |
| MF | 16 | 0.230733 | 0.231125 | 0.230536 |
| FW | 16 | 0.219697 | 0.225883 | 0.222219 |
| DF | 16 | 0.210458 | 0.211547 | 0.210590 |
| GK | 12 | 0.068874 | 0.069755 | 0.069100 |

## Home/Away/Difference Summary

| side | features | random_forest_importance | lightgbm_importance | consensus_importance |
|---|---|---|---|---|
| Ratio | 20 | 0.275937 | 0.244550 | 0.259809 |
| Home | 20 | 0.240766 | 0.271939 | 0.255629 |
| Away | 20 | 0.237630 | 0.264243 | 0.250458 |
| Difference | 20 | 0.245666 | 0.219268 | 0.232046 |

## Low-Importance Removal Candidates

| feature | consensus_rank | consensus_score | random_forest_rank | lightgbm_rank | family |
|---|---|---|---|---|---|
| available_count_FWDiff | 61 | 0.005245 | 53 | 64 | Availability |
| home_available_count_MF | 62 | 0.005202 | 57 | 63 | Availability |
| starter_count_DFRatio | 63 | 0.004977 | 64 | 62 | Starter availability |
| home_available_count_DF | 64 | 0.004968 | 65 | 59 | Availability |
| starter_count_FWDiff | 65 | 0.004786 | 61 | 66 | Starter availability |
| away_available_count_DF | 66 | 0.004154 | 67 | 67 | Availability |
| home_available_count_FW | 67 | 0.004019 | 66 | 68 | Availability |
| away_starter_count_FW | 68 | 0.003834 | 69 | 69 | Starter availability |
| home_starter_count_FW | 69 | 0.003693 | 68 | 70 | Starter availability |
| away_starter_count_DF | 70 | 0.003668 | 74 | 65 | Starter availability |
| starter_count_DFDiff | 71 | 0.003246 | 71 | 72 | Starter availability |
| available_player_countDiff | 72 | 0.003146 | 72 | 71 | Availability |
| available_player_countRatio | 73 | 0.002987 | 70 | 76 | Availability |
| away_available_player_count | 74 | 0.002842 | 73 | 73 | Availability |
| home_available_player_count | 75 | 0.002496 | 75 | 75 | Availability |
| home_starter_count_DF | 76 | 0.002460 | 76 | 74 | Starter availability |
| available_count_GKDiff | 77 | 0.000959 | 77 | 77 | Availability |
| available_count_GKRatio | 78 | 0.000696 | 78 | 78 | Availability |
| away_available_count_GK | 79 | 0.000538 | 79 | 79 | Availability |
| home_available_count_GK | 80 | 0.000327 | 80 | 80 | Availability |

## High Model Disagreement

| feature | random_forest_importance | lightgbm_importance | random_forest_rank | lightgbm_rank | relative_disagreement |
|---|---|---|---|---|---|
| available_count_FWDiff | 0.006478 | 0.004131 | 53 | 64 | 0.362254 |
| home_starter_count_FW | 0.004604 | 0.002874 | 68 | 70 | 0.375684 |
| starter_count_DFDiff | 0.004049 | 0.002524 | 71 | 72 | 0.376779 |
| available_player_countRatio | 0.004412 | 0.001817 | 70 | 76 | 0.588216 |
| available_count_GKDiff | 0.001314 | 0.000656 | 77 | 77 | 0.500894 |
| available_count_GKRatio | 0.000947 | 0.000480 | 78 | 78 | 0.493593 |
| away_available_count_GK | 0.000784 | 0.000334 | 79 | 79 | 0.573914 |
| home_available_count_GK | 0.000607 | 0.000127 | 80 | 80 | 0.790270 |

## Next Validation Step

Run an ablation backtest after removing low-importance features. Do not remove features solely from importance scores.
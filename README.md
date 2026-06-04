#  Spaceship Titanic — Binary Classification
**Kaggle Competition | Predict Passenger Transport to Alternate Dimension**

##  Project Overview

In the year 2912, the Spaceship Titanic collided with a spacetime anomaly and nearly half of its 13,000 passengers were transported to an alternate dimension. Using recovered personal records, the task is to **predict which passengers were transported** (True/False binary classification).

- **Goal:** Predict `Transported` (True/False) for 4,277 test passengers
- **Metric:** Classification Accuracy
- **Training samples:** ~8,700 passengers
- **Test samples:** ~4,277 passengers

##  File Structure

```
├── train.csv                  # 8,693 rows, 14 columns (includes Transported)
├── test.csv                   # 4,277 rows, 13 columns (no Transported)
├── sample_submission.csv      # Required submission format
└── spaceship_submission.csv   #  Final predictions (ready to upload)

##  Pipeline Summary

### 1. Feature Engineering
| New Feature | Description |
|---|---|
| `Deck`, `Side` | Split from `Cabin` (format: deck/num/side) |
| `GroupSize` | Number of passengers sharing the same group ID |
| `Solo` | Binary flag: passenger travelling alone |
| `TotalSpend` | Sum of RoomService + FoodCourt + ShoppingMall + Spa + VRDeck |
| `AnySpend` | Binary: spent anything at all |
| `IsChild` | Binary: Age < 13 |

### 2. Missing Value Imputation
| Strategy | Features |
|---|---|
| Fill `0` | All 5 spending columns |
| Fill `False` | CryoSleep, VIP |
| Median | Age |
| Mode | HomePlanet, Destination, Deck, Side |

### 3. Encoding
- CryoSleep, VIP → integer (0/1)
- HomePlanet, Destination, Deck, Side → LabelEncoder
- Dropped: Name, CabinNum, PassengerId, Group (after extracting GroupSize)

### 4. Models & Ensemble

| Model | CV Accuracy | Blend Weight |
|---|---|---|
| XGBoost | 0.80179 ± 0.011 | 40% |
| LightGBM | 0.79823 ± 0.007 | 40% |
| GradientBoosting | 0.80490 ± 0.009 | 20% |

Final prediction = averaged probabilities → threshold at 0.5 → True/False


##  Submission Validation

| Check | Result |
|---|---|
| Rows |  4,277 |
| Columns |  `PassengerId`, `Transported` |
| PassengerIds |  Exact match with test.csv |
| Values |  True / False only |
| Nulls |  0 |
| True count |  2,215 (~51.8%) |
| False count |  2,062 (~48.2%) |

##  How to Submit

1. Go to [Kaggle Competition Page](https://www.kaggle.com/c/spaceship-titanic)
2. Click **Submit Predictions**
3. Upload `spaceship_submission.csv`
4. View your Accuracy score on the leaderboard

##  Dependencies

```bash
pip install pandas numpy scikit-learn xgboost lightgbm

## Feature Reference

| Feature | Type | Description |
|---|---|---|
| PassengerId | str | gggg_pp — group and position |
| HomePlanet | cat | Departure planet |
| CryoSleep | bool | In suspended animation |
| Cabin | str | deck/num/side |
| Destination | cat | Destination exoplanet |
| Age | float | Passenger age |
| VIP | bool | VIP service |
| RoomService | float | Billing amount |
| FoodCourt | float | Billing amount |
| ShoppingMall | float | Billing amount |
| Spa | float | Billing amount |
| VRDeck | float | Billing amount |
| Name | str | Dropped (not predictive) |
| **Transported** | **bool** | **Target variable** |

Author:Afreed,Ashraf

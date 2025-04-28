# Movie Rating Prediction

Predict IMDb ratings of Indian movies using robust ML techniques and meaningful feature engineering.

---

## 📦 Dataset
The dataset contains the following columns:

- **Name** — Movie title  
- **Year** — Release year  
- **Duration** — Runtime in minutes  
- **Genre** — Primary genre (or comma-separated list)  
- **Rating** — IMDb rating  
- **Votes** — Number of votes  
- **Director** — Director name  
- **Actor 1**, **Actor 2**, **Actor 3** — Top three billed cast members  

---

## 🔍 Exploratory Data Analysis
- **Rating Distribution**: Right-skewed, most films rate between 6 and 8.  
- **Top Genres**: Drama and Action dominate.  

---

## 🛠️ Approach

1. **Data Loading & Cleaning**  
   - Load CSV with `encoding='ISO-8859-1'`.  
   - Drop rows missing `Rating`.  
   - Convert `Duration` (e.g. “109 min”) to numeric and fill missing with median.  
   - Fill missing `Votes` with 0.  
   - Coerce `Year` to integer.  

2. **EDA**  
   - Histogram of `Rating`.  
   - Bar chart of top `Genre` counts.  

3. **Feature Engineering**  
   - **Director_Avg_Rating** & **Genre_Avg_Rating**  
   - **Movie_Age** = 2025 − `Year`  
   - **Num_Genres** = count of comma-separated items in `Genre`  
   - **Log_Votes** = log1p(`Votes`)  
   - **Scaled_Duration** & **Scaled_Votes** via `StandardScaler`  
   - **Actor_\<n\>_Avg_Rating** for each of `Actor 1–3`  
   - **Title_Word_Count** & **Title_Char_Count** from `Name`  
   - **Decade** = floor(`Year`/10)×10 (Label-encoded if needed)  
   - **Director_Genre_Interaction** = `Director_Avg_Rating` × `Genre_Avg_Rating`  
   - **Year_Avg_Rating** = average rating by `Year`  
   - **Title_Sentiment** = TextBlob polarity of `Name`  
   - **Time_Since_Last_Movie** = `Year` − director’s most recent prior `Year`  
   - **Cumulative_Votes** per director  
   - **PCA** (PC1, PC2) on `[Duration, Votes, Director_Avg_Rating, Genre_Avg_Rating]`  

4. **Modeling**  
   - **Random Forest** with `GridSearchCV` (5-fold CV over 24 hyperparameter combos)  
   - **XGBoost** baseline  
   - Compare via **MSE** & **R²**  

5. **Visualization**  
   - Feature importances  
   - Model comparison charts (bar plots of MSE & R²)  

6. **Reproducibility**  
   - Fixed `random_state` across all algorithms  
   - Include `requirements.txt` for dependencies  
   - Meaningful Git commits  

---

## 📈 Results


| Model          |  MSE   |   R²    |
|:---------------|:------:|:-------:|
| Random Forest  | 0.7070 | 0.6197  |
| XGBoost        | 0.8035 | 0.5678  |

---
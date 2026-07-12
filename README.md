# GeoOptim — Multi-Label POI Classification for Kathmandu

Geospatial machine learning pipeline that predicts the presence of six commercial POI
categories across Kathmandu Valley using H3 hexagonal grid features derived from
Overture Maps, WorldPop population data, Copernicus DEM, and building footprints.

## Project Structure

```
Code/
  0_Data_download.ipynb      # Download raw Overture Maps data + Copernicus DEM
  1_Data.ipynb               # Exploratory data analysis
  2_data_cleaning.ipynb      # Clean POIs, roads, population raster
  3_Data_engineering.ipynb   # Feature engineering, H3 binning, inference grid
  4_Train_Models.ipynb       # Train RF, LightGBM, MLP, XGBoost classifiers
  5_gridsearch.ipynb         # Empty placeholder
  6_Inference.ipynb          # Run MLP model on full prediction grid
Docs/
  super_class.csv            # 206 Overture categories mapped to 18 super classes
  GeoOptima_Proposal.pdf
  Proposal.docx
Output/
  Models/                    # Serialized .pkl model files + scaler + thresholds
Data/                        # Gitignored — run notebook 0 to populate
```

## Dependencies

```bash
pip install pandas numpy geopandas shapely h3 rasterio rasterstats \
  scikit-learn xgboost lightgbm overturemaps joblib matplotlib seaborn
```

## Pipeline

Run notebooks in order. Each produces outputs consumed by later steps.

### High-Level Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GeoOptim Pipeline — Overview                         │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐
  │  Notebook 0  │   DATA ACQUISITION
  │  Download    │   ─────────────────
  │              │   Fetches raw geospatial data from Overture Maps API
  │              │   (places, roads, buildings, land use, water) and
  │              │   downloads Copernicus GLO-30 DEM elevation tile.
  │              │   Crops all layers to Kathmandu bbox.
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  Notebook 1  │   EXPLORATORY DATA ANALYSIS
  │  EDA Only    │   ─────────────────────────
  │              │   Inspects raw/processed layers. Plots distributions
  │              │   of POI categories, road classes, elevation, and
  │              │   population density. No outputs consumed downstream.
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  Notebook 2  │   DATA CLEANING
  │  Clean       │   ─────────────
  │              │   • Drops NAs, keeps minimal columns
  │              │   • Maps 206 Overture categories → 18 super classes
  │              │   • Groups road classes into 3 functional groups
  │              │   • Clips WorldPop raster to study area
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  Notebook 3  │   FEATURE ENGINEERING
  │  Engineer    │   ──────────────────
  │              │   • Converts POIs → H3 hexagons (res 10)
  │              │   • Computes multi-scale buffer POI counts (500m, 1km, 2km)
  │              │   • Calculates road distances + densities per hexagon
  │              │   • Extracts population counts from raster
  │              │   • Computes building footprint statistics
  │              │   • Extracts DEM elevation + slope per hexagon
  │              │   • Builds training dataset + full inference grid
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  Notebook 4  │   MODEL TRAINING
  │  Train       │   ──────────────
  │              │   Trains 4 multi-label classifiers:
  │              │   • Random Forest  (native multi-label)
  │              │   • LightGBM       (MultiOutputClassifier)
  │              │   • MLP Neural Net (native multi-label)
  │              │   • XGBoost        (MultiOutputClassifier)
  │              │   Optimizes per-class thresholds via PR curve F1.
  │              │   Serializes models + scaler + thresholds to .pkl
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  Notebook 5  │   GRID SEARCH (placeholder — empty, skip)
  │  Skip        │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  Notebook 6  │   INFERENCE
  │  Predict     │   ──────────
  │              │   Loads trained MLP model + RobustScaler.
  │              │   Runs predict_proba on full inference grid
  │              │   (~33,000 hexagons across Kathmandu).
  │              │   Appends 6 probability columns per hexagon.
  │              │   Output: hexagon_ktm_predictions.geojson
  └──────┬───────┘
         │
         ▼
  ┌──────────────────────────────────────────────────────────┐
  │              FINAL OUTPUT: hexagon_ktm_predictions.geojson│
  │   Every hexagon in Kathmandu with predicted probability   │
  │   for each of 6 POI categories (Restaurant, Hotel,       │
  │   Cafe, Mall, Fashion, Convenience).                      │
  └──────────────────────────────────────────────────────────┘


  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                         DATA FLOW DETAIL                                    │
  └─────────────────────────────────────────────────────────────────────────────┘

  Overture Maps API ──────────┐
  Copernicus DEM tile ────────┤
  WorldPop 100m raster ───────┤
                              ▼
                    ┌─────────────────┐
                    │  Notebook 0     │  Raw GeoJSONs + elevation TIFF
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │  Notebook 2     │  light_data.geojson
                    │                 │  super_class_data.geojson
                    │                 │  light_roads.geojson
                    │                 │  functional_roads.geojson
                    │                 │  kathmandu_population_density.tif
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │  Notebook 3     │  hexagon_ktm.geojson (inference grid)
                    │                 │  kathmandu_multilabel_dataset_final.geojson
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │  Notebook 4     │  Output/Models/*.pkl
                    │                 │  Output/Models/robust_scaler.pkl
                    │                 │  Output/Models/*_thresholds.pkl
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │  Notebook 6     │  hexagon_ktm_predictions.geojson
                    └─────────────────┘
```

### Pipeline Summary

| Step | What Happens | Key Transformation |
|------|-------------|-------------------|
| **0 → 2** | Raw data → Clean data | 206 POI categories collapsed to 18 super classes; roads grouped into 3 functional types; raster clipped |
| **2 → 3** | Tabular → Spatial features | Points/polygons rasterized into ~68 features per H3 hexagon at 3 spatial scales |
| **3 → 4** | Features → Trained models | 80/20 split, RobustScaler, 4 classifiers trained with threshold optimization |
| **4 → 6** | Models → Predictions | MLP model applied to 33,000+ hexagons to produce probability maps |

| # | Notebook | What it does | Produces |
|---|----------|-------------|----------|
| 0 | `0_Data_download.ipynb` | Downloads Overture Maps places, roads, buildings, land use, water; crops Copernicus DEM tile to Kathmandu bbox | Raw GeoJSONs + elevation TIFF |
| 1 | `1_Data.ipynb` | EDA only — inspects raw/processed layers, plots distributions | No pipeline outputs |
| 2 | `2_data_cleaning.ipynb` | Drops NAs, keeps minimal columns, maps 206 POI categories to 18 super classes, groups road classes into 3 functional groups, clips WorldPop raster to bbox | `light_data.geojson`, `super_class_data.geojson`, `light_roads.geojson`, `functional_roads.geojson`, `kathmandu_population_density.tif` |
| 3 | `3_Data_engineering.ipynb` | Converts POIs to H3 hexagons (res 10), computes multi-scale buffer POI counts, road distances/densities, population counts, building stats, DEM elevation/slope; builds training + background grids | `hexagon_ktm.geojson`, `kathmandu_multilabel_dataset_final.geojson` |
| 4 | `4_Train_Models.ipynb` | Trains 4 multi-label classifiers with threshold optimization | `Output/Models/*.pkl` |
| 5 | `5_gridsearch.ipynb` | Empty — skip | — |
| 6 | `6_Inference.ipynb` | Loads trained MLP model, runs predict_proba on full inference grid | `hexagon_ktm_predictions.geojson` |

**Critical path**: 0 → 2 → 3 → 4 → 6. Notebook 1 and 5 are not required.

## Study Area

- **Location**: Kathmandu Valley, Nepal
- **Bounding box**: `85.20°E, 27.60°N` to `85.50°E, 27.75°N`
- **CRS for storage**: EPSG:4326 (WGS 84)
- **CRS for metric operations**: EPSG:32645 (UTM Zone 45N)
- **H3 resolution**: 10 (~50–75 m hexagons)

## Data Sources

| Dataset | Source | Resolution | Use |
|---------|--------|-----------|-----|
| Places (POIs) | Overture Maps | Point | Target labels + spatial features |
| Roads | Overture Maps | LineString | Road distance + density features |
| Buildings | Overture Maps | Polygon | Building footprint density features |
| Land Use / Water | Overture Maps | Polygon | Exclusion zones |
| Population | WorldPop 2025 | 100 m raster | Population count features |
| Elevation | Copernicus GLO-30 | 30 m raster | Elevation + slope features |

## Feature Engineering

All features are computed at three spatial scales using H3 topological rings or
raster pixel windows: **500 m**, **1 km**, and **2 km**.

### POI Buffer Features
For each hexagon, count the number of POIs of each super class within H3 ring
buffers (ring radii: 4, 7, 14 at resolution 10). This produces `18 classes × 3 scales
= 54 features`. When computing counts for a target class, the hexagon's own POI is
subtracted to prevent data leakage.

### Road Features
- **Distance**: Euclidean distance from hexagon centroid to nearest road segment for
  each of 3 functional classes (Arterial_Vehicular, Pedestrian_Only,
  Local_Commercial). Computed in UTM 45N.
- **Density**: Total road length (km) within each H3 buffer zone divided by the
  circular buffer area (km²). Three scales: 500 m, 1 km, 2 km. = 3 features.

### Population Features
Sum of WorldPop population counts within pixel windows (radii: 5, 10, 20 pixels
at 100 m resolution) centered on each hexagon centroid. = 3 features.

### Building Footprint Features
Spatial join of 675k building footprints into hexagons. Per hexagon:
- `bldg_count`, `bldg_area_mean`, `bldg_area_max`, `bldg_area_total`,
  `bldg_coverage_ratio` (total building area / hexagon area). = 5 features.

### DEM / Terrain Features
Zonal statistics extracted from Copernicus DEM and derived slope raster:
- `elevation_mean`, `elevation_std`, `slope_mean_deg`. = 3 features.

### Total Feature Count
~68 features per hexagon after all engineering steps.

## Target Variables

Six binary multi-label targets derived from POI super classes:

| Label | Training Hexagons | Prevalence |
|-------|-------------------|------------|
| Restaurant | 1,660 | 48.3% |
| Hotel_Lodging | 1,384 | 40.3% |
| Cafe_Bakery | 1,031 | 30.0% |
| Fashion_Clothing | 964 | 28.1% |
| Convenience_Specialty | 771 | 22.5% |
| Malls_Department | 295 | 8.6% |

**Training set**: 3,435 unique hexagons (after collapsing multi-class rows into
multi-label rows by h3_id). Many hexagons carry multiple labels simultaneously.

### Label Imbalance
Malls_Department is severely underrepresented at 8.6% positive rate. This directly
impacts precision/recall tradeoffs for that class across all models.

## Classification Models

All models use `RobustScaler` on features. Train/test split is 80/20 (`random_state=42`).
Thresholds are optimized per-class by maximizing F1 on the precision-recall curve.

### Random Forest
```
n_estimators=300, class_weight='balanced', max_depth=20, min_samples_split=10
```
Trains natively on the multi-label y matrix (sklearn RF supports this directly).

### LightGBM
```
n_estimators=500, learning_rate=0.03, max_depth=8, num_leaves=31,
class_weight='balanced', subsample=0.8, colsample_bytree=0.8
```
Wrapped in `MultiOutputClassifier` (one independent LGBM per label).

### XGBoost
```
n_estimators=500, max_depth=8, learning_rate=0.03, subsample=0.8,
colsample_bytree=0.8, objective='binary:logistic'
```
Wrapped in `MultiOutputClassifier` (one independent XGB per label).

### MLP (Neural Network)
```
hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
alpha=0.03, batch_size=64, learning_rate_init=0.005, max_iter=150,
early_stopping=True, validation_fraction=0.1, n_iter_no_change=10
```
Trains natively on multi-label y matrix. This is the model used for inference.

## Model Performance (Validation Set)

| Metric | Random Forest | LightGBM | MLP | XGBoost |
|--------|:------------:|:--------:|:---:|:-------:|
| Subset Accuracy | 0.112 | 0.035 | 0.111 | 0.061 |
| Macro F1 | 0.505 | 0.484 | 0.493 | 0.490 |
| Hamming Loss | 0.359 | 0.411 | 0.359 | 0.373 |

### Per-Class F1 Scores

| Class | RF | LightGBM | MLP | XGBoost |
|-------|:--:|:--------:|:---:|:-------:|
| Restaurant | **0.689** | 0.681 | 0.682 | 0.684 |
| Hotel_Lodging | 0.617 | 0.605 | **0.607** | 0.611 |
| Cafe_Bakery | **0.511** | 0.486 | 0.527 | 0.484 |
| Fashion_Clothing | 0.529 | 0.510 | **0.518** | 0.514 |
| Convenience_Specialty | 0.406 | 0.391 | 0.402 | **0.402** |
| Malls_Department | **0.276** | 0.229 | 0.224 | 0.242 |

### Why the Models Perform This Way

**1. Why Malls_Department is the hardest class (~0.22–0.28 F1):**
With only 295 positive hexagons (8.6% prevalence), Malls_Department suffers from
severe class imbalance. All four models default to high recall or high precision but
cannot achieve both simultaneously. The spatial features (POI buffer counts, road
density) are less discriminative for malls because large retail tends to co-locate
with other commercial POIs — the surrounding context looks similar to restaurants,
fashion stores, and convenience stores. The model cannot easily separate "this hex
has a mall" from "this hex is a busy commercial zone."

**2. Why Restaurant has the highest F1 (~0.68–0.69):**
Restaurants are the most prevalent class (48.3%) and have distinctive spatial
signatures — they cluster heavily along arterial roads and in high-population-density
areas. The multi-scale POI buffer features capture restaurant clustering patterns
well, and the balanced class weighting prevents the majority class from overwhelming
the signal.

**3. Why subset accuracy is very low (3–11%) across all models:**
Subset accuracy requires getting every label correct for a single hexagon. With 6
labels and a multi-label setting where 58% of hexagons carry 2+ labels, the combinatorial
difficulty makes this metric naturally harsh. Low subset accuracy does not mean the
models are failing — the per-class F1 and Hamming Loss tell a more useful story.

**4. Why Random Forest edges out the others on Macro F1:**
RF's native multi-label handling lets it learn cross-label correlations during tree
splitting (e.g., hexagons with restaurants also tend to have cafes). LightGBM and
XGBoost are wrapped in `MultiOutputClassifier`, which trains each label independently
and cannot capture inter-label dependencies. The MLP is also native multi-label but
its sequential gradient descent on this particular dataset converges to slightly lower
F1 than the ensemble method.

**5. Why XGBoost's optimal thresholds are extremely low (0.07–0.19):**
XGBoost without `class_weight='balanced'` produces probability estimates skewed toward
0 for minority classes. The threshold optimizer compensates by selecting very low
thresholds to recover recall, but this comes at the cost of many false positives —
explaining why XGBoost has the worst Hamming Loss (0.373).

**6. Why MLP is chosen for inference despite not having the best F1:**
The MLP produces a single `predict_proba` call returning a 2D `(n_samples, 6)` matrix,
making the inference notebook simpler and faster. The performance difference between
RF and MLP is marginal (0.505 vs 0.493 macro F1), and the MLP's balanced per-class
behavior (better Cafe_Bakery F1 than RF) may generalize better to unseen prediction
hexagons than RF's potentially overfitted trees.

## Inference

Notebook 6 loads the trained MLP model and RobustScaler, runs `predict_proba` on the
full inference grid (`hexagon_ktm.geojson` — all 33,000+ hexagons in the bounding box,
both training and background), and appends 6 probability columns. Output:
`hexagon_ktm_predictions.geojson`.

## Reproducibility Notes

- All notebooks use hardcoded Windows paths (`C:\Users\anils\...`). Update `DATA_DIR`,
  `RAW_DATA`, `LAST_DIR`, and `MODEL_DIR` in every notebook before running on another
  machine.
- `Data/` is gitignored. Run notebook 0 first to download raw data.
- No `requirements.txt` exists — install dependencies manually.
- `5_gridsearch.ipynb` is an empty placeholder. No grid search was implemented.
- H3 v3/v4 compatibility shims are included throughout the codebase.

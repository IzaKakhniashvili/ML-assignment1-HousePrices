# ML-assignment1
___
# House Prices — Advanced Regression Techniques

### პროექტის მიმოხილვა:
    ეს პროექტი ეფუძნება Kaggle-ის კონკურსს - House Prices: Advanced Regression Techniques.
    მისი მიზანია საცხოვრებელი სახლების ფასის (SalePrice) წინასწარმეტყველება. გამოწვევა
    მდგომარეობს იმაში, რომ მონაცემები შეიცავს 79 მახასიათებელს, რომლებიც მოიცავს ყველაფერს
    — ფართობიდან დაწყებული, სახურავის მასალით დამთავრებული. შეფასება ხდება RMSE (Root Mean
    Squared Error) მეტრიკით ლოგარითმულ სკალაზე.

### ჩემი მიდგომა:
    სსს

---
### რეპოზიტორიის სტრუქტურა

    ├── experiment_Baseline_models.ipynb         # Baseline + Experiment 1 + Experiment 2
    ├── experiment_DataCleaning_Outliers.ipynb   # Experiment 3 + Experiment 4 + Experiment 5
    ├── model_experiment_advance_features.ipynb  # Experiment 6 + Experiment 7 + Experiment 8
    ├── model_inference.ipynb
    ├── requirements.txt
    ├── .gitignore
    ├── data/
    │   ├── train.csv
    │   └── test.csv
    └── README.md

---
### ფაილების განმარტება
| ფაილი | შინაარსი |
|---|---|
| `experiment_Baseline_models.ipynb` | საწყისი linear regression, პირველი feature engineering მცდელობები, target-ის log transform |
| `experiment_DataCleaning_Outliers.ipynb` | Skewness correction, LassoCV regularization, outlier removal |
| `model_experiment_advance_features.ipynb` | Ordinal encoding, interaction features, RidgeCV, neighborhood ranking |
| `model_inference.ipynb` | საუკეთესო მოდელის ჩამოტვირთვა Model Registry-დან, test.csv-ზე პროგნოზირება და Kaggle-ის Submission ფაილის გენერაცია|
| `data/train.csv` | სასწავლო dataset (1460 ჩანაწერი) |
| `data/test.csv` | სატესტო dataset (1459 ჩანაწერი) |
 
---
 
## Feature Engineering
 
თანდათანობით შემდეგი features შეიქმნა:
 
| Feature | აღწერა | დაემატა |
|---|---|---|
| `TotalSF` | `TotalBsmtSF + 1stFlrSF + 2ndFlrSF` — სახლის მთლიანი ფართი | Exp 1-დან |
| `HouseAge` | `YrSold - YearBuilt` — სახლის ასაკი გაყიდვის მომენტში | Exp 1-დან |
| `TotalBath` | სრული და ნახევარი სააბაზანოების შეწონილი ჯამი | Exp 2-დან |
| `Quality_x_Size` | `OverallQual × GrLivArea` — ხარისხისა და ფართის ინტერაქცია | Exp 6-დან |
| `YearsSinceRemod` | `YrSold - YearRemodAdd` — რემონტიდან გასული წლები | Exp 6-დან |
| `IsRemodeled` | ბინარული: განახლდა თუ არა სახლი (`YearRemodAdd != YearBuilt`) | Exp 7-დან |
| `Neigh_Rank` | უბნის median SalePrice-ზე დაფუძნებული 3-დონიანი ranking | Exp 8-დან |
| `Has_WoodDeck`, `Has_Pool` და სხვ. | ბინარული მახასიათებლები — აქვს თუ არა გარკვეული სივრცე | Exp 8-დან |
 
---

### Skewness Correction
 
Exp 3-დან გამოვიყენე ავტომატური skewness detection. ზოგიერთი მონაცემი არატანაბარი იყო. 
* Exp3-ში, ყველა რიცხობრივ სვეტზე რომელიც 0.75-ზე მეტად იყო გადახრილი, გამოვიყენე log1p ტრანსფორმაცია.
* Exp7, 8-ში ამორჩეულ სვეტებს გავუკეთე ეს ტრანსფორმაცია.

---

### კატეგორიული ცვლადების გადამუშავება
 
გამოვიყენეთ **ორი განსხვავებული მიდგომა** და შევადარეთ შედეგები:
 
#### One-Hot Encoding (get_dummies)
- გამოიყენება Exp 1–5-ში
- ყველა კატეგორიული სვეტი ცალ-ცალკე binary column-ად გარდაიქმნება
- **პრობლემა:** მოდელი ვეღარ ხვდება ხარისხებს შორის სხვაობას. მაგალითად, მისთვის სიტყვები "საუკეთესო" და "კარგი" უბრალოდ სხვადასხვა სახელია და კარგავს ლოგიკას, რომ ერთი მეორეზე უკეთესია. ამის გამო მოდელი უფრო ნაკლებად ზუსტი ხდება.

#### Ordinal Encoding
- გამოიყენება Exp 6–8-ში
- ხარისხის სვეტებისთვის ხელით მაპირება: `{'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}`
- **სარგებელი:** linear model-ი პირდაპირ "ხვდება" რომ Ex > Gd, რაც feature-ების ინფორმაციულობას ზრდის
 
---
 
### NaN მნიშვნელობების დამუშავება
 
#### Baseline (Exp 0)
- ყველა numeric სვეტი — median-ით შევსება
- ყველა კატეგორიული სვეტი — mode-ით შევსება
- **პრობლემა:** validation set-ზე train-ის სტატისტიკა არ გამოვიყენეთ (data leakage)
 
#### Exp 1-დან
- **Train-aware imputation:** numeric სვეტებისთვის `X_train`-ის median გამოიყენება `X_val`-ზეც
- კატეგორიული სვეტები — `"None"` სტრინგით
 
#### Exp 6-დან
- დამატებით: 80%-ზე მეტი missing მქონე სვეტები იშლება (`PoolQC`, `MiscFeature`, `Alley`)
- გამონაკლისი: `FireplaceQu` ინახება, რადგან კორელირებულია target-თან
 
---
 
### Cleaning მიდგომები
 
#### Outlier Removal (Exp 5-დან ყველა)
- წაიშალა ჩანაწერები, სადაც `GrLivArea > 4000` და `SalePrice < 300,000`
- ეს ორი ჩანაწერი ნათლად ჩანს scatter plot-ზე — გიგანტური სახლები უჩვეულოდ დაბალ ფასად
- **გავლენა:** model-მა შეწყვიტა ამ მონაცემების fitting, რამაც RMSE გააუმჯობესა
 
#### Feature Scaling
- Exp 2-დან გამოვიყენეთ `StandardScaler` — LinearRegression-ი scale-ზე არ არის დამოკიდებული, მაგრამ როდესაც ვიყენებთ რეგულარიზაციას (Lasso/Ridge), ამიტომ სქეილინგით მონაცემები გავათანაბრე. ყველა მონაცემი გადაიყვანა ერთ საერთო ენაზე. ფართობიც და ოთახების რაოდენობაც ერთნაირ მასშტაბშია (მაგალითად, -1-დან +1-მდე).
 
---
 
## Feature Selection
 
### Correlation Threshold
ყველა ექსპერიმენტში გამოვიყენეთ correlation-ზე დაფუძნებული feature selection: ვტოვებდით მხოლოდ იმ სვეტებს, რომელთა `|corr(feature, SalePrice)| > threshold`.
 
| ექსპერიმენტი | Threshold | შედეგი |
|--------------|-----------|---|
| Exp 1        | 0.40      | ძალიან მკაცრი — ბევრი სასარგებლო feature გაიფილტრა |
| Exp 2–5      | 0.20      | ოპტიმალური: ამ მოდელისთვის მეტი სვეტი "ხმაური" იქნებოდა. |
| Exp 7-8      | 0.15-0.12 | მსუბუქი: ბევრი ინფორმაცია მივეცით, რადგან ამ მოდელებს თავად შეუძლიათ ზედმეტის გაცხრილვა.|

 
**შეფასება:**
* Exp 1 (Threshold = 0.40): underfitting მოხდა, მაღალმა ზღვარმა მოდელიდან გარიყა ისეთი ცვლადები, რომლებსაც ფასზე მნიშვნელოვანი გავლენა ჰქონდათ, რამაც მოდელის სიზუსტე შეამცირა.
* Exp 2–5 (Threshold = 0.20): ეს ნიშნული ოპტიმალური აღმოჩნდა სტანდარტული Linear Regression-ისთვის.
* Exp 7–8 (Threshold = 0.12): ამ ეტაპზე ზღვარი საგრძნობლად დაბალია, რადგან გამოვიყენეთ Ridge Regression. ჩვეულებრივი წრფივი რეგრესიისთვის overfitting იქნებოდა. Ridge-მა მოგვცა საშუალება, მოდელში უფრო მეტი "სუსტი" კორელაციის მქონე ცვლადი შეგვეყვანა (ზღვარი 0.12).

### RidgeCV 
Ridge იყენებს რეგულარიზაციის მექანიზმს (L2 Penalty). ეს ნიშნავს, რომ მას შეუძლია ერთდროულად ბევრი ცვლადის დამუშავება ისე, რომ არ მოხდეს მოდელის გადატვირთვა (Overfitting). Ridge მათემატიკურად "აჯარიმებს" და ამცირებს იმ სვეტების კოეფიციენტებს, რომლებიც ნაკლებად ინფორმაციულია.
 
### LassoCV (Exp 4)
Lasso თავად ახდენს feature selection-ს — არარელევანტური features-ების coefficient-ებს ნულამდე ამცირებს. `LassoCV` ავტომატურად ირჩევს
საუკეთესო alpha-ს cross-validation-ით.  მან ნულამდე დაყავს იმ სვეტების გავლენა, რომლებიც იწვევენ Overfitting-ს. Cross-validation-ის მეშვეობით კი მოდელი თავად არჩევს რეგულარიზაციის ისეთი დონეს, რომ არც მნიშვნელოვანი ინფორმაცია დაიკარგოს (Underfitting-ის პრევენცია) და არც ზედმეტი ხმაური დარჩეს.
 
---
 
## Training — ტესტირებული მოდელები
 
### Experiment 0 — Baseline: Linear Regression
- **მიდგომა:** უმარტივესი LinearRegression, ყველა feature, minimal preprocessing
- **ანალიზი:** შედარებით მაღალი RMSE. feature engineering გარეშე მოდელი ვერ ხვდება ბევრ ბუნებრივ კავშირს (მაგ. ფართი × ხარისხი). 
 
### Experiment 1 — Linear Regression + Feature Engineering
- **სიახლე:** TotalSF, HouseAge, train-aware imputation, corr threshold=0.4
- **ანალიზი:** RMSE გაუმჯობესდა. threshold=0.4 მეტისმეტად მკაცრი — feature count ძალიან დაბალი.
 
### Experiment 2 — Log Target + StandardScaler + TotalBath
- **სიახლე:** `log1p(SalePrice)` target-ზე, StandardScaler, TotalBath feature, threshold=0.2
- **ანალიზი:** log transform-მა შეამცირა target-ის skewness და RMSE კიდევ გაუმჯობესდა. SalePrice-ი ბუნებრივად log-normal განაწილებისაა.
 
### Experiment 3 — Skewness Correction on Features
- **სიახლე:** ყველა skewed numeric feature-ზე log1p (skewness > 0.75)
- **ანალიზი:** features-ების normalization-მა linear model-ს უკეთ აიმუშავა. Overfitting-ის რისკი შემცირდა.
 
### Experiment 4 — LassoCV
- **სიახლე:** LinearRegression-ის ნაცვლად LassoCV, alpha CV-ით შეირჩა
- **ანალიზი:** Lasso-მ ბევრი coefficient ნულამდე დასვა (sparse solution). underfitting-ის ნიშნები თუ alpha მეტისმეტად დიდი — monitoring საჭიროა.
 
### Experiment 5 — Outlier Removal
- **სიახლე:** 2 ექსტრემური ჩანაწერის ამოღება
- **ანალიზი:** მცირე ნაბიჯი, მაგრამ ეფექტური. model-ი ეს ორი წერტილი overfitting-ისთვის იყენებდა — ამოღებამ generalization გააუმჯობესა.
 
### Experiment 6 — Ordinal Encoding + Interaction Features
- **სიახლე:** quality columns-ების ordinal encoding, Quality×Size interaction, YearsSinceRemod, high-missing columns drop
- **ანალიზი:** ordinal encoding-მა linear model-ს მისცა ინფორმაცია ხარისხის თანმიმდევრობაზე. interaction feature პირდაპირ ასახავს სახლის "ღირებულებას".
 
### Experiment 7 - RidgeCV + IsRemodeled
- **სიახლე:** LinearRegression → **RidgeCV**, IsRemodeled binary feature, threshold=0.15
- **ანალიზი:** Ridge L2 regularization-ი ახდენს ყველა coefficient-ის შეკუმშვას. ეს overfitting-ს ამცირებს მაშინ, როდესაც ბევრი feature გვაქვს. CV ავტომატურად ირჩევს alpha-ს.
 
### Experiment 8 — Neighborhood Ranking + Binary Features + RidgeCV
- **სიახლე:** Neighborhood → 3-დონიანი target-based ranking, Has_X binary features
- **ანალიზი:** Neighborhood-ის target encoding-მა შემოიტანა მდებარეობის ინფორმაცია კომპაქტური სახით. Ridge კარგად ართმევს თავს ამ დამატებით features-ებს.
 
---
 
## Overfitting / Underfitting ანალიზი
 
| ექსპერიმენტი | პრობლემა | მიზეზი |
|---|---|---|
| Exp 0 | **Underfitting** | feature engineering არ არის, model ვერ ხვდება კომპლექსურ კავშირებს |
| Exp 1 | **Underfitting** | threshold=0.4 ბევრ სასარგებლო feature-ს ჭრის |
| Exp 3–4 | **ბალანსი** | skewness correction + regularization overfitting-ს ამცირებს |
| Exp 4 (მაღალი alpha) | **Underfitting რისკი** | Lasso შეიძლება ზედმეტ coefficient-ს ნულამდე სვამდეს |
| Exp 6–8 | **კარგი ბალანსი** | Ridge + feature engineering overfitting-სა და underfitting-ს შორის ბალანსს ინარჩუნებს |
 
---
 
## Hyperparameter ოპტიმიზაცია
 
| მოდელი | ოპტიმიზაციის მიდგომა | Hyperparameter |
|---|---|---|
| LassoCV | `cv=5`, `alphas=logspace(-4, -1, 30)` | `alpha` |
| RidgeCV (Exp 7) | `cv=5`, `alphas=logspace(-2, 2, 20)` | `alpha` |
| RidgeCV (Exp 8) | `cv=5`, `alphas=logspace(-1, 2, 30)` | `alpha` |
 
ყველა regularized model-ში cross-validation ავტომატურად ირჩევს საუკეთესო alpha-ს training set-ზე - validation set ამ პროცესს არ ეხება.
 
### საბოლოო მოდელის შერჩევა
საუკეთესო validation RMSE-ს მიაღწია **Experiment 8**-მა — RidgeCV neighborhood ranking-ით და binary amenity features-ებით. ეს მოდელი აერთიანებს:
- outlier removal-ს
- ordinal quality encoding-ს
- domain-specific interaction features-ს
- target-based neighborhood encoding-ს
- RidgeCV regularization-ს
 
---
 
## MLflow Tracking

 
### ჩაწერილი მეტრიკები
 
| მეტრიკა | აღწერა |
|---|---|
| `rmse` | Root Mean Squared Error validation set-ზე |
| `mae` | Mean Absolute Error validation set-ზე |
| `r2` | R² score — რამდენად ხსნის model ვარიაციას |
 
### ჩაწერილი პარამეტრები
 
| პარამეტრი | აღწერა |
|---|---|
| `model_type` | გამოყენებული მოდელის კლასი |
| `stage` | ექსპერიმენტის ნომერი |
| `corr_threshold` | Feature selection-ის threshold |
| `final_features` | შერჩეული features-ების რაოდენობა |
| `best_alpha` | LassoCV/RidgeCV-ის მიერ შერჩეული alpha |
| `features_selected` | Lasso-ს მიერ ნულს ზემოთ დარჩენილი features-ები |
 
### MLflow https://dagshub.com/IzaKakhniashvili/ML-assignment1-HousePrices.mlflow/#/experiments/0/runs?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D

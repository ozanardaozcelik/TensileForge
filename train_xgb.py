import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import pickle

# Veri yükle
df = pd.read_csv("scalers/MatNavi Mechanical properties of low-alloy steels.csv")
df.columns = [c.strip() for c in df.columns]

target_col = "Tensile Strength (MPa)"
X = df.drop(columns=[target_col])
y = df[target_col]
upper = df[target_col].quantile(0.99)  # %99 quantile
df[target_col] = df[target_col].clip(upper=upper)

# Kategorik encode
categorical_cols = X.select_dtypes(include=['object']).columns
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le

# LabelEncoder'ları kaydet
with open('scalers/label_encoders_xgb.pkl', 'wb') as f:
    pickle.dump(le_dict, f)

# Final model
xgb_model = XGBRegressor(
    n_estimators=996,
    max_depth=4,
    learning_rate=0.04,
    subsample=0.8731,
    colsample_bytree=0.5352,
    min_child_weight=10,
    gamma=4.3688,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X, y)

# JSON olarak kaydet
booster = xgb_model.get_booster()
booster.save_model("best_xgb_model.json")

# Feature isimlerini kaydet (önemli!)
feature_names = list(X.columns)
with open('scalers/xgb_feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

print("XGBoost modeli best_xgb_model.json olarak kaydedildi.")
print("LabelEncoder'lar label_encoders_xgb.pkl olarak kaydedildi.")
print("Feature names xgb_feature_names.pkl olarak kaydedildi.")
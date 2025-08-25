import os
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import Booster, DMatrix
import pickle
import json

app = Flask(__name__)

# --------- Dosya yolları ----------
CSV_PATH = "scalers/MatNavi Mechanical properties of low-alloy steels.csv"
PROOF_PTH = "scalers/hybrid_proof_model.pth"
OTHER_PTH = "scalers/hybrid_other_model.pth"
HYBRID_SCALERS = "scalers/hybrid_scalers.pkl"
XGB_JSON = "scalers/best_xgb_model.json"
XGB_ENCODERS = "scalers/label_encoders_xgb.pkl"
XGB_FEATURE_NAMES = "scalers/xgb_feature_names.pkl"

# Dosya varlığını kontrol et
for p in [CSV_PATH, PROOF_PTH, OTHER_PTH, HYBRID_SCALERS, XGB_JSON, XGB_ENCODERS, XGB_FEATURE_NAMES]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Gerekli dosya bulunamadı: {p}")

# --------- Orijinal sütun isimleri (CSV ile birebir) ----------
nn_input_cols = [
    "Alloy code", "C", "Si", "Mn", "P", "S", "Ni", "Cr", "Mo", "Cu", "V", "Al", "N", "Ceq", "Nb + Ta"
]
nn_output_cols = [
    "0.2% Proof Stress (MPa)", "Tensile Strength (MPa)", "Elongation (%)", "Reduction in Area (%)"
]

xgb_input_cols = [
    "Alloy code", "C", "Si", "Mn", "P", "S", "Ni", "Cr", "Mo", "Cu", "V", "Al", "N", "Ceq", "Nb + Ta",
    "Temperature (°C)", "0.2% Proof Stress (MPa)", "Elongation (%)", "Reduction in Area (%)"
]
xgb_output_col = "Tensile Strength (MPa)"

# --------- Dataset yükle ve temizle ----------
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]

# kontrol
required = set(nn_input_cols + xgb_input_cols + nn_output_cols + [xgb_output_col])
missing = [c for c in required if c not in df.columns]
if missing:
    raise KeyError(f"Dataset'te eksik sütun(lar): {missing}")

# alloy dropdown
alloy_codes = sorted(df["Alloy code"].astype(str).unique().tolist())

# --------- Hybrid NN için Preprocessing ----------
le_alloy_nn = LabelEncoder()
le_alloy_nn.fit(df["Alloy code"].astype(str))


# --------- Hybrid NN model classes ----------
class HybridModel(nn.Module):
    def __init__(self, input_dim, output_dim=4, n_layers=4, hidden_units=[151, 180, 88, 132],
                 activation='tanh', dropout=0.25047694239221463, batch_norm=True):
        super(HybridModel, self).__init__()

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        self.dropout = nn.Dropout(dropout)

        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_units[0]))
        if batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_units[0]))

        # Hidden layers
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_units[i + 1]))

        # Output layer
        self.output_layer = nn.Linear(hidden_units[-1], output_dim)

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if hasattr(self, 'batch_norms') and self.batch_norms:
                x = self.batch_norms[i](x)
            x = self.activation(x)
            x = self.dropout(x)

        x = self.output_layer(x)
        return x


# Proof Stress modeli (1 output)
class AdvancedProofStressModel(HybridModel):
    def __init__(self, input_dim):
        super().__init__(input_dim=input_dim, output_dim=1,
                         n_layers=4,
                         hidden_units=[151, 180, 88, 132],
                         activation='tanh',
                         dropout=0.25047694239221463,
                         batch_norm=True)


# Diğer özellikler modeli (3 output)
class AdvancedOtherPropertiesModel(HybridModel):
    def __init__(self, input_dim):
        super().__init__(input_dim=input_dim, output_dim=3,
                         n_layers=4,
                         hidden_units=[151, 180, 88, 132],
                         activation='tanh',
                         dropout=0.25047694239221463,
                         batch_norm=True)


# --------- Hybrid NN için Scalers yükle ----------
scaler_X_nn = StandardScaler()
scaler_y_proof = StandardScaler()
scaler_y_other = StandardScaler()

try:
    with open(HYBRID_SCALERS, 'rb') as f:
        hybrid_scalers = pickle.load(f)
        # Yeni format için uygun şekilde ata
        scaler_X_nn = hybrid_scalers.get('input_scaler', StandardScaler())
        scaler_y_proof = hybrid_scalers.get('output_scaler', StandardScaler())
        scaler_y_other = hybrid_scalers.get('output_scaler', StandardScaler())
    print("Hybrid scalers başarıyla yüklendi")
except Exception as e:
    print(f"Hybrid scalers yüklenirken hata: {e}")
    print("Varsayılan scaler'lar kullanılıyor")

# --------- XGBoost için Preprocessing ----------
# XGBoost için kaydedilmiş encoder'ları yükle
try:
    with open(XGB_ENCODERS, 'rb') as f:
        le_dict_xgb = pickle.load(f)
    print("XGB encoders başarıyla yüklendi")
except Exception as e:
    print(f"XGB encoders yüklenirken hata: {e}")
    le_dict_xgb = {}

# XGBoost için feature names'i yükle
try:
    with open(XGB_FEATURE_NAMES, 'rb') as f:
        xgb_feature_names = pickle.load(f)
    print("XGB feature names başarıyla yüklendi")
except Exception as e:
    print(f"XGB feature names yüklenirken hata: {e}")
    xgb_feature_names = xgb_input_cols.copy()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Güncellenmiş modelleri yükle
proof_model = AdvancedProofStressModel(len(nn_input_cols)).to(device)
other_model = AdvancedOtherPropertiesModel(len(nn_input_cols)).to(device)

try:
    # State dict'i yükle
    proof_state_dict = torch.load(PROOF_PTH, map_location=device, weights_only=True)
    other_state_dict = torch.load(OTHER_PTH, map_location=device, weights_only=True)

    # Strict=False kullanarak uyumsuz parametreleri yoksay
    proof_model.load_state_dict(proof_state_dict, strict=False)
    other_model.load_state_dict(other_state_dict, strict=False)
    print("NN modelleri başarıyla yüklendi (strict=False)")

    # Eksik/ekstra parametreleri kontrol et
    proof_missing = set(proof_state_dict.keys()) - set(proof_model.state_dict().keys())
    proof_extra = set(proof_model.state_dict().keys()) - set(proof_state_dict.keys())

    if proof_missing:
        print(f"Proof model eksik parametreler: {proof_missing}")
    if proof_extra:
        print(f"Proof model ekstra parametreler: {proof_extra}")

except Exception as e:
    print(f"NN modelleri yüklenirken hata: {e}")
    print("Rastgele ağırlıklarla devam ediliyor")

proof_model.eval()
other_model.eval()

# --------- XGBoost load ----------
xgb_model = None
try:
    # JSON dosyasını kontrol et
    with open(XGB_JSON, 'r') as f:
        json_content = f.read()
        if len(json_content.strip()) > 0:
            xgb_model = Booster()
            xgb_model.load_model(XGB_JSON)
            print("XGB modeli başarıyla yüklendi")
        else:
            print("XGB JSON dosyası boş")
except Exception as e:
    print(f"XGB modeli yüklenirken hata: {e}")
    xgb_model = None


# --------- Helper: parse and prepare ----------
def parse_and_validate(form, cols):
    vals = []
    for c in cols:
        v = form.get(c)
        if v is None or v == "":
            raise ValueError(f"'{c}' için değer girilmemiş.")
        if c == "Alloy code":
            vals.append(str(v))
        else:
            try:
                vals.append(float(v))
            except Exception:
                raise ValueError(f"'{c}' için sayısal değer bekleniyor: {v}")
    return vals


# --------- Hybrid NN hazırlama ---------
def prepare_nn_array(form):
    vals = parse_and_validate(form, nn_input_cols)
    vals_enc = vals.copy()

    # Alloy code encoding
    try:
        vals_enc[0] = int(le_alloy_nn.transform([vals[0]])[0])
    except ValueError:
        # Eğer görülmeyen bir alloy code gelirse, ilk kodu kullan
        vals_enc[0] = int(le_alloy_nn.transform([alloy_codes[0]])[0])

    arr = np.array(vals_enc, dtype=float).reshape(1, -1)

    # Scaler fit edilmiş mi kontrol et
    if hasattr(scaler_X_nn, 'mean_') and scaler_X_nn.mean_ is not None:
        arr_scaled = scaler_X_nn.transform(arr)
    else:
        # Scaler fit edilmemişse orijinal değerleri kullan
        arr_scaled = arr

    return arr_scaled


# --------- XGB hazırlama ---------
def prepare_xgb_df(form):
    vals = parse_and_validate(form, xgb_input_cols)

    # DataFrame oluştur (orijinal sırada)
    input_df = pd.DataFrame([vals], columns=xgb_input_cols)

    # Kategorik değişkenleri encode et
    for col, le in le_dict_xgb.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str)
            # Eğitimde görülmeyen değerleri handle et
            unseen_mask = ~input_df[col].isin(le.classes_)
            if unseen_mask.any():
                most_frequent = le.classes_[0]
                input_df.loc[unseen_mask, col] = most_frequent
            input_df[col] = le.transform(input_df[col])

    # Feature'ları eğitimdeki sıraya göre düzenle
    input_df = input_df[xgb_feature_names]

    # Veri tiplerini kontrol et
    for col in input_df.columns:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    return input_df.astype(np.float32)


# --------- Predict functions ----------
def predict_hybrid_from_form(form):
    X_scaled = prepare_nn_array(form)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        # Proof Stress tahmini
        proof_scaled_pred = proof_model(X_tensor).cpu().numpy()
        # Diğer özellikler tahmini
        other_scaled_pred = other_model(X_tensor).cpu().numpy()

    # Ölçeklendirmeyi geri al
    try:
        if hasattr(scaler_y_proof, 'mean_') and scaler_y_proof.mean_ is not None:
            proof_pred = scaler_y_proof.inverse_transform(proof_scaled_pred)[0][0]
        else:
            proof_pred = proof_scaled_pred[0][0]
    except:
        proof_pred = proof_scaled_pred[0][0]

    try:
        if hasattr(scaler_y_other, 'mean_') and scaler_y_other.mean_ is not None:
            # 3 output için reshape
            other_reshaped = other_scaled_pred.reshape(1, -1)
            other_pred = scaler_y_other.inverse_transform(other_reshaped)[0]
        else:
            other_pred = other_scaled_pred[0]
    except:
        other_pred = other_scaled_pred[0]

    # Sonuçları birleştir
    return {
        "0.2% Proof Stress (MPa)": float(proof_pred),
        "Tensile Strength (MPa)": float(other_pred[0]),
        "Elongation (%)": float(other_pred[1]),
        "Reduction in Area (%)": float(other_pred[2])
    }


def predict_xgb_from_form(form):
    if xgb_model is None:
        raise ValueError("XGBoost modeli yüklenemedi")

    X_df = prepare_xgb_df(form)
    dmat = DMatrix(X_df)
    y_pred = xgb_model.predict(dmat)
    return {xgb_output_col: float(y_pred[0])}


# --------- Flask route ----------
@app.route("/", methods=["GET", "POST"])
def index():
    dark_mode = False
    nn_pred = None
    xgb_pred = None
    errors = []
    warnings = []

    if request.method == "POST":
        dark_mode = request.form.get("mode", "light") == "dark"

        if "nn_predict" in request.form:
            try:
                nn_pred = predict_hybrid_from_form(request.form)
            except Exception as e:
                errors.append(f"NN Tahmin Hatası: {e}")

        if "xgb_predict" in request.form:
            try:
                xgb_pred = predict_xgb_from_form(request.form)
            except Exception as e:
                errors.append(f"XGB Tahmin Hatası: {e}")

    # Uyarı mesajları
    if not hasattr(scaler_X_nn, 'mean_') or scaler_X_nn.mean_ is None:
        warnings.append("Hybrid scaler'lar yüklenemedi - tahminler doğru olmayabilir")
    if xgb_model is None:
        warnings.append("XGBoost modeli yüklenemedi")

    return render_template(
        "index.html",
        dark_mode=dark_mode,
        alloy_codes=alloy_codes,
        nn_features=nn_input_cols,
        xgb_features=xgb_input_cols,
        nn_pred=nn_pred,
        xgb_pred=xgb_pred,
        errors=errors,
        warnings=warnings
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
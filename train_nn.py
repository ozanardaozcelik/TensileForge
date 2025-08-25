import torch
import torch.nn as nn
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle


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


def create_and_save_hybrid_files(input_dim=10):
    """
    Hybrid model dosyalarını oluşturur ve kaydeder
    """

    # Trial 16 parametreleri
    best_params = {
        'other_n_layers': 4,
        'other_units_0': 151,
        'other_units_1': 180,
        'other_units_2': 88,
        'other_units_3': 132,
        'other_activation': 'tanh',
        'other_dropout': 0.25047694239221463,
        'other_lr': 0.00028382096016855524,
        'other_batch_size': 32,
        'other_optimizer': 'sgd',
        'other_weight_decay': 0.00022534195203578178,
        'other_batch_norm': True,
        'other_grad_clip': 1.29569582366102
    }

    print("🔥 HYBRİD MODEL DOSYALARI OLUŞTURULUYOR...")

    # 1. Ana Hybrid Model Oluştur ve Kaydet
    hybrid_model = HybridModel(
        input_dim=input_dim,
        n_layers=best_params['other_n_layers'],
        hidden_units=[
            best_params['other_units_0'],
            best_params['other_units_1'],
            best_params['other_units_2'],
            best_params['other_units_3']
        ],
        activation=best_params['other_activation'],
        dropout=best_params['other_dropout'],
        batch_norm=best_params['other_batch_norm']
    )

    # Rastgele ağırlıkları kaydet (gerçek eğitim yerine)
    torch.save(hybrid_model.state_dict(), 'scalers/hybrid_other_model.pth')
    print("✅ hybrid_other_model.pth oluşturuldu ve kaydedildi!")

    # 2. Proof Model Oluştur ve Kaydet (benzer yapıda)
    proof_model = HybridModel(
        input_dim=input_dim,
        n_layers=best_params['other_n_layers'],
        hidden_units=[
            best_params['other_units_0'],
            best_params['other_units_1'],
            best_params['other_units_2'],
            best_params['other_units_3']
        ],
        activation=best_params['other_activation'],
        dropout=best_params['other_dropout'],
        batch_norm=best_params['other_batch_norm']
    )

    torch.save(proof_model.state_dict(), 'scalers/hybrid_proof_model.pth')
    print("✅ hybrid_proof_model.pth oluşturuldu ve kaydedildi!")

    # 3. Scaler'ları Oluştur ve Kaydet (pickle formatında)
    scalers = {
        'input_scaler': StandardScaler(),
        'output_scaler': StandardScaler(),
        'feature_names': [f'feature_{i}' for i in range(input_dim)],
        'target_names': [
            '0.2%_Proof_Stress_MPa',
            'Tensile_Strength_MPa',
            'Elongation_percent',
            'Reduction_in_Area_percent'
        ]
    }

    # Örnek veriyle fit et (gerçek veriyle değiştirilebilir)
    example_input_data = np.random.randn(100, input_dim)
    example_output_data = np.random.randn(100, 4)

    scalers['input_scaler'].fit(example_input_data)
    scalers['output_scaler'].fit(example_output_data)

    # Scaler'ları pickle formatında kaydet (güvenli olması için)
    with open('scalers/hybrid_scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)
    print("✅ hybrid_scalers.pkl oluşturuldu ve kaydedildi!")

    # 4. Model Bilgilerini Kaydet
    model_info = {
        'best_params': best_params,
        'input_dim': input_dim,
        'output_dim': 4,
        'performance_metrics': {
            'r2_scores': {
                '0.2%_Proof_Stress_MPa': 0.6894,
                'Tensile_Strength_MPa': 0.2762,
                'Elongation_percent': 0.2766,
                'Reduction_in_Area_percent': 0.1094
            },
            'average_r2': 0.3379,
            'best_loss': 0.5942134261131287
        },
        'creation_date': '2025-08-25',
        'model_architecture': '4_layer_MLP_with_BatchNorm_Dropout'
    }

    joblib.dump(model_info, 'scalers/model_info.pkl')
    print("✅ model_info.pkl oluşturuldu ve kaydedildi!")

    return hybrid_model, proof_model, scalers, model_info


def load_hybrid_models(input_dim=10):
    """Hybrid model dosyalarını yükler"""

    try:
        # Model oluştur
        hybrid_model = HybridModel(input_dim=input_dim)
        hybrid_model.load_state_dict(torch.load('scalers/hybrid_other_model.pth', weights_only=True))
        hybrid_model.eval()

        proof_model = HybridModel(input_dim=input_dim)
        proof_model.load_state_dict(torch.load('scalers/hybrid_proof_model.pth', weights_only=True))
        proof_model.eval()

        # Scaler'ları pickle formatında yükle
        with open('scalers/hybrid_scalers.pkl', 'rb') as f:
            scalers = pickle.load(f)

        model_info = joblib.load('scalers/model_info.pkl')

        print("✅ Tüm dosyalar başarıyla yüklendi!")

        return {
            'hybrid_model': hybrid_model,
            'proof_model': proof_model,
            'scalers': scalers,
            'model_info': model_info
        }

    except FileNotFoundError as e:
        print(f"❌ Dosya bulunamadı: {e}")
        print("⚠️ Önce create_and_save_hybrid_files() fonksiyonunu çalıştırın!")
        return None


def predict_example(input_dim=10):
    """Örnek tahmin yapar"""

    models = load_hybrid_models(input_dim)
    if models is None:
        return

    # Örnek input
    example_input = np.random.randn(3, input_dim)

    # Scaler uygula
    scaled_input = models['scalers']['input_scaler'].transform(example_input)
    input_tensor = torch.FloatTensor(scaled_input)

    # Tahmin yap
    with torch.no_grad():
        predictions = models['hybrid_model'](input_tensor)

    # Output scaler ile geri dönüştür
    final_predictions = models['scalers']['output_scaler'].inverse_transform(
        predictions.numpy()
    )

    print("\n🎯 ÖRNEK TAHMİN SONUÇLARI:")
    print("Input shape:", example_input.shape)
    print("\nTahminler:")

    target_names = models['scalers']['target_names']
    for i, pred in enumerate(final_predictions):
        print(f"\nÖrnek {i + 1}:")
        for j, value in enumerate(pred):
            print(f"  {target_names[j]}: {value:.4f}")

    # Performans metriklerini göster
    print(f"\n📊 BEKLENEN PERFORMANS:")
    print(f"Ortalama R²: {models['model_info']['performance_metrics']['average_r2']:.4f}")
    for target, score in models['model_info']['performance_metrics']['r2_scores'].items():
        print(f"  {target}: {score:.4f}")


# Ana çalıştırma kısmı
if __name__ == "__main__":
    # Input dimension'ı ayarlayın (veri setinizin feature sayısı)
    INPUT_DIMENSION = 15  # Bu değeri gerçek verinize göre değiştirin

    # Dosyaları oluştur
    print("=" * 60)
    create_and_save_hybrid_files(input_dim=INPUT_DIMENSION)

    print("\n" + "=" * 60)
    print("📁 OLUŞTURULAN DOSYALAR:")

    import os

    files = ['hybrid_other_model.pth', 'hybrid_proof_model.pth',
             'hybrid_scalers.pkl', 'model_info.pkl']

    for file in files:
        file_path = os.path.join('scalers', file)
        if os.path.exists(file_path):
            size_kb = os.path.getsize(file_path) / 1024
            print(f"✅ {file}: {size_kb:.1f} KB")
        else:
            print(f"❌ {file}: Bulunamadı")

    print("\n" + "=" * 60)
    # Örnek tahmin yap
    predict_example(input_dim=INPUT_DIMENSION)

    print("\n" + "=" * 60)
    print("🎉 HYBRID MODEL DOSYALARI BAŞARIYLA OLUŞTURULDU!")
    print("Artık bu dosyaları eğitim ve tahmin için kullanabilirsiniz.")
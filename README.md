# 🩺 Diabetes Feature Engineering & ML Project

## 📌 İş Problemi
Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir.

## 📊 Veri Seti
- **Kaynak:** ABD Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri
- **Gözlem:** 768 | **Değişken:** 9
- **Hedef:** Outcome (1: Diyabet, 0: Sağlıklı)

## 🔧 Proje Adımları
### Görev 1 — Keşifçi Veri Analizi (EDA)
- Genel resim inceleme
- Kategorik & numerik değişken analizi
- Hedef değişken analizi
- Aykırı değer analizi
- Eksik değer analizi (gizli 0 değerleri)
- Korelasyon analizi

### Görev 2 — Feature Engineering
- 0 değerlerini NaN'a çevirme
- Outcome grubuna göre medyan doldurma
- Aykırı değer baskılama (IQR)
- 15+ yeni değişken türetme
- Label & One-Hot Encoding
- RobustScaler ile standartlaştırma

## 📈 Model Sonuçları

| Metrik    | Random Forest | XGBoost |
|-----------|:------------:|:-------:|
| Accuracy  | **0.900**    | 0.887   |
| Recall    | **0.892**    | 0.802   |
| Precision | 0.810        | **0.867** |
| F1        | **0.850**    | 0.833   |
| AUC       | **0.900**    | 0.868   |

## 🚀 Başlangıçtan Sona İyileşme

| Aşama | Accuracy |
|-------|----------|
| Ham veri | ~%74 |
| İlk Feature Engineering | %78 |
| Outcome grubuna göre doldurma + RobustScaler | **%90** |

## 🛠️ Kullanılan Teknolojiler
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Matplotlib, Seaborn

## 📁 Proje Yapısı
```
diabetes-feature-engineering/
├── diabets.py        # Ana proje kodu
├── README.md         # Proje açıklaması
└── .gitignore
```

## 👩‍💻 Yazar
Hilal Zerk Demirkan — Miuul Veri Bilimi & Yapay Zeka Bootcamp

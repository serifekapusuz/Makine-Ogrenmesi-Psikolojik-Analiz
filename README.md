# 🧠 Makine Öğrenmesi ile Depresyon Riski Tahmin Sistemi

Bu proje, bireylerin psikolojik durumlarını değerlendiren 9 soruluk (PHQ-9 formatına benzer) bir anket verisi kullanarak **Depresyon Riski** tahmini yapan uçtan uca bir Makine Öğrenmesi (Machine Learning) ve Derin Öğrenme (Deep Learning) sistemidir. Proje, veri ön işleme, sınıf dengesizliği giderme (SMOTE) ve interaktif bir kullanıcı arayüzü (UI) içerir.

## 🚀 Proje Özellikleri

* **Akıllı Veri Dönüşümü:** Kullanıcıların metinsel anket cevapları ("Her zaman", "Hiçbir zaman" vb.) NLP tabanlı bir mantıkla otomatik olarak 0-3 arası sayısal skorlara dönüştürülür.
* **Veri Dengesizliği Çözümü (SMOTE):** Tıbbi veri setlerinde sıkça görülen "sağlıklı hasta fazlalığı" problemi, SMOTE algoritması ile sentetik veri üretilerek dengelenmiştir (Accuracy Paradox önlenmiştir).
* **Geniş Model Yelpazesi:** 5 farklı Geleneksel ML Modeli (Random Forest, Logistic Regression, SVM, KNN, Naive Bayes) ve 1 adet Yapay Sinir Ağı (ANN) eğitilmiştir.
* **Erken Durdurma (Early Stopping):** Yapay Sinir Ağı modelinde aşırı öğrenmeyi (Overfitting) engellemek için Dropout ve Early Stopping mekanizmaları kullanılmıştır.
* **İstatistiksel Analiz:** Modellerin performans farklarının tesadüfi olup olmadığını ölçmek için **McNemar Testi** uygulanmıştır.
* **İnteraktif Kullanıcı Arayüzü (UI):** `ipywidgets` kullanılarak Jupyter Notebook içerisinde çalışan, kullanıcı dostu bir arayüz tasarlanmıştır. Model, anında risk analizi yapabilir.

## 🛠️ Kullanılan Teknolojiler

* **Dil:** Python 3.x
* **Veri İşleme & Analiz:** Pandas, NumPy
* **Makine Öğrenmesi:** Scikit-Learn
* **Derin Öğrenme:** TensorFlow / Keras
* **Dengesiz Veri Çözümü:** Imbalanced-Learn (SMOTE)
* **Görselleştirme:** Matplotlib, Seaborn
* **Arayüz (UI):** ipywidgets

## 📊 Veri Seti ve Sorular

Model, aşağıdaki 9 temel semptom üzerinden değerlendirme yapmaktadır:
1. İlgi kaybı veya zevk alamama
2. Moral bozukluğu, depresif hissetme
3. Uyku problemleri (Çok/Az)
4. Yorgunluk veya enerji düşüklüğü
5. İştah bozukluğu (Çok/Az)
6. Kendini kötü/başarısız hissetme
7. Odaklanma sorunu
8. Hareketlerde yavaşlama/huzursuzluk
9. Kendine zarar verme düşüncesi

Hedef değişken (Target), bu cevapların şiddetine göre `0 (Sağlıklı - Düşük Risk)` ve `1 (Depresyon Riski)` olarak ikili (binary) sınıflandırılmıştır.

## ⚙️ Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

**1. Gerekli kütüphaneleri yükleyin:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow imbalanced-learn ipywidgets statsmodels openpyxl

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import warnings

# Uyarıları gizle
warnings.filterwarnings('ignore')

class DepresyonUygulamasi:
    def __init__(self, root):
        self.root = root
        self.root.title("Depresyon Riski Tahmin Sistemi")
        self.root.geometry("600x750")
        self.root.configure(bg="#f0f0f0")

        # --- 1. MODELİ ARKA PLANDA EĞİT ---
        self.status_label = tk.Label(root, text="Sistem Hazırlanıyor... Lütfen Bekleyiniz.", bg="#f0f0f0", fg="blue")
        self.status_label.pack(pady=10)
        self.root.update() # Arayüzü güncelle
        
        try:
            self.modeli_egit()
            self.status_label.config(text="Sistem Hazır! Lütfen bilgileri giriniz.", fg="green")
        except Exception as e:
            self.status_label.config(text=f"Hata Oluştu: {str(e)}", fg="red")
            messagebox.showerror("Hata", f"Veri seti yüklenemedi!\n{str(e)}")
            return

        # --- 2. ARAYÜZ ELEMANLARI ---
        
        # Başlık
        main_title = tk.Label(root, text="Öğrenci Depresyon Riski Analizi", font=("Arial", 16, "bold"), bg="#f0f0f0")
        main_title.pack(pady=10)

        # Soru Alanı (Scrollable Frame yapılabilir ama basit tutalım)
        self.questions = [
            "1. İlgi kaybı veya zevk alamama?",
            "2. Moral bozukluğu, depresif hissetme?",
            "3. Uyku problemleri (Çok/Az)?",
            "4. Yorgunluk veya enerji düşüklüğü?",
            "5. İştah bozukluğu (Çok/Az)?",
            "6. Kendini kötü/başarısız hissetme?",
            "7. Odaklanma sorunu?",
            "8. Hareketlerde yavaşlama/huzursuzluk?",
            "9. Kendine zarar verme düşüncesi?"
        ]
        
        self.combos = []
        
        # Seçenekler (Puan Karşılıkları)
        self.secenekler = {
            "Hiç (Not at all)": 0,
            "Birkaç gün (Several days)": 1,
            "Günlerin yarısından fazlası": 2,
            "Hemen hemen her gün": 3
        }

        # Formu Oluştur
        frame = tk.Frame(root, bg="#f0f0f0")
        frame.pack(pady=10, padx=20, fill="both", expand=True)

        for i, soru in enumerate(self.questions):
            lbl = tk.Label(frame, text=soru, bg="#f0f0f0", font=("Arial", 10), anchor="w")
            lbl.grid(row=i, column=0, sticky="w", pady=5)
            
            combo = ttk.Combobox(frame, values=list(self.secenekler.keys()), state="readonly", width=30)
            combo.current(0) # Varsayılan: Hiç
            combo.grid(row=i, column=1, padx=10, pady=5)
            self.combos.append(combo)

        # Buton
        btn_tahmin = tk.Button(root, text="RİSKİ HESAPLA", font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", command=self.tahmin_et, height=2)
        btn_tahmin.pack(pady=20, fill="x", padx=50)

        # Sonuç Alanı
        self.lbl_sonuc = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="#f0f0f0")
        self.lbl_sonuc.pack(pady=10)

    def text_to_score(self, text):
        # Akıllı Dönüşüm Fonksiyonu (Colab'deki ile aynı)
        text = str(text).lower()
        if 'not' in text or 'good' in text or 'fine' in text or 'confident' in text or 'no ' in text: return 0
        elif 'constant' in text or 'nearly' in text or 'dead' in text or 'empty' in text or 'suicid' in text: return 3
        elif 'half' in text or 'frequent' in text or 'lot' in text: return 2
        elif 'several' in text or 'few' in text: return 1
        else: return 1

    def modeli_egit(self):
        # Veriyi Oku
        df = pd.read_excel("veri.xlsx") # Dosya adının aynı olduğundan emin ol!
        
        # Sütun İsimleri Düzenle
        yeni_isimler = ['ID', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'Total_Score', 'Severity']
        if len(df.columns) == 12: df.columns = yeni_isimler
        else: df.columns = [f"Col_{i}" for i in range(len(df.columns)-1)] + ['Severity']
        df = df.drop(columns=['ID', 'Total_Score'], errors='ignore')

        # Dönüşüm
        soru_sutunlari = [col for col in df.columns if 'Severity' not in col and 'Target' not in col]
        for col in soru_sutunlari:
            df[col] = df[col].apply(self.text_to_score)

        # Hedef Belirle
        def severity_to_binary(val):
            val = str(val).lower()
            return 0 if 'none' in val or 'minimal' in val else 1
        
        df['Target'] = df['Severity'].apply(severity_to_binary)
        X = df.drop(columns=['Severity', 'Target'])
        y = df['Target']
        
        self.feature_columns = X.columns # Sütun isimlerini sakla

        # Ölçeklendirme ve SMOTE
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
        
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_scaled, y)
        
        # Model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

    def tahmin_et(self):
        try:
            # Kullanıcı Girdilerini Al
            girdiler = []
            for combo in self.combos:
                secilen_metin = combo.get()
                puan = self.secenekler[secilen_metin]
                girdiler.append(puan)
            
            # DataFrame'e çevir
            input_df = pd.DataFrame([girdiler], columns=self.feature_columns)
            
            # Ölçekle
            input_scaled = self.scaler.transform(input_df)
            
            # Tahmin
            proba = self.model.predict_proba(input_scaled)[0][1] * 100
            
            # Ekrana Yaz
            if proba > 50:
                self.lbl_sonuc.config(text=f"SONUÇ: YÜKSEK RİSK (%{proba:.1f}) 🔴\nLütfen bir uzmana danışın.", fg="red")
            else:
                self.lbl_sonuc.config(text=f"SONUÇ: DÜŞÜK RİSK (%{proba:.1f}) 🟢\nSağlıklı görünüyorsunuz.", fg="green")
                
        except Exception as e:
            messagebox.showerror("Hata", f"Tahmin sırasında hata oluştu:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DepresyonUygulamasi(root)
    root.mainloop()
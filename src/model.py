#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Güneş Sistemi Gezegen Tanıma Modeli
===================================
Bu modül, OpenCV kullanarak Güneş Sistemi gezegenlerini tanıyan bir model içerir.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class SolarSystemPlanetRecognizer:
    """
    OpenCV ve makine öğrenmesi kullanarak Güneş Sistemi gezegenlerini tanıyan sınıf.
    """
    
    def __init__(self, dataset_path=None):
        """
        SolarSystemPlanetRecognizer sınıfının başlatıcısı.
        
        Parameters:
        -----------
        dataset_path : str, optional
            Veri setinin bulunduğu dizin yolu. Belirtilmezse, varsayılan olarak '../dataset/merged_planets' kullanılır.
        """
        self.dataset_path = dataset_path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset', 'merged_planets')
        self.image_size = (128, 128)  # Tüm görüntüleri bu boyuta yeniden boyutlandır
        self.features = []
        self.labels = []
        self.label_encoder = LabelEncoder()
        self.model = None
        self.class_names = []
        
    def load_dataset(self):
        """
        Veri setini yükler ve özellikleri/etiketleri hazırlar.
        
        Returns:
        --------
        tuple
            (features, labels) şeklinde özellikler ve etiketler.
        """
        print(f"Veri seti yükleniyor: {self.dataset_path}")
        
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Veri seti dizini bulunamadı: {self.dataset_path}")
        
        # Veri setindeki her gezegen klasörünü işle
        for planet_name in os.listdir(self.dataset_path):
            planet_dir = os.path.join(self.dataset_path, planet_name)
            
            # Sadece dizinleri işle
            if not os.path.isdir(planet_dir):
                continue
                
            # Güneş klasörü boş olabilir, kontrol et
            if planet_name == "Sun" and len(os.listdir(planet_dir)) == 0:
                print(f"Uyarı: {planet_name} klasörü boş, atlanıyor.")
                continue
                
            print(f"İşleniyor: {planet_name}")
            
            # Bu gezegene ait tüm görüntüleri işle
            for img_file in os.listdir(planet_dir):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                img_path = os.path.join(planet_dir, img_file)
                
                try:
                    # Görüntüyü oku ve yeniden boyutlandır
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Uyarı: {img_path} okunamadı, atlanıyor.")
                        continue
                        
                    img = cv2.resize(img, self.image_size)
                    
                    # Özellik çıkarma
                    features = self._extract_features(img)
                    
                    # Özellikleri ve etiketi kaydet
                    self.features.append(features)
                    self.labels.append(planet_name)
                    
                except Exception as e:
                    print(f"Hata: {img_path} işlenirken bir sorun oluştu: {e}")
        
        # Etiketleri sayısal değerlere dönüştür
        self.class_names = sorted(list(set(self.labels)))
        self.labels = self.label_encoder.fit_transform(self.labels)
        
        print(f"Veri seti yüklendi: {len(self.features)} görüntü, {len(self.class_names)} sınıf")
        print(f"Sınıflar: {self.class_names}")
        
        return np.array(self.features), np.array(self.labels)
    
    def _extract_features(self, image):
        """
        Bir görüntüden özellikler çıkarır.
        
        Parameters:
        -----------
        image : numpy.ndarray
            İşlenecek görüntü.
            
        Returns:
        --------
        numpy.ndarray
            Çıkarılan özellikler.
        """
        # Görüntüyü gri tonlamaya dönüştür
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # HOG (Histogram of Oriented Gradients) özelliklerini çıkar
        win_size = (128, 128)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        hog_features = hog.compute(gray)
        
        # Renk histogramı özelliklerini çıkar
        color_features = []
        for i in range(3):  # BGR kanalları
            hist = cv2.calcHist([image], [i], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            color_features.extend(hist)
        
        # Tüm özellikleri birleştir
        all_features = np.concatenate((hog_features.flatten(), color_features))
        
        return all_features
    
    def train_model(self, test_size=0.2, random_state=42):
        """
        Veri setini kullanarak modeli eğitir.
        
        Parameters:
        -----------
        test_size : float, optional
            Test seti için ayrılacak veri oranı. Varsayılan 0.2.
        random_state : int, optional
            Rastgele durum değeri. Varsayılan 42.
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test) şeklinde eğitim ve test verileri.
        """
        # Veri setini yükle
        X, y = self.load_dataset()
        
        # Eğitim ve test setlerine ayır
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        
        print(f"Eğitim seti: {X_train.shape[0]} örnek")
        print(f"Test seti: {X_test.shape[0]} örnek")
        
        # SVM modelini oluştur ve eğit
        self.model = cv2.ml.SVM_create()
        self.model.setType(cv2.ml.SVM_C_SVC)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setGamma(0.1)
        self.model.setC(10.0)
        self.model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
        
        # Eğitim verilerini uygun formata dönüştür
        train_data = np.float32(X_train)
        train_labels = np.int32(y_train)
        
        print("Model eğitiliyor...")
        self.model.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
        print("Model eğitimi tamamlandı.")
        
        return X_train, X_test, y_train, y_test
    
    def evaluate_model(self, X_test, y_test):
        """
        Eğitilmiş modeli değerlendirir.
        
        Parameters:
        -----------
        X_test : numpy.ndarray
            Test özellikleri.
        y_test : numpy.ndarray
            Test etiketleri.
            
        Returns:
        --------
        float
            Model doğruluğu.
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmemiş. Önce train_model() metodunu çağırın.")
        
        # Test verilerini uygun formata dönüştür
        test_data = np.float32(X_test)
        
        # Tahminleri yap
        _, y_pred = self.model.predict(test_data)
        y_pred = y_pred.flatten().astype(int)
        
        # Performans metriklerini hesapla
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Doğruluk: {accuracy:.4f}")
        
        # Sınıflandırma raporu
        print("\nSınıflandırma Raporu:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Karmaşıklık matrisi
        cm = confusion_matrix(y_test, y_pred)
        
        # Karmaşıklık matrisini görselleştir
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Karmaşıklık Matrisi')
        plt.colorbar()
        tick_marks = np.arange(len(self.class_names))
        plt.xticks(tick_marks, self.class_names, rotation=45)
        plt.yticks(tick_marks, self.class_names)
        
        # Hücrelere değerleri ekle
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('Gerçek Etiket')
        plt.xlabel('Tahmin Edilen Etiket')
        
        # Sonuçları kaydet
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
        
        return accuracy
    
    def save_model(self, model_path=None):
        """
        Eğitilmiş modeli kaydeder.
        
        Parameters:
        -----------
        model_path : str, optional
            Modelin kaydedileceği dosya yolu. Belirtilmezse, varsayılan olarak '../models/planet_recognition_model.xml' kullanılır.
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmemiş. Önce train_model() metodunu çağırın.")
        
        if model_path is None:
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, 'planet_recognition_model.xml')
        
        self.model.save(model_path)
        
        # Sınıf adlarını da kaydet
        class_names_path = os.path.join(os.path.dirname(model_path), 'class_names.txt')
        with open(class_names_path, 'w') as f:
            for class_name in self.class_names:
                f.write(f"{class_name}\n")
        
        print(f"Model kaydedildi: {model_path}")
        print(f"Sınıf adları kaydedildi: {class_names_path}")
    
    def load_saved_model(self, model_path=None, class_names_path=None):
        """
        Kaydedilmiş bir modeli yükler.
        
        Parameters:
        -----------
        model_path : str, optional
            Modelin yükleneceği dosya yolu. Belirtilmezse, varsayılan olarak '../models/planet_recognition_model.xml' kullanılır.
        class_names_path : str, optional
            Sınıf adlarının yükleneceği dosya yolu. Belirtilmezse, varsayılan olarak '../models/class_names.txt' kullanılır.
        """
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'planet_recognition_model.xml')
        
        if class_names_path is None:
            class_names_path = os.path.join(os.path.dirname(model_path), 'class_names.txt')
        
        # Modeli yükle
        self.model = cv2.ml.SVM_load(model_path)
        
        # Sınıf adlarını yükle
        with open(class_names_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        
        # Label encoder'ı yeniden oluştur
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.class_names)
        
        print(f"Model yüklendi: {model_path}")
        print(f"Sınıf adları yüklendi: {class_names_path}")
        print(f"Sınıflar: {self.class_names}")
    
    def predict(self, image_path):
        """
        Bir görüntüdeki gezegeni tahmin eder.
        
        Parameters:
        -----------
        image_path : str
            Tahmin edilecek görüntünün dosya yolu.
            
        Returns:
        --------
        tuple
            (predicted_class, confidence) şeklinde tahmin edilen sınıf ve güven değeri.
        """
        if self.model is None:
            raise ValueError("Model henüz yüklenmemiş. Önce train_model() veya load_saved_model() metodunu çağırın.")
        
        # Görüntüyü oku ve yeniden boyutlandır
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Görüntü okunamadı: {image_path}")
            
        img = cv2.resize(img, self.image_size)
        
        # Özellik çıkarma
        features = self._extract_features(img)
        
        # Tahmin yap
        features = np.float32([features])
        _, results = self.model.predict(features)
        predicted_class_idx = int(results[0])
        
        # Güven değerini hesapla (SVM için karar fonksiyonu değerleri)
        confidence = self.model.predict(features, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)[1][0]
        
        # En yüksek güven değerini bul
        max_confidence = np.max(np.abs(confidence))
        
        # Tahmin edilen sınıfı döndür
        predicted_class = self.class_names[predicted_class_idx]
        
        return predicted_class, max_confidence
    
    def visualize_prediction(self, image_path, output_path=None):
        """
        Bir görüntüdeki gezegen tahminini görselleştirir.
        
        Parameters:
        -----------
        image_path : str
            Tahmin edilecek görüntünün dosya yolu.
        output_path : str, optional
            Sonuç görüntüsünün kaydedileceği dosya yolu. Belirtilmezse, görüntü kaydedilmez, sadece gösterilir.
            
        Returns:
        --------
        tuple
            (predicted_class, confidence) şeklinde tahmin edilen sınıf ve güven değeri.
        """
        # Tahmin yap
        predicted_class, confidence = self.predict(image_path)
        
        # Orijinal görüntüyü oku
        img = cv2.imread(image_path)
        
        # Görüntüyü BGR'den RGB'ye dönüştür (matplotlib için)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Görselleştir
        plt.figure(figsize=(10, 8))
        plt.imshow(img_rgb)
        plt.title(f"Tahmin: {predicted_class} (Güven: {confidence:.4f})")
        plt.axis('off')
        
        # Sonucu kaydet veya göster
        if output_path:
            plt.savefig(output_path)
            print(f"Sonuç kaydedildi: {output_path}")
        else:
            plt.show()
        
        return predicted_class, confidence


def main():
    """
    Ana fonksiyon. Modeli eğitir, değerlendirir ve kaydeder.
    """
    # Veri seti yolunu belirle
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset', 'merged_planets')
    
    # Modeli oluştur
    recognizer = SolarSystemPlanetRecognizer(dataset_path)
    
    # Modeli eğit
    X_train, X_test, y_train, y_test = recognizer.train_model()
    
    # Modeli değerlendir
    accuracy = recognizer.evaluate_model(X_test, y_test)
    
    # Modeli kaydet
    recognizer.save_model()
    
    print(f"Model eğitimi ve değerlendirmesi tamamlandı. Doğruluk: {accuracy:.4f}")


if __name__ == "__main__":
    main()

# TR:

# Güneş Sistemi Gezegen Tanıma Projesi

Bu proje, OpenCV ve makine öğrenmesi teknikleri kullanarak Güneş Sistemi gezegenlerini tanıyan bir model geliştirmeyi amaçlamaktadır. Proje, görüntü işleme ve nesne tanıma alanlarında temel bir uygulama sunmaktadır.

## Özellikler

- Güneş Sistemi gezegenlerini (Merkür, Venüs, Dünya, Mars, Jüpiter, Satürn, Uranüs, Neptün ve Plüton) tanıma
- OpenCV kütüphanesi kullanarak görüntü işleme
- HOG (Histogram of Oriented Gradients) ve renk histogramı özellikleri çıkarma
- SVM (Support Vector Machine) sınıflandırıcısı ile model eğitimi
- Görsel tahmin sonuçları

## Gereksinimler

```
opencv-python>=4.5.0
numpy>=1.19.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
```

## Kurulum

```bash
# Projeyi klonlayın
git clone https://github.com/burakkurt07/solar_system_planet_recognition.git
cd solar_system_planet_recognition

# Gerekli paketleri yükleyin
pip install -r requirements.txt
```

## Kullanım

### Model Eğitimi

```bash
python src/model.py
```

### Tahmin

```python
from src.model import SolarSystemPlanetRecognizer

# Modeli yükle
recognizer = SolarSystemPlanetRecognizer()
recognizer.load_saved_model()

# Bir görüntüdeki gezegeni tahmin et
predicted_class, confidence = recognizer.predict('path/to/planet_image.jpg')
print(f"Tahmin edilen gezegen: {predicted_class}, Güven: {confidence:.4f}")

# Tahmin sonucunu görselleştir
recognizer.visualize_prediction('path/to/planet_image.jpg', 'results/prediction_result.png')
```

## Proje Yapısı

```
solar_system_planet_recognition/
├── dataset/                  # Veri seti dizini
│   └── merged_planets/       # Birleştirilmiş gezegen görüntüleri
├── models/                   # Eğitilmiş modeller
├── results/                  # Sonuçlar ve görselleştirmeler
├── src/                      # Kaynak kod
│   └── model.py              # Ana model sınıfı
├── README.md                 # Bu dosya
└── requirements.txt          # Gerekli paketler
```

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakınız.

---


# EN:


# Solar System Planet Recognition Project

This project aims to develop a model that recognizes Solar System planets using OpenCV and machine learning techniques. The project provides a basic application in the fields of image processing and object recognition.

## Features

- Recognition of Solar System planets (Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, and Pluto)
- Image processing using OpenCV library
- Feature extraction with HOG (Histogram of Oriented Gradients) and color histograms
- Model training with SVM (Support Vector Machine) classifier
- Visual prediction results

## Requirements

```
opencv-python>=4.5.0
numpy>=1.19.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
```

## Installation

```bash
# Clone the project
git clone https://github.com/username/solar_system_planet_recognition.git
cd solar_system_planet_recognition

# Install required packages
pip install -r requirements.txt
```

## Usage

### Model Training

```bash
python src/model.py
```

### Prediction

```python
from src.model import SolarSystemPlanetRecognizer

# Load the model
recognizer = SolarSystemPlanetRecognizer()
recognizer.load_saved_model()

# Predict a planet in an image
predicted_class, confidence = recognizer.predict('path/to/planet_image.jpg')
print(f"Predicted planet: {predicted_class}, Confidence: {confidence:.4f}")

# Visualize the prediction result
recognizer.visualize_prediction('path/to/planet_image.jpg', 'results/prediction_result.png')
```

## Project Structure

```
solar_system_planet_recognition/
├── dataset/                  # Dataset directory
│   └── merged_planets/       # Merged planet images
├── models/                   # Trained models
├── results/                  # Results and visualizations
├── src/                      # Source code
│   └── model.py              # Main model class
├── README.md                 # This file
└── requirements.txt          # Required packages
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

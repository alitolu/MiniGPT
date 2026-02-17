# MiniGPT - C# ile Sıfırdan YZ Motoru

## Nedir Bu Proje?
MiniGPT, tamamen C# programlama dili kullanılarak **SIFIRDAN** yazılmış bir Yapay Zeka (LLM) motorudur.
Hiçbir hazır YZ kütüphanesi (Python, PyTorch, TensorFlow vb.) kullanılmamıştır. Her şeyi (Tensor matematiği, Sinir Ağları, Eğitim döngüsü) el ile kodladık. Proje, modern yapay zekaların perde arkasında nasıl çalıştığını anlamak için geliştirilmiş bir eğitim ve AR-GE projesidir.

## Neler Yapabiliyor?

### 1. Kendi Kendine Öğrenme (Eğitim)
- Ham metin dosyalarını (örneğin kitapları) okuyup dilin kurallarını ve mantığını öğrenebiliyor.
- `dataset.txt` dosyasındaki verileri kullanarak sıfırdan eğitim yapabiliyor.

### 2. Sohbet Edebiliyor (Chat)
- Eğitildiği bilgiler ışığında sizinle sohbet edebiliyor.
- Sorulara cevap verip, metnin devamını getirebiliyor.

### 3. Akıllı Ajan (Agent) Yetenekleri
- Sadece konuşmakla kalmıyor, **araç (tool)** kullanabiliyor (Örn: Hesap makinesi).
- **Hafızası var:** Konuştuğunuz eski şeyleri hatırlayabiliyor.
- **Bilgi Bankası (RAG):** Kendi veritabanından bilgi çekip cevaplarına ekleyebiliyor (Örn: Şirket içi dökümanlar).

### 4. Teknoloji Harikası Özellikler
- **FlashAttention:** İşlemciyi (CPU) çok verimli kullanmak için özel matematiksel teknikler içeriyor.
- **Quantization (Q4):** Modeli %75 küçülterek daha az RAM harcamasını sağlıyor.
- **Streaming:** Cevapları ChatGPT gibi kelime kelime ekrana yazdırabiliyor.
- **Web Arayüzü:** Kendi içinde modern bir web sunucusu ve şık bir sohbet ekranı var.

## Nasıl Kullanılır?

Projeyi indirdikten sonra terminalde şu komutu yazmanız yeterli:

    dotnet run

Karşınıza 4 seçenekli bir menü çıkacak:
1. **Train:** Modeli sıfırdan eğitmeye başlar.
2. **Chat:** Konsol üzerinden hızlıca sohbet etmenizi sağlar.
3. **Serve:** Web tarayıcınızdan (`localhost:5000`) erişebileceğiniz modern bir arayüz açar.
4. **Agent:** Akıllı ajan modunu başlatır (Tool kullanabilen versiyon).

## Teknik Özellikler (Mühendisler İçin)
- **Dil:** C# .NET 8.0
- **Mimari:** Transformer (Decoder-only)
- **Optimizasyon:** AdamW Optimizer, CrossEntropy Loss
- **Hızlandırma:** PagedAttention (KV Cache), FlashAttention, SIMD Vectorization
- **Platform:** Windows, Linux, Mac (Tüm .NET destekli sistemler)

Bu proje, bir YZ'nin "Hello World"ü değil, çalışan minyatür bir beynidir.

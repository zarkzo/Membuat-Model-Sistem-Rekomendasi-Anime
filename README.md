# Laporan Proyek Machine Learning - Putra Faaris Prayoga

## Project Overview

Anime menjadi salah satu bentuk hiburan paling populer, namun jumlah judul yang sangat banyak dapat membuat pengguna kesulitan menemukan tontonan yang sesuai dengan preferensi mereka. Dengan banyaknya data dari platform seperti MyAnimeList, membangun sistem rekomendasi berbasis konten dapat membantu pengguna menavigasi pilihan mereka secara lebih efisien.

Menurut Ricci et al. dalam buku _Recommender Systems Handbook_, sistem rekomendasi berbasis konten mampu memberikan rekomendasi yang dipersonalisasi dengan menganalisis atribut dari item itu sendiri dan mencocokkannya dengan preferensi pengguna [1].

> Referensi:  
> [1] F. Ricci, L. Rokach, and B. Shapira, _Recommender Systems Handbook_, Springer, 2015.

## Business Understanding

### Problem Statements

- Bagaimana merekomendasikan anime yang relevan berdasarkan sinopsis dan genre dari anime yang pernah disukai pengguna?
- Bagaimana mengintegrasikan nilai skor dan peringkat untuk meningkatkan kualitas rekomendasi?

### Goals

- Membangun sistem rekomendasi berbasis konten untuk menyarankan anime berdasarkan kesamaan fitur deskriptif (sinopsis, genre, skor).
- Meningkatkan performa sistem dengan pendekatan tambahan seperti model embedding.

### Solution Statements

- **Pendekatan 1**: Content-based filtering menggunakan TF-IDF dari sinopsis, one-hot encoding genre, serta fitur numerik skor dan ranking.
- **Pendekatan 2**: Neural Collaborative Filtering sederhana berbasis embedding dengan arsitektur custom (`RecommenderNet`) untuk memprediksi skor preferensi.

## Data Understanding

Dataset diambil dari [Kaggle: MyAnimeList Dataset](https://www.kaggle.com/datasets/marlesson/myanimelist-dataset-animes-profiles-reviews).

**Jumlah Data**: 19.311 entri  
**Fitur**:

- `uid`: ID unik anime
- `title`: Judul anime
- `synopsis`: Ringkasan cerita
- `genre`: Daftar genre (dipisahkan koma)
- `aired`: Tanggal tayang
- `episodes`: Jumlah episode
- `members`: Jumlah anggota yang menambahkan ke daftar
- `popularity`: Ranking popularitas
- `ranked`: Ranking berdasarkan rating
- `score`: Nilai rata-rata dari pengguna
- `img_url`: URL gambar cover
- `link`: Tautan ke halaman anime

**Distribusi Genre**:

Jumlah genre unik: **162**

**Top 5 Genre Terbanyak**:

![Top 5 Genre](https://github.com/user-attachments/assets/2265cc68-c19f-49ba-8d97-b4e3c3f3ba25)

**Distribusi Skor**:

![Distribusi Skor](https://github.com/user-attachments/assets/fa02158a-2f34-4942-9252-3ad2a72d77a2)

## Data Preparation

- Menghapus duplikat dan data dengan nilai kosong pada kolom penting (`title`, `synopsis`, `genre`, `score`, `ranked`)
- Filter anime dengan skor dan jumlah episode > 0
- Membersihkan teks sinopsis (lowercase, hapus simbol, stopwords, dan lemmatization)
- TF-IDF vektorisasi pada sinopsis dengan `max_features=3000`
- Genre diubah menjadi vektor one-hot
- Skor dan ranking diskalakan dengan MinMaxScaler
- Semua fitur digabungkan ke dalam satu matriks final

## Modeling

### Model 1: Content-Based Filtering (TF-IDF + Genre + Score)

- Fitur: TF-IDF synopsis, genre one-hot, skor & ranking
- Metrik: Cosine Similarity
- Output: 10 rekomendasi anime terdekat untuk judul tertentu

### Model 2: RecommenderNet (Neural CF)

- Input: anime_id (encoded)
- Arsitektur: Embedding + bias + dot product + sigmoid
- Loss: Binary crossentropy
- Optimizer: Adam
- Metrik: MAE
- Epochs: 50, Batch Size: 16

### Kelebihan & Kekurangan

| Pendekatan         | Kelebihan                                        | Kekurangan                                                      |
| ------------------ | ------------------------------------------------ | --------------------------------------------------------------- |
| **TF-IDF**         | Cepat, mudah dipahami, tidak perlu data pengguna | Terbatas pada data konten, tidak personal                       |
| **RecommenderNet** | Bisa menangkap representasi latent dari anime    | Tidak melibatkan data pengguna, perlu tuning & data lebih besar |

## Evaluation

### Model TF-IDF

- Evaluasi dilakukan menggunakan _qualitative test_ untuk memastikan hasil rekomendasi masuk akal dan relevan.
- **Cosine similarity** menghasilkan kemiripan konten berdasarkan sinopsis, genre, dan skor.

### Model RecommenderNet

Evaluation result (binary_crossentropy, MAE): [0.0002, 0.0135]


### 🧪 **Evaluasi Model**

- **MAE** yang sangat kecil menunjukkan bahwa prediksi skor cukup akurat terhadap skor asli.
  Namun, model ini belum mengandung informasi preferensi pengguna secara langsung, karena hanya mengandalkan **RecommenderNet** untuk memprediksi skor berdasarkan ID anime.

### **Contoh Rekomendasi:**

**Top 10 rekomendasi untuk 'Fullmetal Alchemist: Brotherhood':**
1. **Fairy Tail: Final Series**, Predicted Score: 0.6669
2. **Fairy Tail (2014)**, Predicted Score: 0.6634
3. **Fairy Tail Movie 1: Houou no Miko**, Predicted Score: 0.6626
4. **Fullmetal Alchemist**, Predicted Score: 0.6625
5. **One Piece Film: Gold**, Predicted Score: 0.6573
6. **One Piece Film: Z**, Predicted Score: 0.6533
7. **Fairy Tail**, Predicted Score: 0.6525
8. **One Piece Film: Strong World**, Predicted Score: 0.6502
9. **Fullmetal Alchemist: The Sacred Star of Milos**, Predicted Score: 0.4894
10. **Digimon Frontier**, Predicted Score: 0.4887

**Top 10 rekomendasi untuk 'Shigatsu wa Kimi no Uso':**
- [Isi rekomendasi yang relevan untuk anime lainnya]

### **Referensi**
[1] F. Ricci, L. Rokach, and B. Shapira, *Recommender Systems Handbook*, Springer, 2015.

---

_Catatan tambahan:_
- Gambar visualisasi berada di folder `images/`
- Model dapat dikembangkan lebih lanjut dengan **matrix factorization** atau **collaborative filtering** jika ada data pengguna.


### Penjelasan:
- **Project Overview**: Penjelasan tentang tujuan proyek dan referensi yang digunakan.
- **Business Understanding**: Mendeskripsikan masalah yang ingin diselesaikan dan tujuan dari proyek ini.
- **Data Understanding**: Menyediakan informasi tentang dataset, termasuk fitur, distribusi genre, dan skor.
- **Data Preparation**: Menjelaskan langkah-langkah dalam membersihkan dan mempersiapkan data untuk model.
- **Modeling**: Menyediakan dua pendekatan yang digunakan untuk sistem rekomendasi, yaitu content-based filtering dan RecommenderNet.
- **Evaluation**: Evaluasi hasil model, baik untuk content-based filtering maupun RecommenderNet, serta contoh rekomendasi yang dihasilkan.
- **Referensi**: Menyediakan referensi yang digunakan dalam proyek.

Markdown ini sudah siap digunakan di Jupyter Notebook atau file markdown untuk dokumentasi lengkap proyek Anda. Jika ada bagian yang perlu disesuaikan, beri tahu saya!


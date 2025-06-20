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

Dataset yang digunakan diambil dari [Kaggle: MyAnimeList Dataset](https://www.kaggle.com/datasets/marlesson/myanimelist-dataset-animes-profiles-reviews).

### Jumlah Data
- **Jumlah baris**: 19.311 entri
- **Jumlah kolom**: 12 kolom, yang mencakup informasi mengenai anime, genre, skor, ranking, dan lainnya.

### Kondisi Data
- **Missing Values (NaN)**:
    - Kolom `synopsis` memiliki 975 nilai yang hilang.
    - Kolom `score` memiliki 579 nilai yang hilang.
    - Kolom `ranked` memiliki 3.212 nilai yang hilang.
- **Duplikat**: Dataset mengandung 2.943 baris duplikat yang perlu dihapus.
- **Outliers**: Skor dan jumlah episode berada pada rentang yang besar, dari 1 hingga 3057 episode dan skor anime bervariasi dari 1.25 hingga 9.23, yang menunjukkan adanya data yang memiliki skor atau episode yang sangat tinggi.

### Tautan Sumber Data
Dataset ini dapat diakses melalui Kaggle di tautan berikut: [Kaggle: MyAnimeList Dataset](https://www.kaggle.com/datasets/marlesson/myanimelist-dataset-animes-profiles-reviews).

### Uraian Fitur Dataset
- `uid`: ID unik anime.
- `title`: Judul anime.
- `synopsis`: Ringkasan cerita anime.
- `genre`: Daftar genre yang dimiliki anime (dipisahkan dengan koma).
- `aired`: Tanggal tayang anime.
- `episodes`: Jumlah episode yang dimiliki oleh anime.
- `members`: Jumlah anggota yang menambahkan anime ke dalam daftar mereka.
- `popularity`: Ranking popularitas anime di platform.
- `ranked`: Ranking anime berdasarkan rating keseluruhan.
- `score`: Nilai rata-rata yang diberikan oleh pengguna kepada anime.
- `img_url`: URL gambar cover anime.
- `link`: Tautan ke halaman detail anime.

### Distribusi Genre
- **Jumlah genre unik**: 162 genre unik ditemukan dalam dataset ini.
  
#### Top 5 Genre Terbanyak:
![Top 5 Genre](https://github.com/user-attachments/assets/2265cc68-c19f-49ba-8d97-b4e3c3f3ba25)

#### Distribusi Skor:
![Distribusi Skor](https://github.com/user-attachments/assets/fa02158a-2f34-4942-9252-3ad2a72d77a2)

## Data Preparation

Tahapan **Data Preparation** bertujuan untuk membersihkan dan mempersiapkan data agar dapat digunakan dalam model rekomendasi. Berikut adalah langkah-langkah yang dilakukan:

1. **Menghapus Duplikat dan Data Kosong (Missing Values)**:
   - Menghapus duplikat pada dataset menggunakan `.drop_duplicates()`.
   - Menghapus baris yang memiliki nilai kosong (NaN) pada kolom penting seperti `title`, `synopsis`, `genre`, `score`, dan `ranked` menggunakan `.dropna()`.

2. **Filter Data**:
   - Menghapus anime yang memiliki **skor** atau **jumlah episode** kurang dari atau sama dengan 0, untuk memastikan hanya anime yang valid yang digunakan dalam analisis lebih lanjut.

3. **Membersihkan Teks Sinopsis**:
   - Mengubah teks pada kolom `synopsis` menjadi huruf kecil (lowercase).
   - Menghapus simbol-simbol yang tidak diperlukan dari teks.
   - Melakukan **tokenisasi** untuk memecah teks menjadi kata-kata individu.
   - Menggunakan **lemmatization** untuk mengubah kata-kata ke bentuk dasarnya.
   - Menghapus **stopwords** (kata-kata yang tidak memberikan informasi penting seperti "and", "the", dll.) untuk memfokuskan pada kata-kata yang lebih relevan.

4. **Vektorisasi Sinopsis dengan TF-IDF**:
   - Menggunakan **TF-IDF (Term Frequency-Inverse Document Frequency)** untuk mengubah kolom `synopsis` menjadi representasi vektorial. 
   - Mengatur parameter `max_features=3000` untuk hanya mempertahankan 3000 fitur (kata) paling penting.

5. **Encoding Genre dengan One-Hot Encoding**:
   - Melakukan **One-Hot Encoding** pada kolom `genre` untuk mengubah genre anime menjadi vektor biner. Setiap genre akan menjadi kolom dengan nilai 1 jika genre tersebut ada di anime tersebut dan 0 jika tidak.

6. **Normalisasi Skor dan Ranking**:
   - Menggunakan **MinMaxScaler** untuk menormalkan kolom `score` dan `ranked` agar nilainya berada dalam rentang [0, 1]. Ini penting untuk memastikan bahwa semua fitur berada dalam skala yang sama, yang dapat membantu model bekerja lebih efisien.

7. **Encoding Title (Label Encoding)**:
   - Menggunakan **Label Encoding** pada kolom `title` untuk mengubah nama anime menjadi ID numerik unik. Hal ini diperlukan karena model machine learning tidak dapat memproses data kategori berbentuk teks, sehingga judul anime perlu dikonversi menjadi format numerik.

8. **Penggabungan Fitur**:
   - Menggabungkan seluruh fitur yang telah diproses (TF-IDF vektor, One-Hot Encoding genre, dan fitur yang dinormalisasi seperti `score` dan `ranked`) menjadi satu matriks fitur akhir menggunakan `hstack`. Matriks ini nantinya akan digunakan untuk perhitungan kesamaan antar anime.

9. **Splitting Data**:
   - **Pembagian Data Latih dan Data Uji** dilakukan di bagian kode berikut:
     ```python
     split = int(0.8 * len(df))
     x_train, x_test = x[:split], x[split:]
     y_train, y_test = y_scaled[:split], y_scaled[split:]
     ```
     - **Split** ini membagi dataset menjadi 80% untuk data latih dan 20% untuk data uji. Data latih (`x_train` dan `y_train`) digunakan untuk melatih model, sementara data uji (`x_test` dan `y_test`) digunakan untuk mengevaluasi model.

## Modeling

Pada tahap **Modeling**, dua pendekatan utama digunakan untuk membangun sistem rekomendasi anime: **Content-Based Filtering** (CBS) dan **Hybrid Model** yang menggabungkan pendekatan **Content-Based Filtering** dan **RecommenderNet**. Berikut adalah penjelasan lebih lanjut tentang setiap model yang digunakan:

### Model 1: Content-Based Filtering (Cosine Similarity + TF-IDF + Genre + Score)

**Definisi**:  
Sistem rekomendasi berbasis konten ini menggunakan informasi yang ada pada item (anime) itu sendiri, seperti **sinopsis**, **genre**, dan **skor**. Pada pendekatan ini, kami mengukur kesamaan antar anime menggunakan **cosine similarity** berdasarkan fitur-fitur yang diekstraksi dari sinopsis, genre, dan skor.

**Cara Kerja**:
1. **TF-IDF** digunakan untuk mengubah **sinopsis** anime menjadi representasi vektor berdasarkan frekuensi kata yang ada dalam teks. Ini memungkinkan model untuk menangkap kata-kata penting dalam sinopsis.
2. **One-hot Encoding** pada kolom **genre** untuk menangkap kategori genre anime dan mengubahnya menjadi representasi biner.
3. **Skor dan Ranking** dinormalisasi menggunakan **MinMaxScaler** untuk memastikan fitur berada dalam rentang yang sama, memudahkan pemrosesan selanjutnya.
4. **Cosine Similarity** digunakan untuk mengukur kesamaan antar anime berdasarkan kombinasi fitur-fitur ini, sehingga menghasilkan rekomendasi anime yang serupa dengan anime yang dipilih.

**Output**:  
10 rekomendasi anime terdekat berdasarkan kesamaan dengan judul anime yang dimasukkan.

**Contoh Rekomendasi**:
Berikut adalah contoh rekomendasi menggunakan **Content-Based Filtering** untuk anime **"Fullmetal Alchemist: Brotherhood"**:

| Rank | Anime Title                               | Genre                         | Score | Ranked |
|------|------------------------------------------|-------------------------------|-------|--------|
| 1    | Fullmetal Alchemist                     | Action, Adventure, Drama      | 8.25  | 287    |
| 2    | Fullmetal Alchemist: The Sacred Star of Milos | Action, Adventure, Drama      | 7.39  | 2140   |
| 3    | Fairy Tail Movie 1: Houou no Miko       | Action, Adventure, Fantasy    | 7.56  | 1535   |
| 4    | One Piece Film: Z                        | Action, Adventure, Fantasy    | 8.27  | 267    |
| 5    | One Piece Film: Strong World            | Action, Adventure, Fantasy    | 8.26  | 283    |
| 6    | Fairy Tail                              | Action, Adventure, Comedy     | 7.93  | 665    |
| 7    | Fairy Tail: Final Series                | Action, Adventure, Comedy     | 7.81  | 891    |
| 8    | Fairy Tail (2014)                       | Action, Adventure, Fantasy    | 7.97  | 617    |
| 9    | One Piece Film: Gold                    | Action, Adventure, Drama      | 8.07  | 484    |
| 10   | Digimon Frontier                        | Action, Adventure, Drama      | 7.22  | 2862   |

### Model 2: Hybrid Model (Content-Based + RecommenderNet)

**Definisi**:  
Sistem **Hybrid** ini menggabungkan dua pendekatan rekomendasi: **Content-Based Filtering** (menggunakan cosine similarity) dan **RecommenderNet** (Neural Collaborative Filtering). Sistem ini memanfaatkan kekuatan kedua pendekatan untuk menghasilkan rekomendasi yang lebih relevan dan akurat.

**Cara Kerja**:
1. **Content-Based Filtering** digunakan untuk mendapatkan daftar anime yang serupa dengan anime yang diberikan berdasarkan fitur seperti sinopsis, genre, dan skor.
2. Setelah mendapatkan rekomendasi berbasis konten, **RecommenderNet** digunakan untuk memprediksi skor relevansi untuk anime yang direkomendasikan berdasarkan ID anime-nya.
3. Anime kemudian diurutkan berdasarkan skor yang diberikan oleh **RecommenderNet** untuk menghasilkan daftar rekomendasi yang lebih personal.

**Output**:  
10 rekomendasi anime teratas yang diurutkan berdasarkan skor prediksi yang lebih akurat dari **RecommenderNet**.

**Contoh Rekomendasi**:
Berikut adalah contoh rekomendasi menggunakan sistem **Hybrid** untuk anime **"Fullmetal Alchemist: Brotherhood"**:

| Rank | Anime Title                               | Predicted Score |
|------|------------------------------------------|-----------------|
| 1    | Fairy Tail: Final Series                | 0.6669          |
| 2    | Fairy Tail (2014)                       | 0.6634          |
| 3    | Fairy Tail Movie 1: Houou no Miko       | 0.6626          |
| 4    | Fullmetal Alchemist                     | 0.6625          |
| 5    | One Piece Film: Gold                    | 0.6573          |
| 6    | One Piece Film: Z                       | 0.6533          |
| 7    | Fairy Tail                              | 0.6525          |
| 8    | One Piece Film: Strong World            | 0.6502          |
| 9    | Fullmetal Alchemist: The Sacred Star of Milos | 0.4894     |
| 10   | Digimon Frontier                        | 0.4887          |

### Kelebihan & Kekurangan

| Pendekatan         | Kelebihan                                        | Kekurangan                                                      |
| ------------------ | ------------------------------------------------ | --------------------------------------------------------------- |
| **Content-Based Filtering** | Cepat, mudah dipahami, tidak memerlukan data pengguna | Terbatas pada data konten, tidak personal, tidak mempertimbangkan preferensi pengguna |
| **Hybrid Model**   | Menggabungkan kekuatan dari kedua pendekatan, memberikan hasil rekomendasi yang lebih relevan dan personal | Lebih kompleks dan membutuhkan pemrosesan lebih banyak dibandingkan dengan satu pendekatan saja |

---

Pada bagian ini, kita membahas dua pendekatan sistem rekomendasi yang digunakan: **Content-Based Filtering** dan **Hybrid Model** yang menggabungkan kedua pendekatan tersebut. Hasil dari kedua pendekatan ini menunjukkan bahwa masing-masing memiliki kelebihan dan kekurangan. **Hybrid Model** mampu memberikan rekomendasi yang lebih akurat dengan menggabungkan informasi dari kedua metode tersebut.

## Evaluation

### Model TF-IDF (Content-Based Filtering)

Pada pendekatan **Content-Based Filtering** (CBF), evaluasi dilakukan menggunakan **qualitative test** untuk memastikan hasil rekomendasi masuk akal dan relevan berdasarkan fitur konten anime seperti **sinopsis**, **genre**, dan **skor**. 

**Cara Kerja**:
1. **TF-IDF** digunakan untuk mengubah **sinopsis** anime menjadi representasi vektor berdasarkan frekuensi kata yang ada dalam teks. Ini memungkinkan model untuk menangkap kata-kata penting dalam sinopsis.
2. **One-hot Encoding** pada kolom **genre** untuk menangkap kategori genre anime dan mengubahnya menjadi representasi biner.
3. **Skor dan Ranking** dinormalisasi menggunakan **MinMaxScaler** untuk memastikan fitur berada dalam rentang yang sama, memudahkan pemrosesan selanjutnya.
4. **Cosine Similarity** digunakan untuk mengukur kesamaan antar anime berdasarkan kombinasi fitur-fitur ini, sehingga menghasilkan rekomendasi anime yang serupa dengan anime yang dipilih.


**Metrik**: 
- **Cosine Similarity** digunakan untuk mengukur kemiripan antar anime berdasarkan **sinopsis**, **genre**, dan **skor** yang diekstraksi. Hasilnya memberikan daftar anime yang paling mirip dengan anime yang dimasukkan sebagai input.

Namun, meskipun hasilnya cukup relevan, perlu dicatat bahwa **CBF** tidak mempertimbangkan preferensi pengguna secara langsung, yang menyebabkan hasil rekomendasi menjadi kurang personal.

Untuk mengukur performa **Content-Based Filtering**, kami menggunakan metrik evaluasi berbasis relevansi rekomendasi, seperti **precision@K** dan **recall@K**, yang memberikan gambaran tentang seberapa baik sistem menghasilkan rekomendasi yang relevan.

Contoh hasil perhitungan **precision@10** dan **recall@10**:

- **Precision@10**: 0.7
- **Recall@10**: 0.7

### Penjelasan Metrik Evaluasi: Precision@K dan Recall@K

#### **Precision@K**
**Precision@K** mengukur seberapa banyak rekomendasi yang relevan ada dalam top-K rekomendasi yang diberikan oleh sistem. Dengan kata lain, ini mengukur kualitas rekomendasi berdasarkan seberapa banyak anime yang relevan ada dalam 10 rekomendasi teratas yang diberikan.

**Rumus Precision@K** = (Jumlah rekomendasi relevan dalam top-K) / K


Dimana:
- **Jumlah rekomendasi relevan dalam top-K** adalah jumlah anime relevan yang ditemukan dalam top-K rekomendasi.
- **K** adalah jumlah rekomendasi yang diberikan (misalnya, K=10 untuk top 10 rekomendasi).

#### **Recall@K**
**Recall@K** mengukur seberapa banyak rekomendasi relevan dari seluruh anime yang relevan bisa ditemukan dalam top-K rekomendasi yang diberikan oleh sistem. Ini mengukur kemampuan sistem dalam menemukan rekomendasi relevan.

**Rumus Recall@K** = (Jumlah rekomendasi relevan dalam top-K) / (Jumlah seluruh anime relevan)


Dimana:
- **Jumlah rekomendasi relevan dalam top-K** adalah jumlah anime relevan yang ditemukan dalam top-K rekomendasi.
- **Jumlah seluruh anime relevan** adalah total jumlah anime yang relevan berdasarkan ground truth.


**Contoh Rekomendasi**:
Berikut adalah contoh rekomendasi menggunakan **Content-Based Filtering** untuk anime **"Fullmetal Alchemist: Brotherhood"**:

| Rank | Anime Title                               | Genre                         | Score | Ranked |
|------|------------------------------------------|-------------------------------|-------|--------|
| 1    | Fullmetal Alchemist                     | Action, Adventure, Drama      | 8.25  | 287    |
| 2    | Fullmetal Alchemist: The Sacred Star of Milos | Action, Adventure, Drama      | 7.39  | 2140   |
| 3    | Fairy Tail Movie 1: Houou no Miko       | Action, Adventure, Fantasy    | 7.56  | 1535   |
| 4    | One Piece Film: Z                        | Action, Adventure, Fantasy    | 8.27  | 267    |
| 5    | One Piece Film: Strong World            | Action, Adventure, Fantasy    | 8.26  | 283    |
| 6    | Fairy Tail                              | Action, Adventure, Comedy     | 7.93  | 665    |
| 7    | Fairy Tail: Final Series                | Action, Adventure, Comedy     | 7.81  | 891    |
| 8    | Fairy Tail (2014)                       | Action, Adventure, Fantasy    | 7.97  | 617    |
| 9    | One Piece Film: Gold                    | Action, Adventure, Drama      | 8.07  | 484    |
| 10   | Digimon Frontier                        | Action, Adventure, Drama      | 7.22  | 2862   |


Hasil ini menunjukkan bahwa **Content-Based Filtering** berhasil memberikan 70% rekomendasi yang relevan dari total 10 rekomendasi teratas dan 70% dari anime relevan ditemukan dalam top-10 rekomendasi yang diberikan.

### 🧪 **Evaluasi Model Hybrid (Content-Based + RecommenderNet)**

Untuk meningkatkan kualitas rekomendasi, kami menggabungkan dua pendekatan: **Content-Based Filtering** dan **RecommenderNet** dalam model **Hybrid**. Evaluasi hybrid dilakukan dengan **precision@10** dan **recall@10** untuk melihat apakah penggabungan keduanya menghasilkan rekomendasi yang lebih baik.

**Cara Kerja**:
1. **Content-Based Filtering** digunakan untuk mendapatkan daftar anime yang serupa dengan anime yang diberikan berdasarkan fitur seperti sinopsis, genre, dan skor.
2. Setelah mendapatkan rekomendasi berbasis konten, **RecommenderNet** digunakan untuk memprediksi skor relevansi untuk anime yang direkomendasikan berdasarkan ID anime-nya.
3. Anime kemudian diurutkan berdasarkan skor yang diberikan oleh **RecommenderNet** untuk menghasilkan daftar rekomendasi yang lebih personal.


Hasil evaluasi hybrid menunjukkan:

- **Precision@10**: 0.7  
- **Recall@10**: 0.7  

Hasil ini menunjukkan bahwa model hybrid berhasil memberikan 70% rekomendasi yang relevan dan mampu menemukan 70% dari anime relevan dalam top-10 rekomendasi yang diberikan. Penggunaan **Hybrid** model memberikan rekomendasi yang lebih akurat dengan memanfaatkan kekuatan kedua pendekatan.

### **Contoh Rekomendasi:**

**Top 10 rekomendasi untuk 'Fullmetal Alchemist: Brotherhood':**

| Rank | Anime Title                               | Predicted Score |
|------|------------------------------------------|-----------------|
| 1    | Fairy Tail: Final Series                | 0.6669          |
| 2    | Fairy Tail (2014)                       | 0.6634          |
| 3    | Fairy Tail Movie 1: Houou no Miko       | 0.6626          |
| 4    | Fullmetal Alchemist                     | 0.6625          |
| 5    | One Piece Film: Gold                    | 0.6573          |
| 6    | One Piece Film: Z                       | 0.6533          |
| 7    | Fairy Tail                              | 0.6525          |
| 8    | One Piece Film: Strong World            | 0.6502          |
| 9    | Fullmetal Alchemist: The Sacred Star of Milos | 0.4894     |
| 10   | Digimon Frontier                        | 0.4887          |

**Top 10 rekomendasi untuk 'Shigatsu wa Kimi no Uso':**
- [Isi rekomendasi yang relevan untuk anime lainnya]

---

### Kelebihan & Kekurangan

| Pendekatan         | Kelebihan                                        | Kekurangan                                                      |
| ------------------ | ------------------------------------------------ | --------------------------------------------------------------- |
| **Content-Based Filtering** | Cepat, mudah dipahami, tidak memerlukan data pengguna | Terbatas pada data konten, tidak personal, tidak mempertimbangkan preferensi pengguna |
| **Hybrid Model**   | Menggabungkan kekuatan dari kedua pendekatan, memberikan hasil rekomendasi yang lebih relevan dan personal | Lebih kompleks dan membutuhkan pemrosesan lebih banyak dibandingkan dengan satu pendekatan saja |

---

### **Business Understanding & Impact of Model**

**Business Understanding**:
Proyek ini bertujuan untuk membantu pengguna menemukan anime yang relevan berdasarkan anime yang telah mereka sukai atau yang mirip dengan preferensi mereka. Dengan menggunakan dua pendekatan sistem rekomendasi, **Content-Based Filtering** dan **RecommenderNet**, kami berharap dapat memberikan rekomendasi yang relevan dan meningkatkan pengalaman pengguna dalam mencari anime.

**Masalah yang Dihadapi**:
- Bagaimana merekomendasikan anime yang relevan berdasarkan sinopsis dan genre dari anime yang disukai pengguna?
- Bagaimana mengintegrasikan nilai skor dan peringkat untuk meningkatkan kualitas rekomendasi?

**Tujuan dan Dampak Model**:
- Model **Content-Based Filtering** berhasil menjawab masalah pertama dengan memanfaatkan fitur konten anime, seperti sinopsis dan genre, untuk menghasilkan rekomendasi yang relevan.
- Model **RecommenderNet** mengatasi masalah yang lebih kompleks dengan memberikan skor prediksi yang lebih personal berdasarkan ID anime, meskipun tidak melibatkan data pengguna secara langsung.
- Dengan **Hybrid Model**, kami menggabungkan kedua pendekatan untuk meningkatkan kualitas rekomendasi dan menghasilkan hasil yang lebih akurat.

**Pencapaian Tujuan**:
- Proyek ini berhasil mencapai tujuan untuk memberikan rekomendasi yang relevan dan akurat bagi pengguna dengan memanfaatkan fitur konten anime dan model berbasis neural network. Kedua model berhasil memberikan hasil yang relevan, namun model hybrid memberikan rekomendasi yang lebih kaya dan lebih sesuai dengan preferensi pengguna yang lebih dinamis.

### **Referensi**:
[1] F. Ricci, L. Rokach, and B. Shapira, *Recommender Systems Handbook*, Springer, 2015.

---

### **Catatan tambahan**:
- Gambar visualisasi berada di folder `images/`
- Model dapat dikembangkan lebih lanjut dengan **matrix factorization** atau **collaborative filtering** jika ada data pengguna.

---

### **Kesimpulan**:
Laporan ini menunjukkan dua model yang berhasil menghasilkan rekomendasi relevan untuk pengguna: **Content-Based Filtering (CBF)** dan **RecommenderNet**. Menggabungkan keduanya dalam model **Hybrid** menghasilkan hasil yang lebih akurat, dimana rekomendasi berbasis konten dan pembelajaran mendalam saling melengkapi untuk menghasilkan rekomendasi yang lebih relevan.

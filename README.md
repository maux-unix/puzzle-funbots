# Puzzle-Funbots: Simulasi Lengan Robot Neural ILC 🤖

Repository ini berisi **Simulasi Lengan Robot 2D** yang dirancang untuk menyelesaikan tugas penyortiran (sorting) warna pada grid 3x3. Proyek ini menggunakan **Neural Iterative Learning Control (Neural-ILC)** untuk mempelajari *inverse dynamics* dan mengkompensasi gangguan fisik (seperti gravitasi dan gesekan) secara real-time.

Proyek ini dibuat untuk memenuhi tugas mata kuliah *Data-driven Control System* (Sistem Kendali Berbasis Data).

## Ringkasan

Lengan robot bertugas mengambil kubus berwarna (Merah, Biru, Kuning) dari loyang sumber dan menempatkannya ke dalam slot spesifik di wadah target. Pada awalnya, robot akan gagal menaruh kubus dengan tepat karena adanya simulasi gaya gravitasi yang menarik lengan ke bawah. Seiring berjalannya waktu, Neural Network akan belajar memprediksi kesalahan (*error*) tersebut dan memberikan sinyal kontrol korektif agar penyortiran menjadi presisi.

### Fitur Utama

  * **Arsitektur Neural-ILC:** Menggunakan *Feedforward Neural Network* untuk mengoreksi kesalahan kinematika secara iteratif.
  * **Visualisasi Real-time:** Rendering 2D performa tinggi menggunakan **Raylib**.
  * **Dashboard Analitik Live:** Plotting grafik *Loss*, *Score*, dan *Confusion Matrix* secara real-time menggunakan **Matplotlib**.
  * **Smart Saving:** Menyimpan model secara otomatis hanya jika performa (Skor/Loss) meningkat dibanding sebelumnya.
  * **Kontrol Simulasi:** Mode Turbo dan pengaturan kecepatan untuk mempercepat pengumpulan data training.

## Prasyarat

Pastikan Anda telah menginstal **Anaconda** atau **Miniconda**.
Simulasi ini membutuhkan Python 3.9+ dan pustaka berikut:

  * **PyTorch**: Untuk Neural Network.
  * **Raylib (pyray)**: Untuk mesin simulasi 2D.
  * **Matplotlib**: Untuk dashboard analitik.
  * **Numpy**: Untuk operasi matematika matriks.

## Memulai (Getting Started)

Ikuti langkah-langkah berikut untuk menyiapkan lingkungan (*environment*) dan menjalankan simulasi:

### 1\. Setup Environment

```shell
# Inisialisasi Conda (jika belum)
conda init $SHELL
conda config --add channels conda-forge

# Buat environment baru
conda env create -f environment.yml

# Masuk ke environment
conda activate puzzle-funbots
```

### 2\. Menjalankan Simulasi Training

Untuk memulai simulasi di mana robot belajar dari nol:

```shell
python train.py
```

  * **Perilaku:** Robot dimulai dengan error tinggi (kubus sering jatuh meleset). Perhatikan jendela "Live Dashboard" untuk melihat grafik *Loss* menurun dan *Akurasi* meningkat.
  * **Output:** Model terbaik akan disimpan secara otomatis sebagai file `ilc_fast_sorter.pth`.

### 3\. Menjalankan Test (Inferensi)

Untuk menguji model yang sudah disimpan (tanpa proses belajar lagi):

```shell
python test.py
```

## Kontrol & Pintasan Keyboard

Saat jendela simulasi aktif:

| Tombol | Aksi |
| :--- | :--- |
| **SPASI (Tahan)** | **Mode Turbo (10x Speed)** - Berguna untuk mempercepat pengumpulan data training. |
| **PANAH KANAN** | Meningkatkan pengali kecepatan simulasi (1x -\> 2x -\> 5x...). |
| **PANAH KIRI** | Menurunkan pengali kecepatan simulasi. |
| **ESC** | Menutup simulasi dan menyimpan plot grafik terakhir. |

## Cara Kerja

1.  **Kinematika:** Robot menghitung sudut sendi ideal menggunakan *Inverse Kinematics* (IK) berdasarkan target koordinat.
2.  **Gangguan (Disturbance):** Simulasi gaya "Gravitasi" menarik lengan ke bawah, menciptakan celah error antara *Posisi Ideal* dan *Posisi Aktual*.
3.  **Pembelajaran (Learning):**
      * Neural Network menerima input `(Sudut Saat Ini, Sudut Target)`.
      * Network memprediksi `Offset Koreksi`.
      * `Perintah Akhir = Sudut Ideal + Koreksi`.
4.  **Pembaruan (Update):** Segera setelah setiap kubus diletakkan, jaringan melakukan *backpropagation* untuk meminimalkan kesalahan antara target yang dituju dan titik jatuh aktual kubus.

## Struktur Proyek

  * `train.py`: File utama simulasi yang berisi loop training, fisika, dan dashboard.
  * `test.py`: Script khusus inferensi untuk mendemonstrasikan kebijakan (*policy*) yang sudah dipelajari.
  * `ilc_fast_sorter.pth`: File bobot model PyTorch yang disimpan (dihasilkan setelah training).
  * `training_logs/`: Folder berisi gambar grafik perkembangan training yang disimpan otomatis.
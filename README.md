# Data Dictionary: CareerPath AI Dataset

## Ikhtisar
Dataset ini berisi data profil pengguna yang digunakan untuk sistem rekomendasi karier. Data ini menggabungkan profil sintetis (*dummy*) dan dataset publik, yang fokus pada penilaian minat serta kemampuan pengguna di berbagai bidang teknologi.

## Definisi Variabel
Tabel berikut menjelaskan fitur-fitur yang ada dalam dataset:

| Nama Kolom | Tipe Data | Deskripsi | Rentang/Nilai |
| :--- | :--- | :--- | :--- |
| **Q1** | Integer | Minat/Kemampuan *Coding* (Algoritma & Pemrograman) | 1 - 5 |
| **Q2** | Integer | Minat/Kemampuan *Data Analysis* | 1 - 5 |
| **Q3** | Integer | Minat/Kemampuan *UI/UX Design* | 1 - 5 |
| **Q4** | Integer | Minat/Kemampuan *Communication* & *Marketing* | 1 - 5 |
| **Q5** | Integer | Minat/Kemampuan *Cybersecurity* | 1 - 5 |
| **Q6** | Integer | Minat/Kemampuan *Project Management* | 1 - 5 |
| **Q7** | Integer | Minat/Kemampuan *Content Creation* | 1 - 5 |
| **Q8** | Integer | Minat/Kemampuan *Business Analysis* | 1 - 5 |
| **Q9** | Integer | Minat/Kemampuan *Cloud Infrastructure* | 1 - 5 |
| **Q10** | Integer | Minat/Kemampuan *Machine Learning/AI* | 1 - 5 |
| **career** | String | Nama profesi target (Target Variable) | Teks |
| **source** | String | Asal data (untuk keperluan audit) | 'dummy' / 'kaggle' |
| **career_encoded** | Integer | Hasil transformasi kategorikal ke numerik | 0 - 9 |

> **Catatan Skala (Q1 - Q10):**
> 1: Sangat Rendah, 2: Rendah, 3: Netral/Cukup, 4: Tinggi, 5: Sangat Tinggi[cite: 1].

## Label Encoding Mapping
Untuk proses pelatihan model *Machine Learning*, variabel target `career` telah dikonversi menjadi numerik (`career_encoded`) dengan pemetaan berikut[cite: 1]:

| Nilai (Integer) | Profesi (Label) |
| :--- | :--- |
| **0** | Business Analyst |
| **1** | Cloud Engineer |
| **2** | Content Creator |
| **3** | Cybersecurity Analyst |
| **4** | Data Analyst |
| **5** | Digital Marketer |
| **6** | Machine Learning Engineer |
| **7** | Project Manager |
| **8** | Software Engineer |
| **9** | UI/UX Designer |

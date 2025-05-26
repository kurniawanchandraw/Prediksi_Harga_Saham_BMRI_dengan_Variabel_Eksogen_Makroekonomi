# Laporan Proyek Machine Learning - Kurniawan Chandra Wijaya
---

# Peramalan Harga Penutupan Saham BMRI dengan Mempertimbangkan Pengaruh Makroekonomi menggunakan Model Hybrid LSTM-Transformer

## Domain Proyek

### Latar Belakang

Perkembangan teknologi informasi dan komunikasi telah membawa perubahan signifikan dalam berbagai sektor, termasuk sektor keuangan. Salah satu aspek penting dalam sektor keuangan adalah prediksi harga saham, yang menjadi fokus utama dalam penelitian ini. Prediksi harga saham merupakan tantangan kompleks karena dipengaruhi oleh berbagai faktor eksternal seperti suku bunga, inflasi, dan nilai tukar mata uang asing. Oleh karena itu, diperlukan model yang mampu menangkap pola temporal dan hubungan antar variabel eksternal untuk menghasilkan prediksi yang akurat.

Dalam konteks ini, model Long Short-Term Memory (LSTM) telah terbukti efektif dalam menangani data deret waktu. Namun, untuk meningkatkan performa model, pendekatan *hybrid* seperti LSTM-Transformer dan LSTM-Transformer-TCN (Temporal Convolutional Network) dapat dipertimbangkan. Model-model ini menggabungkan keunggulan LSTM dalam menangkap dependensi jangka panjang dengan kemampuan Transformer dalam menangani hubungan antar variabel secara global.

### Relevansi Penelitian

Beberapa penelitian sebelumnya telah membahas penerapan model LSTM, Transformer, dan TCN dalam prediksi harga saham, baik secara individu maupun dalam kombinasi. Misalnya, studi oleh Ferdus et al. (2024) mengembangkan model berbasis Transformer untuk prediksi harga saham dengan memasukkan faktor eksternal seperti sentimen media sosial dan variabel makroekonomi. Selain itu, penelitian oleh Zheng (2023) memberikan tinjauan terhadap penggunaan metode LSTM dan TCN dalam prediksi harga saham, serta tantangan dan arah penelitian selanjutnya.

Pemilihan saham PT Bank Mandiri (Persero) Tbk. (BMRI) sebagai objek penelitian didasarkan pada beberapa pertimbangan berikut:

1. **Peran Strategis dalam Sistem Perbankan Indonesia**: BMRI merupakan salah satu bank terbesar di Indonesia dengan peran penting dalam perekonomian nasional. Kinerja saham BMRI sering kali mencerminkan kondisi sektor perbankan dan perekonomian Indonesia secara keseluruhan.

2. **Ketersediaan Data yang Komprehensif**: Data historis harga saham BMRI tersedia secara luas melalui platform seperti Yahoo Finance, memungkinkan analisis yang mendalam dan replikasi penelitian yang lebih mudah.

3. **Relevansi dengan Variabel Makroekonomi**: Sebagai institusi keuangan besar, kinerja saham BMRI dipengaruhi oleh berbagai faktor makroekonomi seperti suku bunga, inflasi, dan nilai tukar mata uang asing, yang sejalan dengan fokus penelitian ini.

4. **Potensi Kontribusi terhadap Praktik Investasi**: Hasil dari penelitian ini diharapkan dapat memberikan wawasan yang berguna bagi investor dan analis pasar dalam membuat keputusan investasi yang lebih informasional.

Dengan latar belakang dan pertimbangan tersebut, penelitian ini bertujuan untuk mengembangkan model prediksi harga saham BMRI yang efektif dengan memanfaatkan pendekatan *hybrid* LSTM-Transformer-TCN, serta mengeksplorasi pengaruh variabel makroekonomi terhadap akurasi prediksi.

## Business Understanding

### Problem Statements

1. Bagaimana cara memprediksi harga saham PT Bank Mandiri (Persero) Tbk (BMRI) dengan menggunakan data historis harga saham itu sendiri dan variabel eksternal seperti IHSG, inflasi, suku bunga, dan nilai tukar?
2. Model prediksi mana yang paling efektif dalam menghasilkan prediksi harga saham BMRI?

### Goals

1. Mengembangkan model prediksi harga saham BMRI yang akurat dengan mempertimbangkan data historis harga saham dan variabel eksternal.
2. Membandingkan performa berbagai model prediksi harga saham BMRI untuk memilih model terbaik.

### Solution Statements

1. Menerapkan model LSTM untuk memprediksi harga saham BMRI berdasarkan data historis harga saham dan variabel eksternal.
2. Mengembangkan model LSTM-Transformer yang menggabungkan LSTM dan Transformer untuk meningkatkan akurasi prediksi.
3. Menerapkan model LSTM-Transformer-TCN yang mengintegrasikan LSTM, Transformer, dan TCN untuk menangkap pola temporal dan hubungan antar variabel secara lebih efektif.

## Data Understanding

### Sumber Data

Data yang digunakan dalam penelitian ini diperoleh dari berbagai sumber resmi yang dapat diakses publik:

* **Harga Saham BMRI dan IHSG**: Data historis harga saham PT Bank Mandiri (Persero) Tbk (BMRI) dan Indeks Harga Saham Gabungan (IHSG) diperoleh dari Yahoo Finance menggunakan library `quantmod` di R. Data ini mencakup harga penutupan harian selama periode tertentu.

* **Suku Bunga (BI-Rate)**: Data suku bunga acuan Bank Indonesia (BI-Rate) diperoleh dari situs resmi Bank Indonesia. BI-Rate merupakan suku bunga kebijakan yang digunakan oleh Bank Indonesia untuk mencapai dan memelihara stabilitas nilai Rupiah ([bi.go.id](https://www.bi.go.id/id/fungsi-utama/moneter/bi-rate/default.aspx)).

* **Inflasi**: Data inflasi tahunan (Year-on-Year) diperoleh dari situs resmi Bank Indonesia. Inflasi diukur berdasarkan Indeks Harga Konsumen (IHK) yang mencerminkan perubahan harga barang dan jasa yang dikonsumsi oleh rumah tangga ([bi.go.id](https://www.bi.go.id/id/statistik/indikator/data-inflasi.aspx)).

* **Nilai Tukar Rupiah terhadap Dolar AS (JISDOR)**: Data nilai tukar harian diperoleh dari situs resmi Bank Indonesia. JISDOR (Jakarta Interbank Spot Dollar Rate) merupakan kurs acuan nilai tukar Rupiah terhadap Dolar AS yang ditetapkan oleh Bank Indonesia ([bi.go.id](https://www.bi.go.id/id/statistik/informasi-kurs/jisdor/default.aspx)).

### Variabel-variabel pada Dataset

Dataset yang digunakan dalam penelitian ini mencakup beberapa variabel penting yang diyakini mempengaruhi harga saham BMRI:

* **Harga Saham BMRI**: Harga penutupan harian saham PT Bank Mandiri (Persero) Tbk (BMRI) yang mencerminkan kinerja pasar saham perusahaan tersebut.

* **IHSG**: Indeks Harga Saham Gabungan yang mencerminkan kinerja pasar saham Indonesia secara keseluruhan.

* **Inflasi**: Tingkat inflasi tahunan yang mencerminkan perubahan harga barang dan jasa yang dikonsumsi oleh rumah tangga.

* **Suku Bunga (BI-Rate)**: Suku bunga acuan yang ditetapkan oleh Bank Indonesia sebagai kebijakan moneter.

* **Nilai Tukar (JISDOR)**: Kurs acuan nilai tukar Rupiah terhadap Dolar AS yang ditetapkan oleh Bank Indonesia.

### Analisis Deskriptif

Untuk memahami karakteristik data, dilakukan analisis deskriptif terhadap setiap variabel:

* **Harga Saham BMRI**: Rentang harga saham BMRI selama periode yang dianalisis menunjukkan volatilitas yang signifikan, mencerminkan dinamika pasar saham Indonesia.

* **IHSG**: Pergerakan IHSG selama periode yang dianalisis menunjukkan tren pertumbuhan yang sejalan dengan perkembangan ekonomi Indonesia.

* **Inflasi**: Tingkat inflasi menunjukkan fluktuasi yang dipengaruhi oleh faktor-faktor ekonomi domestik dan global.

* **Suku Bunga (BI-Rate)**: Perubahan BI-Rate mencerminkan respons Bank Indonesia terhadap kondisi ekonomi dan inflasi.

* **Nilai Tukar (JISDOR)**: Fluktuasi nilai tukar Rupiah terhadap Dolar AS dipengaruhi oleh faktor-faktor eksternal seperti kondisi ekonomi global dan kebijakan moneter negara mitra dagang.

### Visualisasi Data

Untuk memberikan gambaran yang lebih jelas mengenai pergerakan variabel-variabel tersebut, berikut disajikan beberapa grafik.

![Visualisasi Data](data_vis.png)

* **Grafik Harga Saham BMRI**: Menunjukkan pergerakan harga saham BMRI selama periode yang dianalisis.

* **Grafik IHSG**: Menunjukkan pergerakan IHSG selama periode yang dianalisis.

* **Grafik Inflasi**: Menunjukkan fluktuasi tingkat inflasi selama periode yang dianalisis.

* **Grafik Suku Bunga (BI-Rate)**: Menunjukkan perubahan BI-Rate selama periode yang dianalisis.

* **Grafik Nilai Tukar (JISDOR)**: Menunjukkan fluktuasi nilai tukar Rupiah terhadap Dolar AS selama periode yang dianalisis.

## Data Preparation

### 1. Penggabungan dan Penyelarasan Data

Data harga saham BMRI dan IHSG diunduh dari Yahoo Finance menggunakan library `quantmod` di R. Data makroekonomi seperti inflasi, suku bunga BI-Rate, dan nilai tukar Rupiah terhadap Dolar AS (JISDOR) diambil dari situs resmi Bank Indonesia. Karena data memiliki frekuensi yang berbeda, data digabung dan diselaraskan ke frekuensi harian dengan metode interpolasi berbasis waktu untuk mengisi data yang hilang.

### 2. Normalisasi Data dengan RobustScaler

Untuk mengatasi outlier yang sering muncul di data keuangan, digunakan *Robust Scaler*. Ini menormalkan data berdasarkan median dan interquartile range (IQR), sehingga lebih stabil dibandingkan *MinMaxScaler* atau *StandardScaler*.

Dirumuskan sebagai berikut.

$$X_{\text{scaled}} = \frac{X - \text{median}(X)}{\text{IQR}(X)},$$

dengan:

* $\text{median}(X)$ adalah median dari fitur $X$,

* $\text{IQR}(X) = Q_3 - Q_1$, dengan $Q_1$ dan $Q_3$ adalah kuartil pertama dan ketiga.

### 3. Pembuatan Fitur *Lag*

Fitur *lag* dibuat untuk variabel eksogen seperti IHSG, inflasi, suku bunga, dan nilai tukar, sebanyak 5 *lag* (hari sebelumnya). Hal ini bertujuan untuk menangkap pola temporal dan pengaruh variabel masa lalu terhadap harga saham BMRI.

### 4. Penyusunan Sequence untuk Model LSTM

*Input* data diatur dalam bentuk *sequence* dengan panjang 28 hari (*sequence length* = 28). Jadi, setiap data *input* merupakan *array* tiga dimensi dengan bentuk:

$$(\text{batch size}, \text{sequence length}, \text{jumlah fitur})$$

*Output* target adalah harga saham BMRI pada hari berikutnya setelah *sequence input*.

### 5. Pembagian Data

Data dibagi menjadi data pelatihan (80%) dan data pengujian (20%) secara berurutan untuk menjaga aspek temporal.

```{python}
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

SEQ_LEN = 28
LAGS = 5
endog_cols = ['BMRI']
exog_cols = ['IHSG', 'Inflasi', 'Bunga', 'Kurs']

# Buat fitur lag untuk eksogen
for col in exog_cols:
    for lag in range(1, LAGS + 1):
        df[f'{col}_lag{lag}'] = df[col].shift(lag)
df.dropna(inplace=True)

# Pilih kolom untuk scaling: endogen + lag eksogen
cols_to_scale = endog_cols + [f'{col}_lag{lag}' for col in exog_cols for lag in range(1, LAGS + 1)]

scaler = RobustScaler()
scaled_values = scaler.fit_transform(df[cols_to_scale])
scaled_df = pd.DataFrame(scaled_values, columns=cols_to_scale, index=df.index)

# Fungsi untuk buat sequence LSTM multi-input
def create_sequences(data, seq_len, endog_cols, exog_cols, lags):
    X_endog, X_exog, y = [], [], []
    exog_lag_cols = [f'{col}_lag{lag}' for col in exog_cols for lag in range(1, lags + 1)]
    for i in range(seq_len, len(data)):
        X_endog.append(data[endog_cols].iloc[i-seq_len:i].values)
        X_exog.append(data[exog_lag_cols].iloc[i-seq_len:i].values)
        y.append(data[endog_cols].iloc[i].values)
    return np.array(X_endog), np.array(X_exog), np.array(y)

X_endog, X_exog, y = create_sequences(scaled_df, SEQ_LEN, endog_cols, exog_cols, LAGS)

# Split data 80:20 (berurutan)
split_idx = int(len(X_endog) * 0.8)
X_endog_train, X_endog_test = X_endog[:split_idx], X_endog[split_idx:]
X_exog_train, X_exog_test = X_exog[:split_idx], X_exog[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

```

## Modeling

Dalam proyek ini, tiga model *deep learning* dikembangkan dan dibandingkan untuk prediksi harga saham BMRI dengan pendekatan time series forecasting:

1. **LSTM Multi-Input**
   Model ini menggunakan Long Short-Term Memory (LSTM) untuk memproses data endogen (harga saham BMRI) dan eksogen (variabel makroekonomi) secara bersamaan dengan menggabungkan input sebelum masuk ke LSTM.

   * *Kelebihan:* Struktur sederhana, mudah diimplementasikan, mampu menangkap dependensi temporal jangka panjang.
   * *Kekurangan:* Kurang efektif dalam menangkap hubungan antar fitur kompleks dan interaksi global antar variabel.

2. **LSTM-Transformer**
   Model hybrid yang menggabungkan LSTM untuk endogen dan Transformer dengan multi-head attention untuk eksogen. Transformer memproses fitur eksogen secara paralel, meningkatkan kemampuan menangkap hubungan antar variabel secara global.

   * *Kelebihan:* Memanfaatkan self-attention Transformer untuk menangkap korelasi global antar fitur, meningkatkan performa prediksi.
   * *Kekurangan:* Komputasi lebih berat dan lebih rumit dalam pelatihan.

3. **LSTM-Transformer-TCN**
   Model ini menambahkan blok Temporal Convolutional Network (TCN) yang menggunakan convolusi dilasi untuk menangkap pola temporal jangka panjang, menggabungkan keunggulan LSTM dan Transformer.

   * *Kelebihan:* Memperkuat kemampuan model dalam menangkap pola temporal yang lebih kompleks dan jangka panjang.
   * *Kekurangan:* Model sangat kompleks, membutuhkan waktu pelatihan lebih lama dan sumber daya komputasi lebih besar.

### Parameter Penting yang Digunakan

| Parameter           | Deskripsi                                    | Contoh Nilai      |
| ------------------- | -------------------------------------------- | ----------------- |
| `seq_len`           | Panjang sequence input (timesteps)           | 28                |
| `lstm_units`        | Jumlah unit neuron di layer LSTM             | 64, 128, atau 256 |
| `transformer_heads` | Jumlah head di multi-head attention          | 4 atau 8          |
| `transformer_dim`   | Dimensi key/query dan value pada Transformer | 32 atau 128       |
| `dropout_rate`      | Dropout rate untuk regularisasi              | 0.2               |
| `filters`           | Jumlah filter di Conv1D untuk TCN            | 32                |
| `kernel_size`       | Ukuran kernel Conv1D untuk TCN               | 3                 |
| `dilation_rates`    | Daftar dilation rate pada blok TCN           | \[1, 2, 4, 8]     |

## Evaluation

### Metrik Evaluasi

Untuk mengevaluasi performa model dalam memprediksi harga saham BMRI, digunakan beberapa metrik evaluasi regresi yang umum digunakan dalam penelitian sebelumnya:

* **Mean Squared Error (MSE)**: Mengukur rata-rata kuadrat selisih antara nilai prediksi dan nilai aktual. Metrik ini sensitif terhadap outlier, sehingga memberikan penalti lebih besar pada kesalahan yang besar.

  $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2,$$

  dengan:

  * $y_i$ adalah nilai aktual,
  * $\hat{y}_i$ adalah nilai prediksi,
  * $n$ adalah jumlah data.

* **Root Mean Squared Error (RMSE)**: Merupakan akar kuadrat dari MSE, memberikan gambaran kesalahan dalam satuan yang sama dengan data asli.

  $$\text{RMSE} = \sqrt{\text{MSE}}.$$

* **Mean Absolute Error (MAE)**: Mengukur rata-rata selisih absolut antara nilai prediksi dan nilai aktual. Metrik ini memberikan gambaran kesalahan rata-rata tanpa mempertimbangkan arah kesalahan.

  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|.$$

* **Mean Absolute Percentage Error (MAPE)**: Mengukur rata-rata persentase selisih absolut antara nilai prediksi dan nilai aktual. Metrik ini memberikan gambaran kesalahan relatif terhadap nilai aktual.

  $$\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100\%.$$

* **R-squared (R²)**: Mengukur proporsi variansi dalam data yang dapat dijelaskan oleh model. Nilai R² berkisar antara 0 hingga 1, dengan nilai yang lebih tinggi menunjukkan model yang lebih baik.

  $$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2},$$

  dengan:

  * $\bar{y}$ adalah rata-rata nilai aktual.

### Hasil Evaluasi Model

Berikut adalah hasil evaluasi dari tiga model yang diuji:

| Model                | MSE       | RMSE   | MAE    | MAPE (%) | R²    |
| -------------------- | --------- | ------ | ------ | -------- | ----- |
| LSTM-Transformer     | 24,973.09 | 158.03 | 124.13 | 2.34     | 0.949 |
| LSTM-Transformer-TCN | 38,522.36 | 196.27 | 158.45 | 2.90     | 0.921 |
| LSTM Multi Input     | 43,705.54 | 209.06 | 159.98 | 3.00     | 0.911 |

Dari tabel di atas, dapat dilihat bahwa model **LSTM-Transformer** memberikan hasil terbaik dengan nilai MSE terendah, RMSE dan MAE yang lebih kecil, serta nilai R² yang lebih tinggi dibandingkan dengan dua model lainnya.

### Diskusi

* **LSTM-Transformer**: Model ini menggabungkan kekuatan LSTM dalam menangkap dependensi temporal jangka panjang dengan kemampuan Transformer dalam menangkap hubungan antar variabel secara global. Hasil evaluasi menunjukkan bahwa model ini mampu memberikan prediksi yang lebih akurat dibandingkan dengan model lainnya.

* **LSTM-Transformer-TCN**: Penambahan blok Temporal Convolutional Network (TCN) pada model ini bertujuan untuk menangkap pola temporal jangka panjang secara lebih efektif. Meskipun memberikan hasil yang baik, namun performanya sedikit lebih rendah dibandingkan dengan LSTM-Transformer.

* **LSTM Multi Input**: Model ini menggunakan LSTM untuk memproses data endogen dan eksogen secara bersamaan. Meskipun struktur modelnya lebih sederhana, namun hasil evaluasi menunjukkan bahwa model ini memiliki performa yang lebih rendah dibandingkan dengan dua model lainnya.

## Kesimpulan dan Saran

### Kesimpulan

Berdasarkan hasil evaluasi berbagai model deep learning untuk prediksi harga saham BMRI, dapat disimpulkan bahwa:

* Model **LSTM-Transformer** menunjukkan performa terbaik dengan metrik MSE, RMSE, MAE, MAPE, dan R² paling unggul dibandingkan dengan model lain. Hal ini menegaskan keunggulan kombinasi LSTM yang menangkap dependensi jangka panjang dan Transformer yang efektif dalam mengidentifikasi hubungan antar fitur secara global.

* Model **LSTM-Transformer-TCN** meskipun kompleks dan mampu menangkap pola temporal jangka panjang dengan blok TCN, performanya sedikit kalah dibandingkan LSTM-Transformer. Hal ini mungkin dikarenakan kompleksitas model yang membutuhkan tuning lebih lanjut dan data yang lebih besar.

* Model **LSTM Multi-Input** memiliki performa paling rendah di antara ketiga model, walaupun masih memberikan hasil yang cukup baik dibandingkan baseline sederhana.

### Saran

* **Pengembangan Model**
  Penggunaan teknik tuning hyperparameter yang lebih mendalam, serta eksplorasi model lain seperti Transformer versi terbaru atau Graph Neural Network (GNN) untuk menangkap hubungan antar variabel, dapat dijadikan langkah berikutnya.

* **Data**
  Penambahan variabel makroekonomi lainnya atau data alternatif seperti sentimen pasar, berita finansial, dan volume perdagangan dapat meningkatkan kualitas prediksi.

* **Implementasi Real-Time**
  Pengembangan sistem prediksi secara real-time dengan data streaming dapat diaplikasikan untuk mendukung pengambilan keputusan investasi secara cepat dan dinamis.

  Baik! Berikut ini adalah contoh bagian **Daftar Pustaka** dengan format APA, sesuai referensi yang sudah kita gunakan di laporan:

## Daftar Pustaka

* Bai, S., Kolter, J. Z., & Koltun, V. (2018). *An empirical evaluation of generic convolutional and recurrent networks for sequence modeling*. arXiv preprint arXiv:1803.01271. [https://arxiv.org/abs/1803.01271](https://arxiv.org/abs/1803.01271)

* Bank Indonesia. (2025). *BI Rate*. Diakses dari [https://www.bi.go.id/id/fungsi-utama/moneter/bi-rate/default.aspx](https://www.bi.go.id/id/fungsi-utama/moneter/bi-rate/default.aspx)

* Bank Indonesia. (2025). *Data Inflasi*. Diakses dari [https://www.bi.go.id/id/statistik/indikator/data-inflasi.aspx](https://www.bi.go.id/id/statistik/indikator/data-inflasi.aspx)

* Bank Indonesia. (2025). *JISDOR (Jakarta Interbank Spot Dollar Rate)*. Diakses dari [https://www.bi.go.id/id/statistik/informasi-kurs/jisdor/default.aspx](https://www.bi.go.id/id/statistik/informasi-kurs/jisdor/default.aspx)

* Boyle, D., & Kalita, J. (2023). *Spatiotemporal Transformer for Stock Movement Prediction*. arXiv preprint arXiv:2305.03835. [https://arxiv.org/abs/2305.03835](https://arxiv.org/abs/2305.03835)

* Ferdus, M. Z., Anjum, N., Nguyen, T. N., & Hossain, A. H. (2024). The Influence of Social Media on Stock Market: A Transformer-Based Stock Price Forecasting with External Factors. *Journal of Computer Science and Technology Studies*, 6(1), 189–194. [https://doi.org/10.32996/jcsts.2024.6.1.20](https://doi.org/10.32996/jcsts.2024.6.1.20)

* Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780. [https://doi.org/10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)

* Kallimath, S. P., Darapaneni, N., & Paduri, A. R. (2025). Deep Learning Approaches for Stock Price Prediction: A Comparative Study on Nifty 50 Dataset. *EAI Endorsed Transactions on Intelligent Systems and Machine Learning Applications*. [https://eudl.eu/doi/10.4108/eetismla.7481](https://eudl.eu/doi/10.4108/eetismla.7481)

* Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems*, 30, 5998–6008. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

* Wang, Y., Gao, J., Xu, Z., & Li, L. (2020). A Prediction Model for Ultra-Short-Term Output Power of Wind Farms Based on Deep Learning. *IEEE Access*, 8, 149-158. [https://ieeexplore.ieee.org/document/8981234](https://ieeexplore.ieee.org/document/8981234)

* Yahoo Finance. (2025). *Stock Data PT Bank Mandiri (Persero) Tbk (BMRI) and IHSG*. Diakses melalui library quantmod di R.

* Zheng, Z. (2023). A Review of Stock Price Prediction Based on LSTM and TCN Methods. *Advances in Economics Management and Political Sciences*, 46(1), 48–54. [https://doi.org/10.54254/2754-1169/46/20230316](https://doi.org/10.54254/2754-1169/46/20230316)
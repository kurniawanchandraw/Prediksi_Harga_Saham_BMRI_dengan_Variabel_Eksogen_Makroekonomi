# Peramalan Harga Penutupan Saham BMRI dengan Mempertimbangkan Pengaruh Makroekonomi menggunakan Model Hybrid LSTM-Transformer

## Deskripsi Proyek

Proyek ini bertujuan untuk mengembangkan dan membandingkan beberapa model deep learning dalam memprediksi harga saham PT Bank Mandiri (Persero) Tbk (BMRI). Model yang diuji meliputi:

* LSTM Multi Input
* LSTM-Transformer
* LSTM-Transformer-TCN

Data yang digunakan meliputi harga saham BMRI, Indeks Harga Saham Gabungan (IHSG), serta variabel makroekonomi seperti inflasi, suku bunga acuan BI, dan nilai tukar Rupiah terhadap Dolar AS. Data diambil dari sumber resmi dan diolah untuk membentuk dataset time series dengan fitur lag yang sesuai.

## Fitur Utama

* Preprocessing data dengan interpolasi dan scaling menggunakan RobustScaler
* Penggunaan fitur lag untuk menangkap pola temporal
* Pengembangan model deep learning hybrid yang menggabungkan LSTM, Transformer, dan Temporal Convolutional Network (TCN)
* Pelatihan dan evaluasi model dengan metrik MSE, RMSE, MAE, MAPE, dan R-squared
* Visualisasi hasil prediksi dan forecast multi-step

## Struktur Folder

```
/project-root
│
├── data/               # Data mentah dan hasil preprocessing
├── src/                # Kode Python modular (data_loader, preprocessing, modeling, dll)
├── notebooks/          # Jupyter notebook analisis dan eksperimen
├── scripts/            # Kode R untuk mengambil data saham
├── report.md           # Laporan proyek lengkap
├── README.md           # File dokumentasi ini
└── requirements.txt    # Daftar dependencies Python
```

## Cara Menjalankan

1. **Instal dependencies**
   Pastikan Python 3.8+ sudah terinstall, lalu jalankan:

   ```bash
   pip install -r requirements.txt
   ```

2. **Jalankan notebook**
   Buka `notebooks/notebook.ipynb` menggunakan Jupyter Notebook atau JupyterLab:

   ```bash
   jupyter notebook notebooks/notebook.ipynb
   ```

3. Ikuti alur notebook untuk memuat data, preprocessing, membangun model, melatih, dan mengevaluasi.

## Referensi

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

## Kontak

Jika ada pertanyaan atau ingin berdiskusi, silakan hubungi:
**Kurniawan Chandra Wijaya**
Email: [kurniawanchandrawi@gmail.com](mailto:kurniawanchandrawi@gmail.com)

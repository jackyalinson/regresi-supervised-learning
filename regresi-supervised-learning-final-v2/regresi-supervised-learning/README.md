# Supervised Learning — Analisis Regresi

Repository ini berisi tugas **Supervised Learning** pada data simulasi berukuran 1.500.000 observasi dengan 30 prediktor (X1–X30) dan satu respons (Y).

Analisis mencakup:
- uji normalitas residual,
- multikolinearitas,
- heteroskedastisitas,
- autokorelasi residual,
- OLS,
- WLS,
- GLS dengan struktur AR(1),
- Ridge Regression,
- perbandingan model menggunakan RMSE dan AIC.

## Temuan utama

### Ringkasan uji asumsi

| Aspek | Hasil | Interpretasi |
|---|---:|---|
| R² OLS | 0,1866 | Sekitar 18,66% variasi Y dijelaskan model OLS |
| Skewness residual | 1,1429 | Residual menceng ke kanan |
| Kurtosis residual | 6,6931 | Lebih tinggi dari kurtosis normal (= 3) |
| Jarque-Bera | p < 0,001 | Tolak normalitas |
| D’Agostino-Pearson | p < 0,001 | Tolak normalitas |
| Shapiro-Wilk (n=5.000) | p < 0,001 | Tolak normalitas |
| VIF > 10 | 27/30 | Multikolinearitas kuat |
| VIF > 5 | 30/30 | Semua prediktor perlu perhatian |
| Condition number | 57,8 | Indikasi multikolinearitas kuat |
| Breusch-Pagan | p = 0,4048 | Tidak menolak homoskedastisitas |
| White ringkas | p ≈ 1,27×10⁻¹⁶³ | Ada indikasi heteroskedastisitas |
| White penuh (20.000 sampel) | p ≈ 1,96×10⁻⁹³ | Ada indikasi heteroskedastisitas |
| Durbin-Watson | 0,6000 | Autokorelasi positif kuat |
| rho lag-1 | 0,7000 | Korelasi residual lag-1 kuat |
| Breusch-Godfrey (5 lag) | p < 0,001 | Ada autokorelasi |
| Ljung-Box (10 lag) | p < 0,001 | Ada autokorelasi |

### Perbandingan empat model

| Model | RMSE Train | RMSE Test | AIC Native | AIC Common | df Efektif |
|---|---:|---:|---:|---:|---:|
| OLS | 5,7195 | 5,7012 | 7.590.810,84 | 7.590.810,84 | 31,00 |
| WLS | 5,7195 | 5,7012 | 7.590.814,27 | 7.590.810,84 | 31,00 |
| GLS (AR1) | 5,7195 | 5,7012 | 6.782.799,35 | 7.590.834,32 | 32,00 |
| Ridge | 5,7195 | 5,7012 | 7.590.810,81 | 7.590.810,81 | 30,98 |

RMSE data uji keempat model sangat berdekatan. Perbedaan yang lebih relevan adalah tujuan masing-masing metode dalam menangani karakteristik data.

### Alasan penggunaan setiap model

| Model | Peran dalam analisis |
|---|---|
| **OLS** | Baseline regresi dan dasar untuk diagnosis residual |
| **WLS** | Alternatif ketika terdapat indikasi varians residual tidak konstan |
| **GLS (AR1)** | Menangani struktur autokorelasi residual |
| **Ridge** | Mengurangi dampak multikolinearitas melalui regularisasi |

### Ridge

Ridge dipilih melalui **5-fold cross-validation berbasis blok berurutan**.

- Lambda optimal: **10**
- Effective degrees of freedom: **30,98 dari 31**

Nilai lambda yang relatif kecil membuat hasil Ridge sangat dekat dengan OLS pada data ini.

## Interpretasi hasil

Hasil diagnosis menunjukkan bahwa residual OLS tidak normal, terdapat multikolinearitas kuat, dan terdapat bukti heteroskedastisitas dari White test meskipun Breusch-Pagan tidak signifikan. Temuan paling kuat adalah autokorelasi residual, dengan Durbin-Watson 0,6000 dan rho lag-1 sekitar 0,7000.

Keempat model menghasilkan RMSE test yang hampir sama, sekitar 5,7012. Karena itu, hasil ini tidak menunjukkan perbedaan besar dalam akurasi prediksi pada data uji. WLS, GLS, dan Ridge lebih tepat dipahami sebagai pendekatan untuk menangani karakteristik tertentu dari data, bukan sebagai metode yang otomatis menghasilkan peningkatan RMSE yang besar.

AIC **native** WLS/GLS perlu dibaca hati-hati karena likelihood model terboboti/tertransformasi tidak berada pada basis yang sama dengan OLS. AIC **common** dihitung pada skala residual asli sebagai pembanding tambahan.

## Struktur repository

```text
regresi-supervised-learning/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   └── analisis_regresi.ipynb
├── src/
│   └── regresi.py
├── output/
│   ├── figures/
│   └── tables/
└── report/
    └── laporan.md
```

## Menjalankan analisis

Setelah dataset `.txt` tersedia:

```bash
pip install -r requirements.txt
python src/regresi.py --data raw_data_simulasi.txt --outdir output
```

Notebook asli dari Colab juga disimpan di `notebooks/analisis_regresi.ipynb`.

## Catatan

Output pada repository ini merupakan hasil analisis yang telah dijalankan sebelumnya di Google Colab. Dataset mentah tidak disertakan.

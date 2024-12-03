***Project Introduction***

File ini berisikan rincian proses dalam membangun Machine learning model untuk memprediksi harga rumah berdasarkan dari dataset California Housing Prices
# **California Housing Prices**
## **Business Problem Understanding**
### **Context**

California adalah salah satu negara bagian di Amerika Serikat dengan populasi yang padat dan pasar properti yang sangat kompetitif. Harga rumah dipengaruhi oleh berbagai faktor seperti pendapatan median, lokasi geografis, dan fasilitas lingkungan. Dengan banyaknya permintaan akan properti, memahami faktor-faktor yang memengaruhi harga rumah menjadi penting bagi pengembang properti, investor, dan pemerintah. Maka dari itu, untuk meninjau lebih lanjut mengenai faktor yang mempengaruhi nilai rumah, maka kita akan menggunakan dataset California Housing Price.

Dataset California Housing Price adalah dataset yang berisi informasi mengenai median harga rumah per-blok/area di California dari Sensus tahun 1990 sebagaimana yang dilansir pada paper berjudul [Sparse spatial autoregressions](https://www.sciencedirect.com/science/article/abs/pii/S016771529600140X). Dataset ini mencakup berbagai variabel yang mempengaruhi harga rumah, seperti longitude, latitude, median house age, total rooms, total bedrooms, population, households, median income, median house value, dan ocean proximity.
### **Problem Statement**

Di pasar properti yang kompetitif, menentukan harga jual rumah yang tepat adalah tantangan yang signifikan. **Penjual rumah membutuhkan alat yang dapat memberikan estimasi harga berdasarkan berbagai fitur properti, sehingga mereka dapat menetapkan harga yang kompetitif dan menarik bagi pembeli**. Kurangnya panduan yang tepat dapat menyebabkan harga yang terlalu tinggi atau terlalu rendah, yang berakibat pada penjualan yang lambat atau kehilangan potensi keuntungan.
### **Goals**

Tujuan utama dari analisis ini adalah: **untuk membangun alat prediksi harga rumah yang dapat membantu penjual dalam menentukan harga jual yang kompetitif**. Adapun tujuan yang lebih terperinci yaitu:
1.	Menghasilkan model prediksi yang dapat mengestimasi harga rumah berdasarkan fitur-fitur seperti pendapatan median, usia rumah, dan lokasi geografis.
2.	Memberikan wawasan tentang faktor mana yang paling memengaruhi harga rumah.
3.	Membantu para pemangku kepentingan (pembeli rumah, pemilik rumah, developer perusahaan real estate, dan investor properti) dalam membuat keputusan yang lebih baik berdasarkan estimasi nilai pasar rumah.

### **Analytic Approach**

Pendekatan analitik melibatkan menganalisis data untuk menemukan pola-pola dalam fitur yang ada, serta membangun **model regresi** yang dapat memprediksi harga rumah berdasarkan variabel-variabel yang tersedia. Sebelum analisis lebih lanjut, data perlu dibersihkan dan diproses untuk memastikan kualitas dan konsistensi. Setelah itu, model prediktif dapat digunakan untuk membantu perusahaan real estate dalam menyediakan alat prediksi harga jual properti yang baru masuk dalam daftar.
Berikut ini merupakan langkah-langkah analitik yang akan diambil:
1.	Eksplorasi data untuk memahami pola dan distribusi nilai dari setiap fitur.
2.	Pra-pemrosesan data, termasuk menangani nilai yang hilang, standarisasi fitur, dan encoding data kategorikal.
3.	Pemodelan prediktif dengan mencoba pemodelan dengan algoritma seperti:
	- Regresi Linear (sebagai baseline).
	- Model non-linear seperti Decision Tree, KNNeighborhood, Random Forest dan Gradient Boosting.
4.  Evaluasi performa model menggunakan dataset yang telah dibagi menjadi data pelatihan dan pengujian.
5.  Menentukan model dengan performa terbaik yang selanjutnya akan dilakukan tuning hyperparameter untuk meningkatkan performa.
### **Metric Evaluation**

Keberhasilan model akan dievaluasi menggunakan metrik berikut:
- Root Mean Squared Error (RMSE): Memberikan penalti lebih besar untuk kesalahan prediksi yang lebih besar.
- Mean Absolute Error (MAE): Rata-rata kesalahan absolut antara nilai prediksi dan nilai aktual.
- Mean Absolute Percentage Error (MAPE): Rata-rata persentase error yang dihasilkan oleh model regresi.
- R² Score: Mengukur seberapa baik model menjelaskan variasi pada data target.

Semakin kecil nilai RMSE, MAE, dan MAPE yang dihasilkan, berarti model semakin akurat dalam memprediksi harga sewa sesuai dengan limitasi fitur yang digunakan.

Selain itu, kita juga bisa menggunakan nilai R-squared jika model yang nanti terpilih sebagai final model adalah model linear. Nilai R-squared digunakan untuk mengetahui seberapa baik model dapat merepresentasikan varians keseluruhan data. Semakin mendekati 1, maka semakin fit pula modelnya terhadap data observasi. Namun, metrik ini tidak valid untuk model non-linear.

## **Data Understanding**
- Dataset berisi informasi mengenai median harga rumah untuk distrik-distrik di California yang berasal dari sensus tahun 1990. Informasi data-datanya:

### **Attributes Information**

| **Attribute**           | **Data Type** | **Description**                                                                 |
|-------------------------|---------------|---------------------------------------------------------------------------------|
| longitude               | Float         | Ukuran sejauh mana suatu rumah berada ke barat; nilai yang lebih tinggi berarti lebih ke barat. |
| latitude                | Float         | Ukuran sejauh mana suatu rumah berada ke utara; nilai yang lebih tinggi berarti lebih ke utara. |
| housing_median_age      | Integer       | Usia median rumah dalam suatu blok; angka yang lebih rendah menunjukkan bangunan yang lebih baru. |
| total_rooms             | Integer       | Total jumlah ruangan dalam suatu blok.                                          |
| total_bedrooms          | Integer       | Total jumlah kamar tidur dalam suatu blok.                                      |
| population              | Integer       | Total jumlah penduduk tinggal dalam suatu blok.                                 |
| households              | Integer       | Total jumlah rumah tangga, kelompok orang yang tinggal dalam suatu unit rumah, dalam suatu blok. |
| median_income           | Float         | Pendapatan median untuk rumah tangga dalam suatu blok rumah (diukur dalam puluhan ribu Dolar AS). |
| median_house_value      | Float         | Nilai rumah median untuk rumah tangga dalam suatu blok (diukur dalam Dolar AS).  |
| ocean_proximity         | Object        | Lokasi rumah terkait dengan samudera/laut.                                      |

<br>
### **Limitasi Model**

Dari hasil pelatihan-pengujian model akhir yang telah dipilih (XGBoost tuned), dapat dilihat bahwa ternyata model memiliki batasan di mana **model kurang baik dalam memprediksi data yang memiliki harga tinggi**. Ini dikarenakan oleh: 
1. Distribusi data pada harga tinggi tidak merata, membuat model kesulitan untuk mempelajari pola pada rentan harga sekitar 400.000 ke atas. Ditambah lagi dengan adanya penghilangan data outliers yang ekstrim yang terletak di harga 500.000 ke atas, di mana penghilangan outlier ini dilakukan demi menjaga agar data representatif dan stabil, dengan pertimbangan mencegah Model bekerja dengan terlalu menyesuaikan ke data outliers yang mengakibatkan model tersebut menjadi overfitted.
2. Complexity of High-Value Properties: Properti dengan harga tinggi biasanya memiliki karakteristik yang lebih kompleks (misalnya fasilitas mewah, luas rumah) yang dalam kasus ini kita tidak mempunyai fitur-fitur tersebut.

    Pada bab rekomendasi nantinya akan dijelaskan untuk mengatasi limitasi ini.
## **Kesimpulan**
- **Model terbaik adalah XGBoost** Ini dikarenakan cara kerjanya yang melibatkan pembuatan pohon keputusan secara bertahap, di mana setiap pohon baru berusaha memperbaiki kesalahan dari pohon sebelumnya. Selain itu, XGBoost menggunakan gradient descent untuk meminimalkan kesalahan secara iteratif dan regularisasi untuk mencegah overfitting. Dengan pendekatan ini, XGBoost mampu menangkap kompleksitas data dan menghasilkan prediksi yang akurat.

- Setelah melatih model XGBoost dengan fitur polinomial (Degree=2), tidak ada terjadinya peningkatan performa, maka polinomial tidak dilanjutkan. XGBoost setelah tuning merupakan model dengan performa paling baik dengan hasil evaluasi metrik sebagai berikut:

    | Metric | RMSE               | MAE                | **MAPE**             | R²               |
    |--------|--------------------|--------------------|------------------|------------------|
    | Value  | 47899.80365376014  | 32128.262897608492 | **0.17867179881535333** | 0.82331326699297 |

    ***...***
- **Metrik Evaluasi**: Jika kita melihat berdasarkan nilai RMSE, didapati nilai RMSE cukup tinggi. Hal ini dikarenakan metrik RMSE memiliki beberapa kelemahan: RMSE tergantung oleh skala data, jadi semakin besar skala, maka nilai RMSE-nya juga besar menurut [sumber](https://www.aporia.com/learn/root-mean-square-error-rmse-the-cornerstone-for-evaluating-regression-models/?form=MG0AV3). RMSE juga dipengaruhi oleh outlier, semakin banyak outlier maka RMSE juga bisa semakin besar. Seperti yang kita ketahui, data kita memiliki outlier yang cukup banyak, tapi jika outliernya dihilangkan maka kita akan kehilangan informasi yang banyak pula. Oleh karena itu, pada kasus ini saya lebih melihat hasil pemodelan menggunakan metrik MAPE yang tidak terlalu sensitif terhadap adanya outlier, di mana hasil dari metrik MAPE sendiri sebesar 17% yang artinya persen kesalahan hasil prediksi data dibanding data aktual hanya sekitar 17%. Selain itu, nilai MAPE 17% termasuk dalam kategori 'Good Forecast' atau model peramalan yang baik menurut [sumber](https://dqlab.id/kriteria-jenis-teknik-analisis-data-dalam-forecasting?form=MG0AV3).

- **Feature Importance**: Berdasarkan pemodelan yang sudah dilakukan, fitur **'ocean_proximity'** dan **'median_income'** menjadi fitur yang paling berpengaruh terhadap **'median_house_value'**. Hal ini cukup wajar artinya kita dapat mengkonfirmasi bahwa lokasi ternyata masih menjadi predictor yang paling kuat dalam menentukan harga suatu rumah. Semakin rumah tersebut berada dalam area atau kawasan yang elit, tentu saja harganya akan tinggi dan sebaliknya. Dalam kasus ini, rumah yang berada di kawasan pinggir dengan pemandangan laut merupakan rumah yang paling mahal dibandingkan dengan rumah yang berada di lokasi lainnya.
Hal ini juga berbanding lurus dengan fitur 'median_income', di mana rata-rata penghasilan seseorang dalam suatu area akan menentukan harga rumah di sekitarnya. Semakin besar rata-rata penghasilan seseorang di area tersebut, maka akan semakin mahal harga rumahnya, begitu pula sebaliknya.

- **Limitasi Model**: pada grafik prediction vs actual size, model memiliki kecenderungan untuk salah memprediksi harga di 400.000 ke atas. Ini bisa menjadi indikasi bahwa **model mungkin terbatas dalam menangkap variasi dalam data yang lebih tinggi**.




## **Rekomendasi**
- **Penambahan Fitur Relevan**:
Tambahkan fitur seperti **luas rumah, fasilitas rumah, dan perusahaan pengembang** untuk meningkatkan akurasi prediksi harga rumah.

- **Pembaruan Data**:
Perbarui data yang digunakan, karena data saat ini dari tahun 1990 sudah tidak relevan akibat **inflasi dan perubahan kondisi pasar**.

- **Hyperparameter Tuning**:
**Gunakan metode grid search** untuk hyperparameter tuning yang lebih baik dibandingkan randomized search yang hanya memilih kombinasi secara acak.

- **Penggunaan Model**:
Model dapat digunakan untuk prediksi harga rumah yang memiliki fitur serupa dengan dataset California house. Model menunjukkan performa stabil dan tidak overfitting atau underfitting.

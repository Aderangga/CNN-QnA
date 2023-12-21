# Question Answering

## Deskripsi Aplikasi [CNN-QnA]

Aplikasi CNN-QNA adalah sebuah aplikasi question answering yang memungkinkan pengguna untuk memperoleh jawaban dari pertanyaan yang diajukan terkait dengan teks yang telah dimasukkan. Dengan kata lain, aplikasi ini dirancang untuk memberikan respon yang relevan berdasarkan teks input. Ketika pengguna mengajukan pertanyaan yang tidak terdapat dalam konteks teks yang telah dimasukkan, aplikasi ini akan memberikan jawaban secara acak sesuai dengan konteks yang ada. Ini membuat aplikasi CNN-QNA menjadi alat yang berguna untuk mendapatkan informasi atau klarifikasi terkait dengan suatu teks melalui proses tanya jawab.

## Arsitektur Aplikasi

Aplikasi CNN-QNA menggunakan satu library Streamlit untuk mengintegrasikan kedua bagian utama, yaitu front end dan back end, menggunakan bahasa pemrograman Python. Front end dari aplikasi ini juga dibangun menggunakan bahasa pemrograman Python. Alur kerja aplikasi CNN-QNA dimulai dengan pengguna memilih dan memasukkan beberapa paragraf dokumen sebagai konteks. Selanjutnya, pengguna memberikan satu pertanyaan, dan CNN-QNA akan menampilkan jawaban yang sesuai berdasarkan konteks yang telah dimasukkan sebelumnya. Sebelum teks dimasukkan, langkah preprocessing dilakukan terlebih dahulu menggunakan tokenizer. Hal ini diperlukan untuk mentokenisasi input yang diberikan, mempersiapkannya agar dapat diproses oleh model.

Setelah tahap preprocessing selesai, dilakukan proses training model. Proses ini melibatkan pengunduhan pretrained model dan penerapan fine-tuning untuk meningkatkan performa model sesuai dengan tujuan aplikasi. Back end dari aplikasi ini menggunakan library Streamlit dan transformers dengan membawa model dari pipeline sebagai parameter. Nama model yang akan digunakan sudah ditentukan oleh Hugging Face pada akun pengguna. Seluruh elemen ini kemudian dimasukkan ke dalam proses load model untuk persiapan penggunaan dalam aplikasi. Dengan demikian, aplikasi CNN-QNA menyajikan antarmuka yang responsif dan efisien dengan dukungan penuh dari library Streamlit dan transformers.


## Proses


### Dataset

Dataset yang digunakan adalah dataset NewsQA dari Microsoft dimana isinya berupa kumpulan pasangan pertanyaan dan jawaban yang berasal dari artikel Website CNN. Tujuan dari sistem yang dilatih pada kumpulan data ini adalah untuk dapat menjawab pertanyaan tentang konten di artikel CNN.

Dataset NewsQA ini terdiri dari 119.634 pertanyaan dan jawaban yang terdiri dari beberapa jawaban benar dan beberapa pertanyaan yang memang tidak dapat dijawab. Pada NewsQA, jawaban pertanyaan yang benar dapat berupa urutan token apapun dalam teks yang diberikan.

Sumber: https://www.microsoft.com/en-us/download/details.aspx?id=57162


### Algoritma BERT

Algoritma BERT (Bidirectional Encoder Representations from Transformers) adalah sebuah model bahasa yang dikembangkan oleh Google. BERT menggunakan pendekatan transformer, yang merupakan arsitektur jaringan saraf yang sangat sukses dalam pemrosesan bahasa alami. Ini difokuskan pada pemahaman konteks kata dalam suatu kalimat dengan memperhitungkan hubungan antara kata-kata sebelum dan sesudahnya.

![gambar](https://github.com/Aderangga/CNN-QnA/assets/83385281/f10e47a6-6733-4876-b2b9-7f1c704ebcdf)
Sumber:https://www.unite.ai/id/nlp-rise-with-transformer-models-a-comprehensive-analysis-of-t5-bert-and-gpt/

Arsitektur transformer terdiri dari beberapa blok yang saling terhubung, termasuk blok encoder dan decoder. Setiap blok memiliki beberapa sub-layer, seperti multi-head self-attention dan feedforward layer. BERT menggunakan blok encoder untuk tugas pemahaman bahasa. BERT menggunakan blok encoder transformer dan memodifikasi pendekatannya agar bisa melakukan pemrosesan dua arah (bidireksional). 

Dalam konteks BERT, itu berarti model dapat melihat kata-kata sebelum dan sesudahnya dalam suatu kalimat saat memahami kata tertentu. Hal ini menciptakan representasi kata yang lebih baik. BERT di-pre-training pada besar korpus teks tidak terlabel. Dalam fase ini, model belajar untuk memahami konteks dan hubungan antar kata. Model dilatih untuk memprediksi kata yang hilang dalam kalimat atau memprediksi hubungan antara dua kalimat yang saling berhubungan. Setelah pre-training, BERT dapat di-fine-tuning untuk tugas spesifik seperti pemrosesan bahasa alami, klasifikasi teks, dan lainnya. Ini melibatkan melatih model pada dataset yang dilabeli untuk tugas tertentu. Algoritma BERT sangat efektif dalam memahami konteks kata dalam kalimat dan telah menjadi dasar untuk berbagai tugas NLP.


### Preprocessing

Proses dimulai dengan menghubungkan Google Colab ke Google Drive untuk mendapatkan akses ke file yang tersimpan di Drive. Selanjutnya, dilakukan instalasi library transformers yang esensial untuk bekerja dengan model BERT dan fungsi-fungsi pendukungnya.

Berbagai modul dan library seperti `PyTorch`, `Transformers`, `pandas`, `numpy`, dan `matplotlib` diimpor untuk mendukung proses pelatihan dan analisis hasil. Penetapan perangkat (device) digunakan untuk menentukan apakah model akan dilatih menggunakan `GPU atau CPU`. Selain itu, seed acak ditetapkan untuk memastikan reproduktibilitas hasil pelatihan.

Data NewsQA dibaca dari file `JSON`, dan kemudian dilakukan pembersihan dan pemrosesan awal. Dataset kemudian dibagi menjadi dataset pelatihan dan validasi, dengan jumlah data yang dibaca dibatasi untuk keperluan contoh (20000 data pelatihan dan 1000 data validasi).

Proses tokenisasi dilakukan menggunakan tokenizer dari pre-trained BERT untuk menghasilkan indeks token untuk jawaban. Fungsi `calculate_tokenized_ans_indices` dibuat untuk menghitung indeks token jawaban pada dataset, termasuk konversi teks jawaban ke dalam bentuk token menggunakan `tokenizer` BERT.

Selanjutnya, kelas dataset khusus (NewsQA_Dataset) dibuat untuk menyimpan dan mengelola data dalam format yang sesuai dengan kebutuhan pelatihan. Fungsi `calculate_tokenized_ans_indices` diterapkan pada dataset pelatihan dan validasi untuk menghitung indeks token jawaban.

`DataLoader` dibuat untuk mengelola data dalam batch selama pelatihan model. Dilakukan pemeriksaan ketersediaan GPU pada perangkat dan penentuan apakah akan menggunakan GPU atau CPU untuk eksekusi model.

Model BERT untuk pertanyaan jawaban diinisialisasi menggunakan pre-trained model dari `Hugging Face`. Dataset yang telah disiapkan sebelumnya digunakan dalam pelatihan dan validasi model. Proses pelatihan selanjutnya, yang tidak disertakan dalam kutipan di atas, akan melibatkan iterasi melalui beberapa `epoch` untuk meningkatkan kinerja model.



### Training

Pertama, dalam bagian `Batch Size and DataLoader Initialization`, ukuran batch (BATCH_SIZE) diatur sebesar 8, yang menentukan jumlah sampel yang akan diproses pada setiap iterasi pelatihan. DataLoader inisialisasi untuk mengelola dataset pelatihan dan validasi, dengan BATCH_SIZE digunakan untuk dataset pelatihan dan 1 untuk dataset validasi, memastikan bahwa setiap sampel dalam dataset validasi diproses satu per satu.

Selanjutnya, `Train Function` mengacu pada fungsi `train` yang bertanggung jawab untuk melatih model Question Answering. Fungsi ini menggunakan optimizer AdamW dengan learning rate yang ditentukan, dan melibatkan iterasi melalui setiap batch dalam DataLoader pelatihan. Pada setiap iterasi, gradien dihitung melalui proses backpropagation, dan model diperbarui. Setelah selesai setiap epoch pelatihan, model diuji pada dataset validasi untuk mengukur performa pada data yang belum pernah dilihat sebelumnya.

Pada bagian `Training Loop`, terdapat loop untuk setiap epoch pelatihan di mana model didefinisikan sebagai mode pelatihan `(model.train())` dan loss dihitung untuk setiap batch dalam DataLoader pelatihan.

Selanjutnya, dalam `Validation Loop`, model diubah ke mode evaluasi `(model.eval())`, dan loss dihitung untuk setiap batch dalam DataLoader validasi. Loss-validation dihitung dan disimpan untuk setiap epoch.

Fungsi `Plotting Losses (plot_loss_vs_epochs)` digunakan untuk memvisualisasikan perubahan loss selama pelatihan dan validasi. Ini membantu dalam memahami bagaimana model berkembang seiring waktu.

Terakhir, dalam bagian `Model Initialization and Training`, seed acak direset untuk memastikan hasil pelatihan dapat direproduksi. Model BERT untuk pertanyaan jawaban diinisialisasi dari pre-trained model menggunakan Transformers dari Hugging Face. Fungsi pelatihan `(train)` dijalankan untuk melatih model dengan dataset yang telah dipersiapkan sebelumnya. Penting untuk dicatat bahwa bagian ini merupakan akhir dari skrip dan hasil pelatihan, termasuk loss pada setiap epoch, dapat divisualisasikan menggunakan fungsi `plot_loss_vs_epochs`.

![gambar](https://github.com/Aderangga/CNN-QnA/assets/83385281/71c9abc0-dc0d-4fd0-9e07-5b623209226c)


### Testing
Pada proses testing digunakan untuk mengetes model yang sebelumnya sudah di training. Pada aplikasi ini untuk testing menggunakan `validation sets`.

![gambar](https://github.com/Aderangga/CNN-QnA/assets/83385281/281c2a77-2e16-43e8-b075-a0a3f20f28cd)

### Evaluasi
 Ketika mengevaluasi model, memerlukan lebih banyak work karena perlu memetakkan prediksi model untuk kembali ke konteks. Model tersebut mempredikasi posisi jawaban awal dan akhir. Output dari model yaitu loss, logits awal, dan logits akhir. Disinis kami tidak membutuhkan loss. Terdapat satu logits untuk setiap fitur dan setiap token. Untuk memprediksi jawaban pada setiap fitur dilakukan dengan mengambil indeks maksimum logits awal sebagai posisi awal dan indeks maksimum logits akhir sebagai posisi akhir Untuk mengklasifikasikan jawaban, menggunakan skor yang diperoleh dengan menambahkan logits awal dan akhir. Indeks terbaik di logits awal dan akhir akan dipilih dan mengumpulkan semua jawaban yang diprediksi. Setelah itu, semua jawaban akan diurutkan berdasarkan skornya Untuk mempredikasi impossible answer ketika skornya lebih besar dari best non-impossible answer sehingga memberikan post-processing function kita bisa membuat load matric dari datasets library


## Cara Menjalankan Aplikasi
Berikut langkah-langkah untuk menjalankan aplikasi:

1.	Unduh folder bertcnn
2.	Install library yang diperlukan
3.	Run source code dengan command streamlit run main.py


![gambar](https://github.com/Aderangga/CNN-QnA/assets/83385281/ec105ab9-abf7-4960-82b1-34a5c4f69431)


Gambar di atas merupakan tampilan aplikasi CNN-QNA. Untuk menjalankannya, pertama kali user harus menginputkan context ataupun article yang digunakan sebagai sumber jawaban dari pertanyaan yang akan diajukan. Setelah itu user menginputkan pertanyaan yang akan diajukan. Lalu klik tombol "Answers", maka aplikasi akan otomatis memberikan jawaban dari pertanyaan sesuai dengan context ataupun article yang diinputkan.

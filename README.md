# COMPUTER_VISION_606_UAS
# Evaluasi VLM untuk OCR Plat Nomor Kendaraan Indonesia

Proyek ini merupakan pemenuhan tugas Asesmen Akhir Semester (UAS) mata kuliah **Computer Vision (RE604)**. Repositori ini berisi implementasi dan evaluasi sistem *Optical Character Recognition* (OCR) menggunakan *Vision Language Model* (VLM) yang dijalankan secara lokal untuk mengenali teks pada plat nomor kendaraan di Indonesia.

- **Mata Kuliah:** Computer Vision (RE604)
- **Program Studi:** Teknik Robotika
- **Dosen Pengampu:** Eko Rudiawan Jamzuri
- **Oleh:** Said Muhammad Ridho

## Deskripsi Proyek
Sistem ini menggunakan model kecerdasan buatan **LLaVA-Phi-3-Mini** yang dijalankan melalui *local server* **LM Studio**. Gambar pelat nomor dikirim melalui API menggunakan Python untuk diproses. Model diberikan instruksi (*prompt*) spesifik untuk hanya mengekstrak karakter alfanumerik. Sistem juga mengimplementasikan proses *data cleaning* (Regex) untuk menghilangkan noise seperti pembacaan tanggal pajak dan halusinasi AI.

Evaluasi dilakukan menggunakan metrik **Character Error Rate (CER)**.

## Persyaratan Sistem (Prerequisites)
Pastikan sistem Anda memiliki perangkat lunak berikut sebelum menjalankan program:
1. **Python 3.11** atau lebih baru.
2. **LM Studio** (untuk menjalankan model LLM/VLM secara lokal).
3. Library Python:
   ```bash
   pip install openai pandas jiwer pillow

  Persiapan LM StudioBuka aplikasi LM Studio.Unduh dan muat (Load) model multimodal berukuran kecil, direkomendasikan: LLaVA-Phi-3-Mini.Buka tab Local Server.Aktifkan Start Server pada port 1234 (URL bawaan: http://localhost:1234/v1).Cara Eksekusi ProgramLangkah 1: Proses Inferensi VLMFile ocr_vlm.py bertugas membaca gambar dari dataset, mengirimkannya ke LM Studio, dan meminta prediksi teks plat nomor.Jalankan perintah berikut di terminal:Bashpython ocr_vlm.py
Tunggu hingga proses selesai. Program ini akan menghasilkan file mentah bernama ocr_results.csv.Langkah 2: Post-Processing & Evaluasi (Cleaning Data)File clean_data.py bertugas membersihkan hasil prediksi mentah (memotong sisa tanggal pajak dan halusinasi VLM menggunakan Regex) lalu menghitung skor CER akhir.Jalankan perintah berikut di terminal:Bashpython clean_data.py
Program akan menghitung rata-rata error dan menghasilkan file evaluasi akhir bernama ocr_final_report.csv.Metrik EvaluasiTingkat akurasi sistem dievaluasi menggunakan Character Error Rate (CER) dengan formula:$$CER = (S + D + I) / N$$Keterangan:S = Substitusi (Karakter yang salah tebak)D = Delesi (Karakter yang terhapus/hilang)I = Insersi (Karakter tambahan/noise seperti angka pajak)N = Total karakter pada Ground TruthBerdasarkan pengujian pada dataset, model VLM lokal ini mencetak rata-rata CER keseluruhan di angka 0.38.Struktur Repositoriocr_vlm.py : Script utama untuk komunikasi API ke LM Studio dan inferensi gambar.clean_data.py : Script post-processing untuk filter Regex dan kalkulasi CER.ocr_final_report.csv : (Contoh) Hasil akhir evaluasi metrik dalam format tabel.README.md : Dokumentasi eksekusi program.

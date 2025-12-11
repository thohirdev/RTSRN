import numpy as np
import pandas as pd
import os
import sys

def generate_priors_from_csv(csv_path, target_column):
    print(f"--- GENERATOR DICTIONARY PRIORS (THESIS NOVELTY) ---")
    print(f"1. Target File CSV: {csv_path}")
    print(f"2. Target Kolom   : {target_column}")
    
    # 1. Cek Keberadaan File
    if not os.path.exists(csv_path):
        print(f"[ERROR] File '{csv_path}' tidak ditemukan di folder ini!")
        print("Pastikan file CSV sudah dicopy ke folder root RTSRN.")
        return

    # 2. Coba Baca CSV
    try:
        # Kita gunakan error_bad_lines=False (atau on_bad_lines='skip' di pandas baru)
        # untuk jaga-jaga kalau ada baris CSV yang formatnya rusak.
        try:
            df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip')
        except:
            df = pd.read_csv(csv_path, encoding='utf-8', error_bad_lines=False)
            
        print(f"   Berhasil membaca CSV. Total baris data: {len(df)}")
    except Exception as e:
        print(f"[ERROR] Gagal membaca file CSV. Pesan error:\n{e}")
        return

    # 3. Validasi Kolom
    if target_column not in df.columns:
        print(f"\n[ERROR] Kolom '{target_column}' TIDAK DITEMUKAN.")
        print(f"Kolom yang ada: {list(df.columns)}")
        # Fallback logic: Coba cari kolom pertama
        first_col = df.columns[0]
        print(f"Mencoba menggunakan kolom pertama: '{first_col}'...")
        target_column = first_col

    # 4. Proses Teks
    print(f"3. Memproses teks dan menghitung statistik...")
    # Ambil data, buang yang kosong, ubah ke string, lowercase
    text_data = df[target_column].dropna().astype(str).str.lower().tolist()
    
    # Definisi Karakter (Sesuai Model RTSRN: 0-9 + a-z)
    chars = "0123456789abcdefghijklmnopqrstuvwxyz"
    char2idx = {c: i for i, c in enumerate(chars)}
    num_classes = 37 # 36 char + 1 blank
    
    # Inisialisasi Matriks Bigram (37x37)
    bigram_matrix = np.zeros((num_classes, num_classes), dtype=np.float32)
    
    count_pairs = 0
    
    # Loop Corpus
    for line in text_data:
        # Bersihkan karakter aneh (selain 0-9, a-z)
        # Kita gabungkan huruf dalam satu baris kalimat
        clean_chars = [c for c in line if c in char2idx]
        
        if len(clean_chars) < 2:
            continue
            
        # Hitung transisi (Bigram)
        for i in range(len(clean_chars) - 1):
            c1 = clean_chars[i]
            c2 = clean_chars[i+1]
            
            idx1 = char2idx[c1]
            idx2 = char2idx[c2]
            
            bigram_matrix[idx1, idx2] += 1
            count_pairs += 1

    print(f"   Total pasangan karakter dianalisis: {count_pairs}")

    # 5. Normalisasi (Probabilitas)
    row_sums = bigram_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1 
    prob_matrix = bigram_matrix / row_sums
    
    # 6. KUANTISASI (NOVELTY) -> Ubah ke Uint8
    print("4. Melakukan Kuantisasi (Float32 -> Uint8)...")
    quantized_matrix = (prob_matrix * 255).astype(np.uint8)
    
    # 7. Simpan
    output_filename = "quantized_priors.npy"
    np.save(output_filename, quantized_matrix)
    
    print("\n" + "="*40)
    print(f"SUKSES BESAR! Novelty Tesis Siap.")
    print(f"File Output: {output_filename}")
    print(f"Ukuran     : {quantized_matrix.shape}")
    print("="*40)

if __name__ == "__main__":
    # --- KONFIGURASI SESUAI GAMBAR ANDA ---
    CSV_FILENAME = "bilingual_dataset_cleaned_advanced.csv"
    TARGET_COLUMN = "text" 
    # --------------------------------------
    
    generate_priors_from_csv(CSV_FILENAME, TARGET_COLUMN)
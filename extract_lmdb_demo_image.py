import lmdb
import cv2
import numpy as np
import os

def extract_and_separate(lmdb_path, output_root, count=10):
    # 1. Cek Path
    if not os.path.exists(lmdb_path):
        print(f"Error: Folder {lmdb_path} tidak ditemukan!")
        return

    # 2. Siapkan 2 Folder Terpisah
    lr_folder = os.path.join(output_root, 'LR') # Folder untuk gambar buram
    hr_folder = os.path.join(output_root, 'HR') # Folder untuk gambar tajam (Ground Truth)
    
    os.makedirs(lr_folder, exist_ok=True)
    os.makedirs(hr_folder, exist_ok=True)
    
    print(f"Output LR akan disimpan di: {lr_folder}")
    print(f"Output HR akan disimpan di: {hr_folder}")

    # 3. Buka LMDB
    env = lmdb.open(lmdb_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    
    with env.begin(write=False) as txn:
        nSamples = int(txn.get('num-samples'.encode()))
        print(f"Total data: {nSamples}")
        print(f"Mengekstrak {count} pasang gambar...")

        extracted = 0
        for index in range(1, nSamples + 1):
            if extracted >= count:
                break
            
            # --- Key Definitions ---
            key_lr_1 = 'image-%09d' % index
            key_lr_2 = 'image_lr-%09d' % index
            key_hr   = 'image_hr-%09d' % index

            # --- Ambil LR ---
            imgbuf_lr = txn.get(key_lr_1.encode())
            if imgbuf_lr is None:
                imgbuf_lr = txn.get(key_lr_2.encode())

            # --- Ambil HR ---
            imgbuf_hr = txn.get(key_hr.encode())

            # Jika LR tidak ada, skip
            if imgbuf_lr is None:
                continue

            # --- SIMPAN KE FOLDER BERBEDA ---
            
            # 1. Simpan LR
            buf_lr = np.frombuffer(imgbuf_lr, dtype=np.uint8)
            img_lr = cv2.imdecode(buf_lr, cv2.IMREAD_COLOR)
            if img_lr is not None:
                # Simpan di folder LR
                cv2.imwrite(os.path.join(lr_folder, f"img_{index}.png"), img_lr)

            # 2. Simpan HR (Jika ada)
            if imgbuf_hr is not None:
                buf_hr = np.frombuffer(imgbuf_hr, dtype=np.uint8)
                img_hr = cv2.imdecode(buf_hr, cv2.IMREAD_COLOR)
                if img_hr is not None:
                    # Simpan di folder HR
                    cv2.imwrite(os.path.join(hr_folder, f"img_{index}.png"), img_hr)
            
            extracted += 1

    print("\nSelesai! Gambar sudah dipisah.")

if __name__ == '__main__':
    # Ganti path ini sesuai lokasi LMDB
    LMDB_PATH = './textzoom/test/easy' 
    
    # Folder root (nanti di dalamnya akan dibuat folder LR dan HR otomatis)
    OUTPUT_ROOT = './demo_data'
    
    extract_and_separate(LMDB_PATH, OUTPUT_ROOT, count=30)
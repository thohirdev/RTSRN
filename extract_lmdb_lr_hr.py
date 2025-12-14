import lmdb
import cv2
import numpy as np
import os

def extract_images_from_lmdb(lmdb_path, output_folder, count=5):
    # 1. Cek Path
    if not os.path.exists(lmdb_path):
        print(f"Error: Folder {lmdb_path} tidak ditemukan!")
        return

    # 2. Buka Environment LMDB
    env = lmdb.open(lmdb_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Folder output '{output_folder}' dibuat.")

    with env.begin(write=False) as txn:
        # Ambil jumlah data
        nSamples = int(txn.get('num-samples'.encode()))
        print(f"Total data di database: {nSamples}")
        print(f"Mengekstrak {count} sampel pertama (Pasangan LR & HR)...")

        extracted = 0
        
        # Loop index dari 1 sampai nSamples
        for index in range(1, nSamples + 1):
            if extracted >= count:
                break
            
            # --- BAGIAN UPDATE: MENDEFINISIKAN KEY ---
            # Key standar TextZoom:
            # LR biasanya: 'image-%09d' atau 'image_lr-%09d'
            # HR biasanya: 'image_hr-%09d'
            
            key_lr_1 = 'image-%09d' % index      # Coba format 1
            key_lr_2 = 'image_lr-%09d' % index   # Coba format 2 (alternatif)
            key_hr   = 'image_hr-%09d' % index   # Key untuk High Resolution

            # 1. AMBIL IMAGE LOW RES (LR)
            imgbuf_lr = txn.get(key_lr_1.encode())
            if imgbuf_lr is None:
                imgbuf_lr = txn.get(key_lr_2.encode()) # Coba key alternatif

            # 2. AMBIL IMAGE HIGH RES (HR)
            imgbuf_hr = txn.get(key_hr.encode())

            # Jika data LR tidak ada, skip
            if imgbuf_lr is None:
                continue

            # --- PROSES SIMPAN ---
            # Decode LR
            buf_lr = np.frombuffer(imgbuf_lr, dtype=np.uint8)
            img_lr = cv2.imdecode(buf_lr, cv2.IMREAD_COLOR)

            if img_lr is not None:
                # Simpan LR
                save_name_lr = os.path.join(output_folder, f"img_{index}_lr.png")
                cv2.imwrite(save_name_lr, img_lr)
                print(f"Saved LR: {save_name_lr}")
            
            # Decode HR (Jika Ada)
            if imgbuf_hr is not None:
                buf_hr = np.frombuffer(imgbuf_hr, dtype=np.uint8)
                img_hr = cv2.imdecode(buf_hr, cv2.IMREAD_COLOR)
                
                if img_hr is not None:
                    # Simpan HR
                    save_name_hr = os.path.join(output_folder, f"img_{index}_hr.png")
                    cv2.imwrite(save_name_hr, img_hr)
                    print(f"Saved HR: {save_name_hr}")
            else:
                print(f"Warning: HR image untuk index {index} tidak ditemukan (Key: {key_hr})")

            extracted += 1

    print("Selesai!")

if __name__ == '__main__':
    # Ganti path ini sesuai lokasi folder LMDB Anda
    LMDB_PATH = './textzoom/test/easy' 
    
    # Folder tujuan simpan gambar
    OUTPUT_DIR = './test_images_extracted'
    
    extract_images_from_lmdb(LMDB_PATH, OUTPUT_DIR, count=30)
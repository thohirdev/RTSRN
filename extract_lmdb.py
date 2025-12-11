import lmdb
import cv2
import numpy as np
import os

def extract_images_from_lmdb(lmdb_path, output_folder, count=5):
    # 1. Buka Environment LMDB
    if not os.path.exists(lmdb_path):
        print(f"Error: Folder {lmdb_path} tidak ditemukan!")
        return

    env = lmdb.open(lmdb_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Folder output '{output_folder}' dibuat.")

    with env.begin(write=False) as txn:
        # Ambil jumlah data
        nSamples = int(txn.get('num-samples'.encode()))
        nSamples = int(nSamples)
        
        print(f"Total gambar di database: {nSamples}")
        print(f"Mengekstrak {count} gambar pertama...")

        extracted = 0
        # Loop index dari 1 sampai nSamples
        for index in range(1, nSamples + 1):
            if extracted >= count:
                break
                
            # Key gambar di TextZoom biasanya format: "image-000000001"
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())
            
            if imgbuf is None:
                continue

            # Decode bytes ke gambar
            buf = np.frombuffer(imgbuf, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            
            if img is not None:
                # Simpan sebagai PNG
                save_name = os.path.join(output_folder, f"test_img_{index}.png")
                cv2.imwrite(save_name, img)
                print(f"Saved: {save_name}")
                extracted += 1

    print("Selesai! Gambar siap dipakai untuk demo.")

if __name__ == '__main__':
    # Ganti path ini sesuai lokasi folder LMDB easy Anda
    LMDB_PATH = './textzoom/test/easy' 
    
    # Folder tujuan simpan gambar
    OUTPUT_DIR = './test_images'
    
    extract_images_from_lmdb(LMDB_PATH, OUTPUT_DIR, count=10)
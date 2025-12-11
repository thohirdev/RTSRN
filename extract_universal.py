import lmdb
import cv2
import numpy as np
import os

def extract_universal(lmdb_path, output_folder, max_images=10):
    print(f"--- MEMBUKA DATABASE: {lmdb_path} ---")
    
    if not os.path.exists(lmdb_path):
        print(f"[ERROR] Folder {lmdb_path} tidak ditemukan!")
        return

    # Buka Environment
    try:
        env = lmdb.open(lmdb_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    except Exception as e:
        print(f"[ERROR] Gagal membuka LMDB: {e}")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    count_saved = 0
    
    with env.begin(write=False) as txn:
        # Gunakan Cursor untuk melihat SEMUA isi data tanpa menebak kunci
        cursor = txn.cursor()
        
        print("Mulai memindai isi database...")
        
        for key, value in cursor:
            if count_saved >= max_images:
                break
                
            # Decode Key (biasanya bytes) ke String untuk dicek
            try:
                key_str = key.decode('utf-8')
            except:
                key_str = str(key)

            # Lewati metadata (kunci yang bukan gambar)
            if 'num-samples' in key_str:
                continue

            # Coba ubah VALUE menjadi gambar
            try:
                # Convert bytes ke numpy array
                file_bytes = np.frombuffer(value, dtype=np.uint8)
                
                # Coba decode image
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if img is not None:
                    # Berhasil! Ini adalah gambar.
                    # Kita ganti nama filenya biar aman (bersihkan karakter aneh)
                    safe_name = key_str.replace('/', '_').replace('\\', '_')
                    if not safe_name.endswith('.png'):
                        safe_name += ".png"
                        
                    save_path = os.path.join(output_folder, safe_name)
                    cv2.imwrite(save_path, img)
                    
                    print(f"[BERHASIL] Key: {key_str} -> Disimpan ke: {save_path}")
                    count_saved += 1
                else:
                    # Data ada, tapi bukan gambar (mungkin label teks)
                    # print(f"[SKIP] Key: {key_str} (Bukan gambar valid)")
                    pass
                    
            except Exception as e:
                print(f"[ERROR] Key: {key_str} rusak. {e}")

    if count_saved == 0:
        print("\n[KESIMPULAN] Tidak ada gambar yang berhasil diekstrak.")
        print("Kemungkinan isi LMDB Anda hanya teks atau formatnya berbeda.")
    else:
        print(f"\n[SELESAI] Berhasil mengekstrak {count_saved} gambar ke folder '{output_folder}'.")

if __name__ == '__main__':
    # Pastikan path ini benar!
    LMDB_PATH = './textzoom/test/easy' 
    OUTPUT_DIR = './test_images'
    
    extract_universal(LMDB_PATH, OUTPUT_DIR, max_images=10)
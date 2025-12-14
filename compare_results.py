import cv2
import os
import numpy as np

def create_comparison_grid(lr_dir, sr_dir, hr_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Folder output '{output_dir}' dibuat.")

    filenames = [f for f in os.listdir(hr_dir) if f.endswith(('.png', '.jpg'))]
    filenames.sort()

    print(f"Memproses {len(filenames)} gambar...")
    
    count = 0
    for fname in filenames:
        # 1. Definisi Path
        path_lr = os.path.join(lr_dir, fname)
        path_hr = os.path.join(hr_dir, fname)
        sr_filename = f"compare_{fname}" 
        path_sr = os.path.join(sr_dir, sr_filename) 

        # 2. Baca Gambar
        img_lr = cv2.imread(path_lr)
        img_hr = cv2.imread(path_hr)
        img_sr = cv2.imread(path_sr)

        if img_lr is None or img_hr is None or img_sr is None:
            continue

        # 3. Persiapan Ukuran
        h, w, _ = img_hr.shape

        # Resize semua ke ukuran HR
        img_input_view = cv2.resize(img_lr, (w, h), interpolation=cv2.INTER_NEAREST)
        img_bicubic = cv2.resize(img_lr, (w, h), interpolation=cv2.INTER_CUBIC)
        img_sr = cv2.resize(img_sr, (w, h))
        img_hr = img_hr 

        # --- 4. FUNGSI BARU: Tambah Footer Putih & Teks Tengah ---
        def add_caption_below(image, text, color=(0, 0, 0)):
            h_img, w_img = image.shape[:2]
            footer_height = 35  # Tinggi area putih di bawah gambar
            
            # 1. Buat kanvas baru (Putih)
            # Tingginya = Tinggi Gambar + Footer
            new_h = h_img + footer_height
            canvas = np.ones((new_h, w_img, 3), dtype=np.uint8) * 255 
            
            # 2. Tempel gambar asli di bagian atas
            canvas[0:h_img, 0:w_img] = image
            
            # 3. Hitung Posisi Teks agar Center
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            
            # Mendapatkan ukuran teks (lebar, tinggi)
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_w, text_h = text_size
            
            # Rumus Center X: (Lebar Gambar - Lebar Teks) / 2
            x = (w_img - text_w) // 2
            # Rumus Center Y: Tinggi Gambar + (Tinggi Footer + Tinggi Teks)/2
            y = h_img + (footer_height + text_h) // 2 - 4 # -4 biar agak naik dikit
            
            # 4. Tulis Teks
            cv2.putText(canvas, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
            
            return canvas

        # Beri Label (Warna BGR: Blue, Green, Red)
        # Hitam (0,0,0) lebih formal untuk jurnal, tapi saya pakai warna pembeda sesuai request sebelumnya
        panel_1 = add_caption_below(img_input_view, "Input (LR)", color=(0, 0, 0))    # Hitam
        panel_2 = add_caption_below(img_bicubic, "Bicubic", color=(255, 0, 0))       # Biru
        panel_3 = add_caption_below(img_sr, "RTSRN (Ours)", color=(0, 128, 0))       # Hijau Tua
        panel_4 = add_caption_below(img_hr, "Ground Truth", color=(0, 0, 0))         # Hitam

        # 5. Gabung Gambar (Horizontal)
        # Karena tinggi gambar bertambah (ada footer), separator juga harus menyesuaikan
        h_new, _, _ = panel_1.shape
        sep = np.ones((h_new, 5, 3), dtype=np.uint8) * 255 # Separator putih 5px
        
        combined = np.hstack((panel_1, sep, panel_2, sep, panel_3, sep, panel_4))

        # 6. Simpan
        save_path = os.path.join(output_dir, f"grid_{fname}")
        cv2.imwrite(save_path, combined)
        count += 1

    print(f"\nSelesai! {count} gambar disimpan di: {output_dir}")

if __name__ == '__main__':
    # --- KONFIGURASI FOLDER ---
    LR_FOLDER = './demo_data/LR'
    HR_FOLDER = './demo_data/HR'
    SR_FOLDER = './demo_result_images'
    OUTPUT_FOLDER = './comparison_results'
    
    create_comparison_grid(LR_FOLDER, SR_FOLDER, HR_FOLDER, OUTPUT_FOLDER)
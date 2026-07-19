import os
import base64
import re
import pandas as pd
import jiwer
from openai import OpenAI
from io import BytesIO
from PIL import Image

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# --- BAGIAN 1: KONFIGURASI FOLDER ---
base_dataset_path = r"D:\COOLYEAH\SEMESTER 6\Computer Vision 604\UAS\archive (2)\Indonesian License Plate Recognition Dataset"
test_images_folder = os.path.join(base_dataset_path, "images", "test")
test_labels_folder = os.path.join(base_dataset_path, "labels", "test")
classes_file = r"D:\COOLYEAH\SEMESTER 6\Computer Vision 604\UAS\archive (2)\Indonesian License Plate Recognition Dataset\classes.names"

# --- BAGIAN 2: FUNGSI BANTUAN ---
def encode_image(image_path, max_size=(512, 512)):
    """Memperkecil gambar dan mengubahnya ke Base64 agar komputasi lebih ringan."""
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.thumbnail(max_size)
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

def get_ground_truth_from_yolo(label_path, class_names):
    """Membaca file txt YOLO, mengurutkan dari kiri ke kanan, dan menggabungkan karakternya."""
    if not os.path.exists(label_path):
        return ""
    
    with open(label_path, "r") as f:
        lines = f.readlines()
        
    chars = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            x_center = float(parts[1])
            if class_id < len(class_names):
                chars.append((x_center, class_names[class_id]))
                
    # Urutkan berdasarkan koordinat X (kiri ke kanan)
    chars.sort(key=lambda x: x[0])
    return "".join([c[1] for c in chars])

# --- BAGIAN 3: PROSES UTAMA ---
results = []

# Membaca daftar karakter dari classes.names
with open(classes_file, "r") as f:
    classes = [line.strip() for line in f.readlines()]

for filename in os.listdir(test_images_folder):
    if filename.endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(test_images_folder, filename)
        
        base_name = os.path.splitext(filename)[0]
        label_path = os.path.join(test_labels_folder, f"{base_name}.txt")
        
        ground_truth = get_ground_truth_from_yolo(label_path, classes)
        
        if not ground_truth:
            print(f"Melewati {filename}: File label tidak ditemukan atau kosong.")
            continue
            
        base64_image = encode_image(image_path)
        print(f"Menganalisis: {filename} (Target: {ground_truth})")
        
        try:
            # Menggunakan instruksi super ketat dan batas max_tokens
            response = client.chat.completions.create(
                model="local-model",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Read the main license plate number in this image. You must ONLY output the alphanumeric characters of the license plate. DO NOT describe the image. DO NOT write any explanations. DO NOT include the expiry date. Just the alphanumeric characters. Example: B1234XYZ"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                temperature=0.0,
                max_tokens=15  # MEMUTUS LOOPING AI: Maksimal jawaban hanya 15 token/karakter
            )
            
            # Mengambil jawaban mentah
            prediction = response.choices[0].message.content.strip().upper()
            
            # PEMBERSIHAN DATA (POST-PROCESSING)
            # 1. Hapus semua karakter kecuali huruf (A-Z) dan angka (0-9)
            prediction = re.sub(r'[^A-Z0-9]', '', prediction)
            
            # 2. Potong maksimal 9 karakter (Membuang sisa angka tanggal pajak di belakang)
            prediction = prediction[:9]
            
            cer_score = jiwer.cer(ground_truth, prediction)
            print(f"  GT: {ground_truth} | Pred: {prediction} | CER: {cer_score:.2f}\n")
            
            results.append({
                "image": filename,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "CER_score": cer_score
            })
            
        except Exception as e:
            print(f"  Gagal memproses {filename}: {e}\n")

# --- BAGIAN 4: SIMPAN HASIL KE CSV ---
if results:
    df = pd.DataFrame(results)
    df.to_csv("ocr_results.csv", index=False)
    print("Selesai! Hasil disimpan di ocr_results.csv")
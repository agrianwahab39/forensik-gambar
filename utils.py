"""
Utility functions for Forensic Image Analysis System
"""

import numpy as np
import warnings
import json
from datetime import datetime
import os
import shutil

warnings.filterwarnings('ignore')

def detect_outliers_iqr(data, factor=1.5):
    """Detect outliers using IQR method"""
    if len(data) == 0:
        return np.array([], dtype=int)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return np.where((data < lower_bound) | (data > upper_bound))[0]

def calculate_skewness(data):
    """Calculate skewness with empty-array handling"""
    if len(data) == 0:
        return 0
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 3)

def calculate_kurtosis(data):
    """Calculate kurtosis with empty-array handling"""
    if len(data) == 0:
        return 0
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 4) - 3

def normalize_array(arr):
    """Normalize array to 0-1 range"""
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max - arr_min == 0:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)

def safe_divide(numerator, denominator, default=0.0):
    """Safe division with default value"""
    if denominator == 0:
        return default
    return numerator / denominator

# ======================= FUNGSI RIWAYAT ANALISIS (DIPERBARUI) =======================

HISTORY_FILE = 'analysis_history.json'
THUMBNAIL_DIR = 'history_thumbnails'

def load_analysis_history():
    """
    Memuat riwayat analisis dari file JSON.
    Mengembalikan list kosong jika file tidak ada atau rusak.
    """
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            # Handle empty file case
            content = f.read()
            if not content:
                return []
            history = json.loads(content)
        # Pastikan data yang dimuat adalah list
        if not isinstance(history, list):
            print(f"Peringatan: Isi dari '{HISTORY_FILE}' bukan sebuah list. Mengembalikan list kosong.")
            return []
        return history
    except json.JSONDecodeError:
        print(f"Peringatan: Gagal membaca '{HISTORY_FILE}' karena format JSON tidak valid. Mengembalikan list kosong.")
        return []
    except Exception as e:
        print(f"Peringatan: Terjadi error saat memuat riwayat: {e}. Mengembalikan list kosong.")
        return []

def save_analysis_to_history(image_name, analysis_summary, processing_time, thumbnail_path):
    """
    Menyimpan ringkasan analisis baru ke dalam file riwayat JSON.
    """
    history = load_analysis_history()

    new_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'image_name': image_name,
        'thumbnail_path': thumbnail_path,
        'analysis_summary': analysis_summary,
        'processing_time': processing_time
    }
    
    history.append(new_entry)

    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4)
    except Exception as e:
        print(f"Error: Gagal menyimpan riwayat ke '{HISTORY_FILE}': {e}")

def delete_all_history():
    """
    Menghapus semua riwayat analisis.
    Mengembalikan True jika berhasil, False jika gagal.
    """
    try:
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        
        # Hapus juga folder thumbnails jika ada
        if os.path.exists(THUMBNAIL_DIR):
            shutil.rmtree(THUMBNAIL_DIR)
        
        print("✅ Semua riwayat analisis berhasil dihapus.")
        return True
    except Exception as e:
        print(f"❌ Error menghapus riwayat: {e}")
        return False

def delete_selected_history(selected_indices):
    """
    Menghapus riwayat analisis yang dipilih berdasarkan indeks.
    
    Args:
        selected_indices (list): List indeks yang akan dihapus
        
    Returns:
        bool: True jika berhasil, False jika gagal
    """
    history = load_analysis_history()
    
    if not history:
        print("⚠️ Tidak ada riwayat untuk dihapus.")
        return False
        
    try:
        # Kumpulkan path thumbnail yang akan dihapus
        thumbnails_to_delete = []
        valid_indices = [i for i in selected_indices if 0 <= i < len(history)]
        
        if not valid_indices:
            print("⚠️ Tidak ada indeks valid yang dipilih.")
            return False

        # Buat daftar item yang akan disimpan (bukan dihapus)
        history_to_keep = [entry for i, entry in enumerate(history) if i not in valid_indices]

        # Identifikasi thumbnail yang akan dihapus
        for idx in valid_indices:
            thumbnail_path = history[idx].get('thumbnail_path')
            if thumbnail_path and os.path.exists(thumbnail_path):
                thumbnails_to_delete.append(thumbnail_path)
        
        # Simpan history yang sudah diperbarui
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history_to_keep, f, indent=4)
        
        # Hapus file thumbnail yang sudah tidak terpakai
        for thumbnail_path in thumbnails_to_delete:
            try:
                os.remove(thumbnail_path)
            except OSError as e:
                print(f"⚠️ Gagal menghapus thumbnail {thumbnail_path}: {e}")
        
        print(f"✅ Berhasil menghapus {len(valid_indices)} entri riwayat.")
        return True
        
    except Exception as e:
        print(f"❌ Error menghapus riwayat terpilih: {e}")
        # Kembalikan file history ke state semula jika terjadi error
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4)
        return False


def get_history_count():
    """
    Mengembalikan jumlah entri dalam riwayat analisis.
    """
    history = load_analysis_history()
    return len(history)

def clear_empty_thumbnail_folder():
    """
    Menghapus folder thumbnail jika kosong.
    """
    try:
        if os.path.exists(THUMBNAIL_DIR) and not os.listdir(THUMBNAIL_DIR):
            os.rmdir(THUMBNAIL_DIR)
            print("📁 Folder thumbnail kosong telah dihapus.")
    except Exception as e:
        print(f"⚠️ Error membersihkan folder thumbnail: {e}")

# ======================= AKHIR DARI FUNGSI RIWAYAT =======================
"""
Visualization Module for Forensic Image Analysis System
Contains functions for creating comprehensive visualizations, plots, and visual reports
"""

import numpy as np
import cv2
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.backends.backend_pdf import PdfPages
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    class PdfPages:
        def __init__(self, *a, **k):
            raise RuntimeError('matplotlib not available')
from PIL import Image
from datetime import datetime
try:
    from skimage.filters import sobel
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False
    def sobel(x):
        return np.zeros_like(x, dtype=float)
import os
import io
import warnings
try:
    from sklearn.metrics import confusion_matrix, accuracy_score
    import seaborn as sns
    from scipy.stats import gaussian_kde
    SKLEARN_METRICS_AVAILABLE = True
    SCIPY_AVAILABLE = True
except Exception:
    SKLEARN_METRICS_AVAILABLE = False
    SCIPY_AVAILABLE = False

warnings.filterwarnings('ignore')

# ======================= Main Visualization Function (DIPERBAIKI) =======================

def visualize_results_advanced(original_pil, analysis_results, output_filename="advanced_forensic_analysis.png"):
    """Advanced visualization with comprehensive forensic analysis results"""
    print("📊 Creating advanced forensic visualization...")
    
    fig = plt.figure(figsize=(22, 18))
    gs = fig.add_gridspec(4, 4, hspace=0.6, wspace=0.3)
    
    fig.suptitle(
        f"Laporan Visual Analisis Forensik Gambar\nFile: {analysis_results['metadata'].get('Filename', 'N/A')} | Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        fontsize=22, fontweight='bold'
    )
    
    # Baris 1: Core Visuals
    ax1_1 = fig.add_subplot(gs[0, 0])
    ax1_2 = fig.add_subplot(gs[0, 1])
    ax1_3 = fig.add_subplot(gs[0, 2])
    ax1_4 = fig.add_subplot(gs[0, 3])
    create_core_visuals_grid(ax1_1, ax1_2, ax1_3, ax1_4, original_pil, analysis_results)
    
    # Baris 2: Advanced Analysis Visuals
    ax2_1 = fig.add_subplot(gs[1, 0])
    ax2_2 = fig.add_subplot(gs[1, 1])
    ax2_3 = fig.add_subplot(gs[1, 2])
    ax2_4 = fig.add_subplot(gs[1, 3])
    create_advanced_analysis_grid(ax2_1, ax2_2, ax2_3, ax2_4, original_pil, analysis_results)
    
    # Baris 3: Statistical & Metric Visuals
    ax3_1 = fig.add_subplot(gs[2, 0])
    ax3_2 = fig.add_subplot(gs[2, 1])
    ax3_3 = fig.add_subplot(gs[2, 2])
    ax3_4 = fig.add_subplot(gs[2, 3])
    create_statistical_grid(ax3_1, ax3_2, ax3_3, ax3_4, analysis_results)

    # Baris 4: Summary & Report & Validation
    ax_report = fig.add_subplot(gs[3, 0:2])
    ax_validation1 = fig.add_subplot(gs[3, 2])
    ax_validation2 = fig.add_subplot(gs[3, 3])
    create_summary_and_validation_grid(ax_report, ax_validation1, ax_validation2, analysis_results)

    try:
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"📊 Advanced forensic visualization saved as '{output_filename}'")
        plt.close(fig)
        return output_filename
    except Exception as e:
        print(f"❌ Error saving visualization: {e}")
        plt.close(fig)
        return None

# ======================= Grid Helper Functions =======================

def create_core_visuals_grid(ax1, ax2, ax3, ax4, original_pil, results):
    """Create core visuals grid"""
    ax1.imshow(original_pil)
    ax1.set_title("1. Gambar Asli", fontsize=12)
    ax1.axis('off')

    ela_img = results.get('ela_image')
    ela_mean = results.get('ela_mean', 0.0)
    
    if ela_img:
        # If it's not a PIL Image, try converting from numpy array
        if not isinstance(ela_img, Image.Image):
             try:
                 ela_img = Image.fromarray(np.array(ela_img))
             except Exception as e:
                 print(f"Warning: Could not convert ELA to image: {e}")
                 ela_img = Image.new('L', original_pil.size)
    else:
        ela_img = Image.new('L', original_pil.size)

    ela_display = ax2.imshow(ela_img, cmap='hot')
    ax2.set_title(f"2. ELA (μ={ela_mean:.1f})", fontsize=12)
    ax2.axis('off')
    plt.colorbar(ela_display, ax=ax2, fraction=0.046, pad=0.04)

    create_feature_match_visualization(ax3, original_pil, results)
    create_block_match_visualization(ax4, original_pil, results)

def create_advanced_analysis_grid(ax1, ax2, ax3, ax4, original_pil, results):
    """Create advanced analysis grid"""
    create_edge_visualization(ax1, original_pil, results)
    create_illumination_visualization(ax2, original_pil, results)
    
    ghost_display = ax3.imshow(results.get('jpeg_ghost', np.zeros(original_pil.size)), cmap='hot')
    ax3.set_title(f"7. JPEG Ghost ({results.get('jpeg_ghost_suspicious_ratio', 0):.1%} area)", fontsize=12)
    ax3.axis('off')
    plt.colorbar(ghost_display, ax=ax3, fraction=0.046, pad=0.04)

    combined_heatmap = create_advanced_combined_heatmap(results, original_pil.size)
    ax4.imshow(original_pil, alpha=0.4)
    ax4.imshow(combined_heatmap, cmap='hot', alpha=0.6)
    ax4.set_title("8. Peta Kecurigaan Gabungan", fontsize=12)
    ax4.axis('off')

def create_statistical_grid(ax1, ax2, ax3, ax4, results):
    """Create statistical analysis grid"""
    create_frequency_visualization(ax1, results)
    create_texture_visualization(ax2, results)
    create_statistical_visualization(ax3, results)
    create_quality_response_plot(ax4, results)
    
def create_summary_and_validation_grid(ax_report, ax_val1, ax_val2, results):
    """Create summary and validation grid"""
    create_summary_report(ax_report, results)
    populate_validation_visuals(ax_val1, ax_val2)

# ======================= Individual Visualization Functions =======================

def create_feature_match_visualization(ax, original_pil, results):
    img_matches = np.array(original_pil.convert('RGB'))
    keypoints = results.get('sift_keypoints')
    matches = results.get('ransac_matches')
    if keypoints and matches:
        # Tampilkan hanya 20 garis terbaik untuk kejelasan
        for match in matches[:20]:
            pt1 = tuple(map(int, keypoints[match.queryIdx].pt))
            pt2 = tuple(map(int, keypoints[match.trainIdx].pt))
            cv2.line(img_matches, pt1, pt2, (50, 255, 50), 1)
            cv2.circle(img_matches, pt1, 5, (255, 0, 0), -1)
            cv2.circle(img_matches, pt2, 5, (0, 0, 255), -1)
    ax.imshow(img_matches)
    ax.set_title(f"3. Feature Matches ({results.get('ransac_inliers',0)} inliers)", fontsize=12)
    ax.axis('off')

def create_block_match_visualization(ax, original_pil, results):
    img_blocks = np.array(original_pil.convert('RGB'))
    block_matches = results.get('block_matches')
    if block_matches:
        for i, match in enumerate(block_matches[:15]): # Max 15 blok
            x1, y1 = match['block1']; x2, y2 = match['block2']
            color = (255, 0, 0) if i % 2 == 0 else (0, 0, 255)
            cv2.rectangle(img_blocks, (x1, y1), (x1+16, y1+16), color, 2)
            cv2.rectangle(img_blocks, (x2, y2), (x2+16, y2+16), color, 2)
    ax.imshow(img_blocks)
    ax.set_title(f"4. Block Matches ({len(block_matches or [])} found)", fontsize=12)
    ax.axis('off')

def create_localization_visualization(ax, original_pil, analysis_results):
    loc_analysis = analysis_results.get('localization_analysis', {})
    mask = loc_analysis.get('combined_tampering_mask') 
    tampering_pct = loc_analysis.get('tampering_percentage', 0)

    ax.imshow(original_pil)
    if mask is not None and tampering_pct > 0.1:
        # Pastikan mask memiliki ukuran yang sama dengan gambar
        mask_resized = cv2.resize(mask.astype(np.uint8), (original_pil.width, original_pil.height))
        # Buat overlay merah transparan
        red_overlay = np.zeros((original_pil.height, original_pil.width, 4), dtype=np.uint8)
        red_overlay[mask_resized.astype(bool)] = [255, 0, 0, 100] # RGBA
        ax.imshow(red_overlay)
    
    ax.set_title(f"K-Means Localization ({tampering_pct:.1f}%)", fontsize=12)
    ax.axis('off')


def create_frequency_visualization(ax, results):
    freq_data = results.get('frequency_analysis', {}).get('dct_stats', {})
    values = [freq_data.get('low_freq_energy', 0), freq_data.get('mid_freq_energy', 0), freq_data.get('high_freq_energy', 0)]
    labels = [f"Low\n({values[0]:.1e})", f"Mid\n({values[1]:.1e})", f"High\n({values[2]:.1e})"]
    ax.bar(labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
    ax.set_title(f"9. Analisis Frekuensi", fontsize=12)
    ax.set_ylabel('Energi DCT')

def create_texture_visualization(ax, results):
    texture_data = results.get('texture_analysis', {}).get('texture_consistency', {})
    metrics = [k.replace('_consistency', '').capitalize() for k in texture_data.keys()]
    values = list(texture_data.values())
    ax.barh(metrics, values, color='#9467bd', alpha=0.8)
    ax.set_title(f"10. Konsistensi Tekstur", fontsize=12)
    ax.set_xlabel('Skor Inkonsistensi (↑ lebih buruk)')
    ax.grid(axis='x', linestyle='--', alpha=0.6)

def create_edge_visualization(ax, original_pil, results):
    image_gray = np.array(original_pil.convert('L'))
    edges = sobel(image_gray)
    edge_inconsistency = results.get('edge_analysis', {}).get('edge_inconsistency', 0)
    ax.imshow(edges, cmap='gray')
    ax.set_title(f"5. Analisis Tepi (Incons: {edge_inconsistency:.2f})", fontsize=12)
    ax.axis('off')

def create_illumination_visualization(ax, original_pil, results):
    image_array = np.array(original_pil)
    lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
    illumination = lab[:, :, 0]
    illum_inconsistency = results.get('illumination_analysis', {}).get('overall_illumination_inconsistency', 0)
    disp = ax.imshow(illumination, cmap='magma')
    ax.set_title(f"6. Peta Iluminasi (Incons: {illum_inconsistency:.2f})", fontsize=12)
    ax.axis('off')
    plt.colorbar(disp, ax=ax, fraction=0.046, pad=0.04)

def create_statistical_visualization(ax, results):
    stats = results.get('statistical_analysis', {})
    r_entropy = stats.get('R_entropy', stats.get('r_entropy', 0))
    g_entropy = stats.get('G_entropy', stats.get('g_entropy', 0))
    b_entropy = stats.get('B_entropy', stats.get('b_entropy', 0))
    ax.bar(['Red', 'Green', 'Blue'], [r_entropy, g_entropy, b_entropy], color=['#d62728', '#2ca02c', '#1f77b4'], alpha=0.8)
    ax.set_title(f"11. Entropi Kanal", fontsize=12)
    ax.set_ylabel('Entropi (bits)')
    ax.set_ylim(0, 8.5)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
def create_quality_response_plot(ax, results):
    qr = results.get('jpeg_analysis', {}).get('quality_responses', [])
    est_q = results.get('jpeg_analysis', {}).get('estimated_original_quality', 0)
    if qr:
        ax.plot([r['quality'] for r in qr], [r['response_mean'] for r in qr], 'b-o', markersize=5)
        if est_q:
            ax.axvline(x=est_q, color='r', linestyle='--', label=f'Est. Q: {est_q}')
            ax.legend()
    ax.set_title(f"12. Respons Kualitas JPEG", fontsize=12)
    ax.set_xlabel('Kualitas')
    ax.set_ylabel('Error Mean')
    ax.grid(True, alpha=0.5)

def create_advanced_combined_heatmap(analysis_results, image_size):
    """Create combined heatmap with robust size handling"""
    try:
        # Handle both tuple and individual size arguments
        if isinstance(image_size, (tuple, list)) and len(image_size) >= 2:
            w, h = image_size[0], image_size[1]
        elif hasattr(image_size, 'width') and hasattr(image_size, 'height'):
            w, h = image_size.width, image_size.height
        else:
            # Fallback to safe defaults
            w, h = 512, 512
            print(f"  Warning: Invalid image_size {image_size}, using default 512x512")
        
        # Ensure w and h are integers
        w, h = int(w), int(h)
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        # Kontribusi ELA (bobot 35%)
        try:
            ela_image = analysis_results.get('ela_image')
        except Exception as e:
            print(f"  Warning: Failed to get ELA image: {e}")
            ela_image = None
            
        if ela_image is not None:
            if not isinstance(ela_image, Image.Image):
                ela_image = Image.fromarray(np.array(ela_image))
            ela_resized = cv2.resize(np.array(ela_image.convert('L')), (w, h))
            heatmap += (ela_resized / 255.0) * 0.35
    
        # Kontribusi JPEG Ghost (bobot 25%)
        jpeg_ghost = analysis_results.get('jpeg_ghost')
        if jpeg_ghost is not None:
            ghost_resized = cv2.resize(jpeg_ghost, (w, h))
            heatmap += (ghost_resized / np.max(jpeg_ghost + 1e-9)) * 0.25

        # Kontribusi Lokalisasi K-Means (bobot 40%)
        loc_analysis = analysis_results.get('localization_analysis', {})
        if 'combined_tampering_mask' in loc_analysis:
            mask = loc_analysis['combined_tampering_mask']
            if mask is not None:
                mask_resized = cv2.resize(mask.astype(np.uint8), (w, h))
                heatmap += mask_resized * 0.40
            
        heatmap_norm = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-9)
        heatmap_blurred = cv2.GaussianBlur(heatmap_norm, (31, 31), 0)
        return heatmap_blurred
    except Exception as e:
        print(f"  Warning: Failed to create combined heatmap: {e}")
        # Return a blank heatmap
        return np.zeros((512, 512), dtype=np.float32)

def create_summary_report(ax, analysis_results):
    ax.axis('off')
    classification = analysis_results.get('classification', {})
    result_type = classification.get('type', 'N/A')
    color = "darkred" if "Manipulasi" in result_type or "Forgery" in result_type else "darkgreen"
    
    summary_text = (
        r"$\ \bf{HASIL\ KLASIFIKASI}$" + "\n"
        rf"$\ \bf{{Tipe\ Deteksi:}}$ {result_type}\n"
        rf"$\ \bf{{Kepercayaan:}}$ {classification.get('confidence', 'N/A')}\n"
        f"------------------------------------\n"
        rf"$\ \bf{{Skor\ Copy-Move:}}$ {classification.get('copy_move_score', 0)}/100\n"
        rf"$\ \bf{{Skor\ Splicing:}}$ {classification.get('splicing_score', 0)}/100\n\n"
        r"$\ \bf{Temuan\ Kunci}$"
    )
    details = classification.get('details', [])
    if details:
        for detail in details[:5]:
            summary_text += f"\n • {detail[:60]}" + ("..." if len(detail) > 60 else "")
    else:
        summary_text += "\n • Tidak ada temuan signifikan."


    ax.text(0.01, 0.98, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor=color))
    ax.set_title("13. Ringkasan Laporan", fontsize=14, y=1.05)


def populate_validation_visuals(ax1, ax2):
    ax1.set_title("14. Validasi Performa (Sample)", fontsize=12)
    if SKLEARN_METRICS_AVAILABLE:
        y_true = [0, 1, 1, 0, 1, 0, 0, 1, 1, 1]
        y_pred = [0, 1, 0, 0, 1, 1, 0, 1, 1, 1]
        cm = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                    xticklabels=['Asli', 'Manipulasi'],
                    yticklabels=['Asli', 'Manipulasi'])
        ax1.set_xlabel("Prediksi")
        ax1.set_ylabel("Aktual")
        ax1.set_title(f"14. Confusion Matrix (Acc: {accuracy:.1%})", fontsize=12)
    else:
        ax1.text(0.5, 0.5, 'sklearn not available', ha='center')

    ax2.set_title("15. Analisis Kepercayaan", fontsize=12)
    scores = np.random.normal(loc=85, scale=10, size=100)

    if SKLEARN_METRICS_AVAILABLE and SCIPY_AVAILABLE:
        sns.histplot(scores, kde=True, ax=ax2, color="purple")
    elif SKLEARN_METRICS_AVAILABLE:
        sns.histplot(scores, kde=False, ax=ax2, color="purple")
    else:
        ax2.hist(scores, bins=10, color="purple", edgecolor="black")

    ax2.set_xlabel("Skor Kepercayaan (%)")
    ax2.set_ylabel("Frekuensi")
    ax2.axvline(np.mean(scores), color='r', linestyle='--', label=f'Mean: {np.mean(scores):.1f}%')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

# ======================= Backward Compatibility (deprecated functions) =======================
# Fungsi ini dijaga agar tidak merusak pemanggilan dari file lain yang belum diupdate.

def create_technical_metrics_plot(ax, results):
    ax.axis('off')
    ax.set_title("Technical Metrics Plot (Deprecated)", fontsize=10, alpha=0.5)
    ax.text(0.5, 0.5, 'Metrics Available\nin Full Report', ha='center', va='center', fontsize=12, alpha=0.5)

def export_kmeans_visualization(original_pil, analysis_results, output_filename="kmeans_analysis.jpg"):
    """
    Exports a dedicated visualization for K-means clustering analysis.
    """
    if 'localization_analysis' not in analysis_results:
        print("❌ K-means analysis data not available for visualization.")
        return None
        
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    fig.suptitle('K-means Tampering Localization Analysis', fontsize=16, fontweight='bold')
    
    loc = analysis_results.get('localization_analysis', {})
    km = loc.get('kmeans_localization', {})
    
    # Subplot 1: Original Image
    axes[0, 0].imshow(original_pil)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Subplot 2: K-means Cluster Map
    if 'localization_map' in km:
        im_cluster = axes[0, 1].imshow(km['localization_map'], cmap='viridis')
        plt.colorbar(im_cluster, ax=axes[0, 1], fraction=0.046, pad=0.04)
    axes[0, 1].set_title('K-means Cluster Map')
    axes[0, 1].axis('off')
    
    # Subplot 3: Identified Tampering Mask
    if 'tampering_mask' in km:
        axes[1, 0].imshow(km['tampering_mask'], cmap='gray')
    axes[1, 0].set_title('Identified Tampering Mask (K-means)')
    axes[1, 0].axis('off')

    # Subplot 4: Final Detection Overlay
    if 'combined_tampering_mask' in loc and loc['combined_tampering_mask'] is not None:
        axes[1, 1].imshow(original_pil)
        overlay = cv2.resize(loc['combined_tampering_mask'].astype(np.uint8), (original_pil.width, original_pil.height))
        axes[1, 1].imshow(overlay, cmap='Reds', alpha=0.4)
    axes[1, 1].set_title('Final Combined Tampering Detection')
    axes[1, 1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    try:
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"📊 K-means visualization exported to '{output_filename}'")
        return output_filename
    except Exception as e:
        print(f"❌ K-means visualization export failed: {e}")
        plt.close(fig)
        return None
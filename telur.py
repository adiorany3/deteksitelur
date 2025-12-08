import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import hashlib

st.set_page_config(page_title="Penghitung Telur", page_icon="🥚", layout="wide")

# Created by: Galuh Adi Insani
# Contact: https://github.com/galuhadiinsani (optional)

# Sidebar untuk informasi dan tips
st.sidebar.title("ℹ️ Informasi Aplikasi")
st.sidebar.markdown("\n---\n**Created by:** Galuh Adi Insani")
st.sidebar.markdown("""
**Aplikasi Penghitung Telur** menggunakan computer vision untuk mendeteksi telur dari gambar.

**🔧 Bagaimana Cara Kerja?**
- **Preprocessing**: Gambar diubah ke grayscale, kontras ditingkatkan, dan noise dikurangi.
- **Teknik Deteksi**:
  - **Contour Detection**: Thresholding, morphological ops (opsional), deteksi kontur, filter area.
  - **Hough Circle**: Deteksi lingkaran menggunakan Hough Transform untuk bentuk oval telur.
- **Filter Area/Radius**: Menyaring objek berdasarkan ukuran untuk menghilangkan noise.

**📋 Cara Penggunaan:**
1. Upload gambar telur (JPG/PNG).
2. Lihat hasil deteksi otomatis.
3. Gunakan "Sesuaikan Parameter Manual" untuk fine-tune jika hasil tidak akurat.

**💡 Tips untuk Akurasi:**
- Jika terlalu banyak deteksi: Tingkatkan Area Minimum atau Threshold.
- Jika terlalu sedikit: Turunkan Area Minimum atau Threshold.
- Aktifkan "Tampilkan Gambar Intermediate" untuk debugging.
- Pilih teknik sesuai kondisi citra:
  - **Hough Circle**: Telur tegas, bulat, sedikit overlap.
  - **Watershed**: Telur saling menempel/overlap.
  - **Adaptive Threshold**: Pencahayaan tidak merata dan bayangan.
  - **Blob Detection / Ellipse Fit**: Untuk menyaring bentuk bulat/oval dan mengurangi false positives.
- Parameter default disesuaikan per gambar berdasarkan testing.
""")

st.title("🥚 Aplikasi Penghitung Jumlah Telur dari Citra Visual")

st.markdown("---")

uploaded_file = st.file_uploader("📤 Unggah gambar telur", type=["jpg", "jpeg", "png"])

@st.cache_data
def hitung_telur(image, filename, custom_thresh=None, custom_area_min=None, teknik="Contour Detection", teknik_params=None):
    """
    Fungsi untuk menghitung jumlah telur dari gambar menggunakan computer vision.
    
    Parameter:
    - image: Array numpy dari gambar (format RGB).
    - filename: Nama file gambar untuk menentukan parameter default.
    - custom_thresh: Nilai threshold kustom (0-255), jika None gunakan default berdasarkan filename.
    - custom_area_min: Nilai area minimum kustom, jika None gunakan default berdasarkan filename.
    - teknik: Teknik deteksi ("Contour Detection", "Hough Circle", "Adaptive Threshold", "Watershed", "Blob Detection", "Ellipse Fit", atau "Auto").
    - teknik_params: Dictionary berisi parameter teknik spesifik (mis. 'dp','minRadius', 'minCircularity' dll.)

    Mengapa parameter berbeda per gambar:
    - Gambar memiliki variasi pencahayaan, kontras, dan ukuran telur.
    - Threshold rendah (mis. 50) untuk gambar gelap, tinggi (145) untuk terang.
    - Area minimum tinggi untuk menghilangkan noise kecil, rendah untuk telur kecil.
    - Morphological operations membantu menyambung telur terputus atau menghilangkan noise.
    
    Tips untuk akurasi:
    - Jika over-detect (terlalu banyak), tingkatkan area_min atau threshold.
    - Jika under-detect (terlalu sedikit), turunkan area_min atau threshold.
    - Gunakan custom parameter untuk kalibrasi cepat.
    - Test dengan gambar threshold untuk debugging.
    """
    # Konversi ke grayscale (image is RGB from PIL)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Equalize histogram untuk meningkatkan kontras
    gray = cv2.equalizeHist(gray)

    # Blur untuk mengurangi noise
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    # Extract technique-level params including shape filters
    if teknik_params is None:
        teknik_params = {}
    min_circularity = float(teknik_params.get('minCircularity', 0.4))
    min_solidity = float(teknik_params.get('minSolidity', 0.5))

    # Handle Hough Circle with parameters
    if teknik == "Hough Circle":
        dp = teknik_params.get('dp', 1.0) if teknik_params else 1.0
        minDist = teknik_params.get('minDist', 20) if teknik_params else 20
        param1 = teknik_params.get('param1', 50) if teknik_params else 50
        param2 = teknik_params.get('param2', 30) if teknik_params else 30
        minRadius = teknik_params.get('minRadius', 10) if teknik_params else 10
        maxRadius = teknik_params.get('maxRadius', 50) if teknik_params else 50
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                                   param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            jumlah = len(circles[0])
            contours = []
            thresh = blur
        else:
            jumlah = 0
            contours = []
            thresh = blur
        return jumlah, contours, thresh

    # Adaptive Threshold
    if teknik == "Adaptive Threshold":
        method = teknik_params.get('method', 'ADAPTIVE_THRESH_GAUSSIAN_C') if teknik_params else 'ADAPTIVE_THRESH_GAUSSIAN_C'
        if method == 'ADAPTIVE_THRESH_GAUSSIAN_C':
            method_cv = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        else:
            method_cv = cv2.ADAPTIVE_THRESH_MEAN_C
        blockSize = teknik_params.get('blockSize', 11) if teknik_params else 11
        C = teknik_params.get('C', 2) if teknik_params else 2
        # adaptiveThreshold requires an 8-bit gray image
        thresh = cv2.adaptiveThreshold(blur, 255, method_cv, cv2.THRESH_BINARY_INV, blockSize, C)
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Watershed segmentation for overlapping eggs
    elif teknik == "Watershed":
        # Use Otsu to create binary image
        ret, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh_otsu, cv2.MORPH_OPEN, kernel, iterations=2)
        # sure background and foreground
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        # threshold for foreground
        ret2, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        # Marker labeling
        ret3, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown==255] = 0
        # Watershed needs a 3-channel image
        img_for_ws = image.copy()
        cv2.watershed(img_for_ws, markers)
        # Count unique markers > 1
        unique_markers = np.unique(markers)
        min_fg_area = teknik_params.get('min_fg_area', 30) if teknik_params else 30
        obj_labels = [m for m in unique_markers if m > 1]
        contours = []
        thresh = thresh_otsu
        # Convert markers to contours for drawing and filter by min foreground area
        for m in obj_labels:
            mask = np.uint8(markers == m)
            area = cv2.countNonZero(mask)
            if area < min_fg_area:
                continue
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                contours.append(cnts[0])
        jumlah = len(contours)

    # Blob detection
    elif teknik == "Blob Detection":
        # threshold: try Otsu or threshold_val
        if custom_thresh is not None:
            _, thresh = cv2.threshold(blur, custom_thresh, 255, cv2.THRESH_BINARY)
        else:
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Setup SimpleBlobDetector params
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = teknik_params.get('minArea', 50.0) if teknik_params else 50.0
        params.maxArea = teknik_params.get('maxArea', 50000.0) if teknik_params else 50000.0
        params.filterByCircularity = True
        params.minCircularity = teknik_params.get('minCircularity', 0.4) if teknik_params else 0.4
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(thresh)
        jumlah = len(keypoints)
        # Represent keypoints as small contours for consistency
        contours = []
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            r = int(kp.size / 2)
            cnt = np.array([[[x-r, y-r]], [[x+r, y-r]], [[x+r, y+r]], [[x-r, y+r]]])
            contours.append(cnt)

    # Ellipse fit
    elif teknik == "Ellipse Fit":
        # Use threshold similar to default
        if custom_thresh is not None:
            thresh_val = custom_thresh
            thresh_type = cv2.THRESH_BINARY
        elif 'telur.png' in filename:
            thresh_val = 50
            thresh_type = cv2.THRESH_BINARY_INV
        else:
            thresh_val = 145
            thresh_type = cv2.THRESH_BINARY
        _, thresh = cv2.threshold(blur, thresh_val, 255, thresh_type)
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        contours_full, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = []
        min_area = teknik_params.get('min_contour_area', 50) if teknik_params else 50
        jumlah = 0
        for cnt in contours_full:
            if cv2.contourArea(cnt) > min_area and len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                (x, y), (MA, ma), angle = ellipse
                # filter by axis ratio to avoid long non-egg shapes
                ratio = min(MA, ma) / max(MA, ma) if max(MA, ma) > 0 else 0
                if ratio > 0.4:
                    jumlah += 1
                    contours.append(cnt)

    else:
        # Teknik Contour Detection (default)
        # Tentukan threshold berdasarkan filename atau custom
        # Threshold memisahkan telur dari latar belakang berdasarkan intensitas cahaya
        if custom_thresh is not None:
            thresh_val = custom_thresh
            thresh_type = cv2.THRESH_BINARY  # Default untuk custom
        elif 'telur.png' in filename:
            thresh_val = 50  # Rendah karena telur gelap di latar terang
            thresh_type = cv2.THRESH_BINARY_INV  # Inversi untuk telur gelap
        else:
            thresh_val = 145  # Tinggi untuk telur terang di latar gelap
            thresh_type = cv2.THRESH_BINARY

        _, thresh = cv2.threshold(blur, thresh_val, 255, thresh_type)

        # Deteksi kontur berdasarkan kondisi
        if 'telur.png' in filename and custom_thresh is None:
            # Morphological operations untuk telur.png: membersihkan noise dan menyambung telur
            kernel = np.ones((3,3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # RETR_EXTERNAL: hanya kontur luar, menghindari deteksi internal telur
        elif 'Telur4.png' in filename and custom_thresh is None:
            # Morphological operations untuk Telur4.png: mirip telur.png tapi dengan threshold berbeda
            kernel = np.ones((3,3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            # Default tanpa morphological: untuk gambar lain atau custom
            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            # RETR_LIST: semua kontur, termasuk internal, untuk fleksibilitas

    # Tentukan area_min berdasarkan filename atau custom.
    # Area minimum menghilangkan objek kecil (noise) atau besar (bukan telur).
    # Jika tidak ada custom, gunakan rasio terhadap ukuran gambar agar lebih robust terhadap resolusi.
    h, w = image.shape[0], image.shape[1]
    image_area = h * w
    default_relative_area = max(50, int(0.0005 * image_area))
    if custom_area_min is not None:
        area_min = custom_area_min
    elif 'telur.png' in filename:
        area_min = max(50, int(0.0003 * image_area))
    elif 'Telur4.png' in filename:
        area_min = max(120, int(0.0004 * image_area))
    elif 'Telur3.png' in filename:
        area_min = max(1000, int(0.002 * image_area))
    else:
        area_min = default_relative_area

# Filter contours by area, circularity, and solidity for robustness
    def contour_valid(cnt):
        area = cv2.contourArea(cnt)
        if area <= area_min:
            return False
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            return False
        circularity = 4 * np.pi * (area / (peri * peri))
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if circularity < min_circularity or solidity < min_solidity:
            return False
        return True

    telur_contours = [cnt for cnt in contours if contour_valid(cnt)]

    return len(telur_contours), telur_contours, thresh

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)
        
        # Validasi ukuran gambar
        if img_np.shape[0] < 100 or img_np.shape[1] < 100:
            st.error("Gambar terlalu kecil. Gunakan gambar dengan resolusi minimal 100x100 piksel.")
            st.stop()
        
        filename = uploaded_file.name
        
        # Ambil nilai konfigurasi yang dimuat (jika ada)
        config_threshold = st.session_state.get('config_threshold', None)
        config_area_min = st.session_state.get('config_area_min', None)
        config_teknik = st.session_state.get('config_teknik', None)
        config_teknik_params = st.session_state.get('config_teknik_params', {})

        # Opsi penyesuaian parameter
        # Checkbox ini memungkinkan user menyesuaikan threshold dan area minimum secara real-time
        # untuk mendapatkan hasil yang lebih akurat tanpa mengubah kode
        custom_default = True if config_threshold is not None or config_area_min is not None or config_teknik is not None else False
        custom = st.checkbox("Sesuaikan Parameter Manual", value=custom_default, key='custom_checkbox')
        teknik_options = ["Auto", "Contour Detection", "Hough Circle", "Adaptive Threshold", "Watershed", "Blob Detection", "Ellipse Fit"]
        if config_teknik in teknik_options:
            teknik_default = config_teknik
        else:
            teknik_default = "Contour Detection"
        teknik = st.selectbox("Teknik Deteksi", teknik_options, index=teknik_options.index(teknik_default))
        # If Auto selected, choose a technique heuristically
        if teknik == 'Auto':
            def choose_technique(img_np_local):
                gray_local = cv2.cvtColor(img_np_local, cv2.COLOR_RGB2GRAY)
                blur_local = cv2.GaussianBlur(gray_local, (9, 9), 0)
                # Try Hough first
                circles_test = cv2.HoughCircles(blur_local, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=5, maxRadius=80)
                if circles_test is not None and len(circles_test[0]) >= 6:
                    return 'Hough Circle'
                # brightness/contrast check -> adaptive
                mean, std = cv2.meanStdDev(gray_local)
                if std[0][0] < 25:
                    return 'Adaptive Threshold'
                # Detect basic contours
                _, thr_tmp = cv2.threshold(blur_local, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cnts_tmp, _ = cv2.findContours(thr_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(cnts_tmp) > 20:
                    return 'Blob Detection'
                if len(cnts_tmp) > 6 and np.mean([cv2.contourArea(c) for c in cnts_tmp]) > 500:
                    return 'Watershed'
                return 'Contour Detection'
            chosen = choose_technique(img_np)
            st.info(f"Auto dipilih: {chosen}")
            teknik = chosen
        # Parameter umum
        if custom:
            thresh_default = config_threshold if config_threshold is not None else 145
            area_default = config_area_min if config_area_min is not None else 0.5
            thresh_val = st.slider("Threshold", 0, 255, int(thresh_default), key='threshold_slider')  # Slider untuk threshold: 0-255
            area_min = st.slider("Area Minimum", 0.0, 20000.0, float(area_default), key='area_slider')  # Slider untuk area: 0-20000
            min_circularity = st.slider("Min Circularity", 0.0, 1.0, float(config_teknik_params.get('minCircularity', 0.4)), key='min_circularity')
            min_solidity = st.slider("Min Solidity", 0.0, 1.0, float(config_teknik_params.get('minSolidity', 0.5)), key='min_solidity')
        else:
            thresh_val = config_threshold if config_threshold is not None else None  # Gunakan default berdasarkan filename
            area_min = config_area_min if config_area_min is not None else None
            min_circularity = float(config_teknik_params.get('minCircularity', 0.4))
            min_solidity = float(config_teknik_params.get('minSolidity', 0.5))

        # Teknik spesifik parameter
        teknik_params = {}
        if teknik == "Hough Circle":
            teknik_params['dp'] = st.slider("DP (Hough scale)", 1.0, 3.0, float(config_teknik_params.get('dp', 1.0)), key='hough_dp')
            teknik_params['minDist'] = st.slider("Min distance antar lingkaran (px)", 5, 100, int(config_teknik_params.get('minDist', 20)), key='hough_minDist')
            teknik_params['param1'] = st.slider("Param1 (Canny higher threshold)", 10, 200, int(config_teknik_params.get('param1', 50)), key='hough_param1')
            teknik_params['param2'] = st.slider("Param2 (Accumulator threshold)", 10, 100, int(config_teknik_params.get('param2', 30)), key='hough_param2')
            teknik_params['minRadius'] = st.slider("Min radius (px)", 5, 50, int(config_teknik_params.get('minRadius', 10)), key='hough_minR')
            teknik_params['maxRadius'] = st.slider("Max radius (px)", 10, 200, int(config_teknik_params.get('maxRadius', 50)), key='hough_maxR')
        elif teknik == "Adaptive Threshold":
            default_method = config_teknik_params.get('method', 'ADAPTIVE_THRESH_GAUSSIAN_C')
            teknik_params['method'] = st.selectbox("Adaptive Method", ["ADAPTIVE_THRESH_MEAN_C", "ADAPTIVE_THRESH_GAUSSIAN_C"], index=0 if default_method=='ADAPTIVE_THRESH_MEAN_C' else 1, key='adaptive_method')
            teknik_params['blockSize'] = st.slider("Block size (odd)", 3, 51, int(config_teknik_params.get('blockSize', 11)), step=2, key='adaptive_block')
            teknik_params['C'] = st.slider("C (offset)", -20, 20, int(config_teknik_params.get('C', 2)), key='adaptive_C')
        elif teknik == "Watershed":
            teknik_params['min_dist'] = st.slider("Min dist peaks (watershed)", 5, 100, int(config_teknik_params.get('min_dist', 20)), key='ws_min_dist')
            teknik_params['min_fg_area'] = st.slider("Min foreground area (px)", 5, 500, int(config_teknik_params.get('min_fg_area', 30)), key='ws_min_fg')
        elif teknik == "Blob Detection":
            teknik_params['minArea'] = st.slider("Min area (blob)", 5.0, 20000.0, float(config_teknik_params.get('minArea', 50.0)), key='blob_minArea')
            teknik_params['maxArea'] = st.slider("Max area (blob)", 50.0, 200000.0, float(config_teknik_params.get('maxArea', 50000.0)), key='blob_maxArea')
            teknik_params['minCircularity'] = st.slider("Min circularity", 0.0, 1.0, float(config_teknik_params.get('minCircularity', 0.4)), key='blob_circ')
        elif teknik == "Ellipse Fit":
            teknik_params['min_contour_area'] = st.slider("Min contour area (px)", 5, 20000, int(config_teknik_params.get('min_contour_area', 50)), key='ellipse_min_area')
        
        # Progress bar untuk simulasi processing
        progress_bar = st.progress(0)
        progress_bar.progress(50)
        
        # Add common shape filter params into teknik_params for processing
        teknik_params = teknik_params or {}
        teknik_params['minCircularity'] = float(min_circularity)
        teknik_params['minSolidity'] = float(min_solidity)
        jumlah, contours, thresh = hitung_telur(img_np, filename, thresh_val, area_min, teknik, teknik_params)
        
        progress_bar.progress(100)
        progress_bar.empty()
        
        # Layout dengan kolom
        col1, col2 = st.columns(2)
        # Option to compare techniques
        compare = st.checkbox("🔁 Bandingkan Teknik Deteksi")
        compare_methods = st.multiselect("Pilih Teknik untuk Perbandingan", ["Contour Detection", "Hough Circle", "Adaptive Threshold", "Watershed", "Blob Detection", "Ellipse Fit"], default=["Contour Detection", "Watershed"])
        with col1:
            st.subheader("🖼️ Gambar Asli")
            st.image(img, caption="Gambar Telur", width=400)
        
        with col2:
            st.subheader("📊 Hasil Analisis")
            st.write(f"**Jumlah telur terdeteksi:** {jumlah}")
            st.write(f"**Nama file:** {filename}")
            st.write(f"**Resolusi:** {img_np.shape[1]} x {img_np.shape[0]}")
            st.write(f"**Teknik:** {teknik}")
            
            # Hitung total kontur sebelum filter
            if teknik == "Hough Circle":
                st.write("**Teknik Hough Circle tidak menggunakan kontur.**")
            else:
                all_contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                total_before = len(all_contours)
                st.write(f"**Total kontur sebelum filter area:** {total_before}")
            
            # Statistik tambahan jika ada kontur
            if teknik == "Hough Circle":
                # Untuk Hough Circle, hitung radius
                dp = teknik_params.get('dp', 1.0) if teknik_params else 1.0
                minDist = teknik_params.get('minDist', 20) if teknik_params else 20
                param1 = teknik_params.get('param1', 50) if teknik_params else 50
                param2 = teknik_params.get('param2', 30) if teknik_params else 30
                minRadius = teknik_params.get('minRadius', 10) if teknik_params else 10
                maxRadius = teknik_params.get('maxRadius', 50) if teknik_params else 50
                circles = cv2.HoughCircles(cv2.GaussianBlur(cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY), (9, 9), 0), cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                                           param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
                if circles is not None:
                    radii = [i[2] for i in circles[0]]
                    avg_radius = sum(radii) / len(radii)
                    min_radius = min(radii)
                    max_radius = max(radii)
                    st.write(f"**Radius rata-rata telur:** {avg_radius:.2f}")
                    st.write(f"**Radius minimum telur:** {min_radius:.2f}")
                    st.write(f"**Radius maksimum telur:** {max_radius:.2f}")
                else:
                    st.write("**Tidak ada lingkaran yang terdeteksi.**")
            elif contours:
                areas = [cv2.contourArea(cnt) for cnt in contours]
                avg_area = sum(areas) / len(areas)
                min_area = min(areas)
                max_area = max(areas)
                st.write(f"**Area rata-rata telur:** {avg_area:.2f}")
                st.write(f"**Area minimum telur:** {min_area:.2f}")
                st.write(f"**Area maksimum telur:** {max_area:.2f}")
            else:
                st.write("**Tidak ada kontur yang memenuhi kriteria.**")
            
            # Penjelasan hasil
            st.write("**Param teknik:**")
            st.json(teknik_params)
            with st.expander("📖 Penjelasan Hasil Perhitungan"):
                st.markdown("""
                **Apa itu Jumlah Telur Terdeteksi?**
                - Ini adalah jumlah objek yang diidentifikasi sebagai telur berdasarkan ukuran dan bentuknya.
                - Jika angka ini tidak sesuai ekspektasi, sesuaikan Area Minimum, Threshold, atau pilih teknik yang lebih cocok.

                **Teknik Deteksi:**
                - **Contour Detection:** Baik untuk kontras kuat antara telur dan latar.
                - **Hough Circle:** Baik untuk telur yang jelas berbentuk lingkaran/oval; cocok jika tidak saling tumpang tindih.
                - **Adaptive Threshold:** Berguna untuk gambar dengan pencahayaan tidak merata.
                - **Watershed:** Disarankan untuk telur yang saling bertumpuk/berhimpitan; memisahkan objek yang saling bersentuhan.
                - **Blob Detection:** Deteksi berdasarkan area dan circularity; baik untuk bentuk sederhana tanpa banyak detail.
                - **Ellipse Fit:** Mem-fit ellipse pada kontur untuk validasi bentuk telur (mengurangi false positives dari objek memanjang).

                **Total Kontur Sebelum Filter Area:**
                - Jumlah total bentuk yang ditemukan sebelum penyaringan berdasarkan ukuran.
                - Jika terlalu banyak, gambar mungkin memiliki noise atau objek kecil.

                **Statistik Area/Radius:**
                - **Rata-rata**: Ukuran telur tipikal (px atau radius untuk Hough).
                - **Minimum/Maksimum**: Variasi ukuran telur.
                - Area diukur dalam piksel persegi; radius diukur dalam piksel jika menggunakan Hough Circle.

                **Tips & Troubleshooting:**
                - Pilih **Adaptive Threshold** jika ada bayangan atau pencahayaan tidak rata.
                - Pilih **Watershed** jika telur saling menempel untuk memisahkan objek individual.
                - Gunakan **Ellipse Fit** atau **Blob Detection** untuk menyaring bentuk yang tidak bulat.
                - Cobalah menyesuaikan `Threshold`, `Area Minimum`, atau parameter teknik spesifik untuk hasil terbaik.
                """)
        
        # Opsi untuk menampilkan threshold image (untuk debugging)
        # Gambar threshold membantu melihat bagaimana segmentasi dilakukan
        # Jika telur tidak terdeteksi dengan baik, sesuaikan threshold
        if st.checkbox("🔍 Tampilkan gambar intermediate (threshold / markers / overlay)"):
            st.subheader("🖼️ Gambar Intermediate")
            if teknik == "Hough Circle":
                overlay = img_np.copy()
                dp = teknik_params.get('dp', 1.0) if teknik_params else 1.0
                minDist = teknik_params.get('minDist', 20) if teknik_params else 20
                param1 = teknik_params.get('param1', 50) if teknik_params else 50
                param2 = teknik_params.get('param2', 30) if teknik_params else 30
                minRadius = teknik_params.get('minRadius', 10) if teknik_params else 10
                maxRadius = teknik_params.get('maxRadius', 50) if teknik_params else 50
                circles = cv2.HoughCircles(cv2.GaussianBlur(cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY), (9, 9), 0), cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                                           param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0, :]:
                        cv2.circle(overlay, (i[0], i[1]), i[2], (0, 255, 0), 3)
                st.image(Image.fromarray(overlay), caption="Hough circles overlay", width=400)
            elif teknik == "Watershed":
                # recompute watershed markers for visualization
                gray_ws = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                blur_ws = cv2.GaussianBlur(gray_ws, (9, 9), 0)
                ret, thresh_otsu = cv2.threshold(blur_ws, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                kernel = np.ones((3,3), np.uint8)
                opening = cv2.morphologyEx(thresh_otsu, cv2.MORPH_OPEN, kernel, iterations=2)
                sure_bg = cv2.dilate(opening, kernel, iterations=3)
                dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
                ret2, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
                sure_fg = np.uint8(sure_fg)
                unknown = cv2.subtract(sure_bg, sure_fg)
                ret3, markers = cv2.connectedComponents(sure_fg)
                markers = markers + 1
                markers[unknown == 255] = 0
                # colorize markers for visualization
                markers_viz = np.zeros_like(img_np)
                unique_markers = np.unique(markers)
                for idx, m in enumerate(unique_markers):
                    if m == 0:
                        continue
                    mask = markers == m
                    color = [int(x) for x in np.random.randint(0, 255, 3)]
                    markers_viz[mask] = color
                st.image(Image.fromarray(markers_viz), caption="Watershed markers (warna acak per objek)", width=400)
            else:
                # default show threshold
                thresh_pil = Image.fromarray(thresh)
                st.image(thresh_pil, caption="Gambar Threshold", width=400)
        
        # Gambar hasil deteksi
        st.subheader("🎯 Deteksi Telur")
        hasil = img_np.copy()
        if teknik == "Hough Circle":
            # Jika Hough Circle, gambar circles (gunakan parameter yang sama seperti saat deteksi)
            dp = teknik_params.get('dp', 1.0) if teknik_params else 1.0
            minDist = teknik_params.get('minDist', 20) if teknik_params else 20
            param1 = teknik_params.get('param1', 50) if teknik_params else 50
            param2 = teknik_params.get('param2', 30) if teknik_params else 30
            minRadius = teknik_params.get('minRadius', 10) if teknik_params else 10
            maxRadius = teknik_params.get('maxRadius', 50) if teknik_params else 50
            circles = cv2.HoughCircles(cv2.GaussianBlur(cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY), (9, 9), 0), cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                                       param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for idx_i, i in enumerate(circles[0, :]):
                    cv2.circle(hasil, (i[0], i[1]), i[2], (0, 255, 0), 3)
                    cv2.circle(hasil, (i[0], i[1]), 2, (0, 0, 255), 3)
                    cv2.putText(hasil, str(idx_i+1), (i[0]-10, i[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        elif teknik == "Ellipse Fit":
            # Gambar ellipse hasil fit
            for idx_i, cnt in enumerate(contours):
                if len(cnt) >= 5 and cv2.contourArea(cnt) > 0:
                    ellipse = cv2.fitEllipse(cnt)
                    cv2.ellipse(hasil, ellipse, (0,255,0), 3)
                    # centroid label
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        cv2.putText(hasil, str(idx_i+1), (cx-10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        else:
            # Gambar contours untuk teknik lainnya (dengan bounding boxes dan labels)
            for idx_i, cnt in enumerate(contours):
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                cv2.rectangle(hasil, (x, y), (x+w_box, y+h_box), (0,255,0), 2)
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    cv2.putText(hasil, str(idx_i+1), (cx-10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
                # draw contour outline
                cv2.drawContours(hasil, [cnt], -1, (0,255,0), 2)
        hasil_pil = Image.fromarray(hasil)
        st.image(hasil_pil, caption="Deteksi Telur", width=400)
        
        # If compare option is enabled, display results for each chosen technique
        if compare and compare_methods:
            st.subheader("🔬 Perbandingan Teknik Deteksi")
            cols = st.columns(len(compare_methods))
            for i, method in enumerate(compare_methods):
                j, cnts, thr = hitung_telur(img_np, filename, thresh_val, area_min, method, teknik_params)
                with cols[i]:
                    st.write(f"**{method}**")
                    st.write(f"Jumlah: {j}")
                    # Render overlay
                    overlay = img_np.copy()
                    if method == 'Hough Circle':
                        dp = teknik_params.get('dp', 1.0) if teknik_params else 1.0
                        minDist = teknik_params.get('minDist', 20) if teknik_params else 20
                        param1 = teknik_params.get('param1', 50) if teknik_params else 50
                        param2 = teknik_params.get('param2', 30) if teknik_params else 30
                        minRadius = teknik_params.get('minRadius', 10) if teknik_params else 10
                        maxRadius = teknik_params.get('maxRadius', 50) if teknik_params else 50
                        circles = cv2.HoughCircles(cv2.GaussianBlur(cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY), (9, 9), 0), cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                                                   param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
                        if circles is not None:
                            circles = np.uint16(np.around(circles))
                            for idx_i, c in enumerate(circles[0]):
                                cv2.circle(overlay, (c[0], c[1]), c[2], (0, 255, 0), 2)
                                cv2.putText(overlay, str(idx_i+1), (c[0]-10, c[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                    else:
                        for idx_i, cnt in enumerate(cnts):
                            x, y, w_box, h_box = cv2.boundingRect(cnt)
                            cv2.rectangle(overlay, (x, y), (x+w_box, y+h_box), (0,255,0), 2)
                            M = cv2.moments(cnt)
                            if M['m00'] != 0:
                                cx = int(M['m10']/M['m00'])
                                cy = int(M['m01']/M['m00'])
                                cv2.putText(overlay, str(idx_i+1), (cx-10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                    st.image(Image.fromarray(overlay), width=300)

        st.markdown("---")
        st.subheader("💾 Opsi Tambahan")
        
        # Opsi download hasil
        hasil_buffer = io.BytesIO()
        hasil_pil.save(hasil_buffer, format="PNG")
        hasil_buffer.seek(0)
        st.download_button("📥 Download Gambar Hasil Deteksi", hasil_buffer, file_name=f"deteksi_{filename}", mime="image/png")
        
        # Opsi simpan/load konfigurasi
        if custom:
            col3, col4 = st.columns(2)
            with col3:
                if st.button("💾 Simpan Konfigurasi"):
                    # Ensure teknik_params includes common shape thresholds
                    teknik_params_save = teknik_params.copy() if teknik_params else {}
                    teknik_params_save['minCircularity'] = float(min_circularity)
                    teknik_params_save['minSolidity'] = float(min_solidity)
                    config = {"threshold": thresh_val, "area_min": area_min, "teknik": teknik, "teknik_params": teknik_params_save}
                    import json
                    with open("config.json", "w") as f:
                        json.dump(config, f)
                    st.success(f"Konfigurasi disimpan: teknik={teknik}, threshold={thresh_val}, area_min={area_min}")
            with col4:
                if st.button("📂 Load Konfigurasi"):
                    try:
                        import json
                        with open("config.json", "r") as f:
                            config = json.load(f)
                        # Set session state so UI widgets update
                        st.session_state['config_threshold'] = config.get('threshold')
                        st.session_state['config_area_min'] = config.get('area_min')
                        st.session_state['config_teknik'] = config.get('teknik')
                        st.session_state['config_teknik_params'] = config.get('teknik_params')
                        st.success(f"Konfigurasi dimuat: teknik={config.get('teknik')}, threshold={config.get('threshold')}, area_min={config.get('area_min')}")
                        st.experimental_rerun()
                    except FileNotFoundError:
                        st.error("File config.json tidak ditemukan.")
        
    except Exception as e:
        st.error(f"Gambar tidak valid atau terjadi error: {e}")
else:
    st.info("Silakan upload gambar telur untuk memulai deteksi.")

# Footer attribution
st.markdown("---")
st.caption("Created by Galuh Adi Insani")
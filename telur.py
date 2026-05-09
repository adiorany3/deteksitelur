import cv2
import numpy as np
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="Deteksi Jumlah Telur",
    page_icon="🥚",
    layout="wide"
)

st.title("🥚 Deteksi dan Penghitung Jumlah Telur")
st.write(
    "Upload gambar telur, lalu sistem akan mendeteksi jumlah telur "
    "menggunakan HSV color masking dan template matching."
)


def create_egg_mask(image_rgb):
    """
    Membuat mask warna telur.
    Cocok untuk telur berwarna coklat/oranye seperti gambar contoh.
    """
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Rentang warna telur coklat/oranye
    lower_egg = np.array([0, 30, 50])
    upper_egg = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower_egg, upper_egg)

    kernel = np.ones((3, 3), np.uint8)

    # Membersihkan noise kecil
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Menutup lubang kecil pada area telur
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def create_egg_template(width=22, height=18):
    """
    Membuat template oval sederhana seperti bentuk telur.
    """
    template = np.zeros((height, width), dtype=np.float32)

    center = (width // 2, height // 2)
    axes = (width // 2 - 2, height // 2 - 2)

    cv2.ellipse(
        template,
        center,
        axes,
        0,
        0,
        360,
        1,
        -1
    )

    template = cv2.GaussianBlur(template, (0, 0), 2)

    return template


def detect_eggs_template(
    image_rgb,
    threshold=0.30,
    template_width=22,
    template_height=18,
    peak_window=23
):
    """
    Mendeteksi telur menggunakan template matching.
    Lebih cocok untuk telur yang saling menempel dibanding contour biasa.
    """
    mask = create_egg_mask(image_rgb)

    egg_map = cv2.GaussianBlur(mask, (0, 0), 3)
    egg_map = egg_map.astype(np.float32) / 255.0

    template = create_egg_template(
        width=template_width,
        height=template_height
    )

    result = cv2.matchTemplate(
        egg_map,
        template,
        cv2.TM_CCOEFF_NORMED
    )

    # Mencari titik puncak lokal
    peak_kernel = np.ones((peak_window, peak_window), dtype=np.float32)
    local_max = cv2.dilate(result, peak_kernel)

    peaks = ((result == local_max) & (result >= threshold)).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(peaks)

    centers = []

    for label_id in range(1, num_labels):
        ys, xs = np.where(labels == label_id)

        if len(xs) == 0:
            continue

        scores = result[ys, xs]
        best_idx = np.argmax(scores)

        x = int(xs[best_idx] + template_width / 2)
        y = int(ys[best_idx] + template_height / 2)
        score = float(scores[best_idx])

        centers.append((x, y, score))

    return centers, mask, result


def draw_detection(image_rgb, centers):
    """
    Menggambar lingkaran dan nomor pada telur yang terdeteksi.
    """
    output = image_rgb.copy()

    for idx, (x, y, score) in enumerate(centers, start=1):
        cv2.circle(output, (x, y), 12, (0, 255, 0), 2)

        cv2.putText(
            output,
            str(idx),
            (x - 8, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

    return output


uploaded_file = st.file_uploader(
    "Upload gambar telur",
    type=["jpg", "jpeg", "png"]
)


with st.sidebar:
    st.header("⚙️ Pengaturan Deteksi")

    threshold = st.slider(
        "Threshold Deteksi",
        min_value=0.15,
        max_value=0.80,
        value=0.30,
        step=0.01
    )

    template_width = st.slider(
        "Lebar Template Telur",
        min_value=18,
        max_value=40,
        value=22,
        step=1
    )

    template_height = st.slider(
        "Tinggi Template Telur",
        min_value=16,
        max_value=36,
        value=18,
        step=1
    )

    peak_window = st.slider(
        "Jarak Minimum Antar Telur",
        min_value=15,
        max_value=45,
        value=23,
        step=2
    )

    st.info(
        "Untuk gambar contoh, gunakan nilai default: "
        "Threshold 0.30, Template 22 x 18, dan Jarak Minimum 23."
    )


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(image)

    centers, mask, result = detect_eggs_template(
        image_rgb=image_rgb,
        threshold=threshold,
        template_width=template_width,
        template_height=template_height,
        peak_window=peak_window
    )

    output = draw_detection(image_rgb, centers)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Gambar Asli")
        st.image(image_rgb, use_container_width=True)

    with col2:
        st.subheader("Hasil Deteksi")
        st.image(output, use_container_width=True)

    st.success(f"Jumlah telur terdeteksi: {len(centers)}")

    with st.expander("Lihat Mask Warna Telur"):
        st.image(
            mask,
            caption="Mask area warna telur",
            use_container_width=True
        )

    with st.expander("Data Titik Deteksi"):
        data = []

        for i, (x, y, score) in enumerate(centers, start=1):
            data.append(
                {
                    "No": i,
                    "X": x,
                    "Y": y,
                    "Score": round(score, 3)
                }
            )

        st.dataframe(data, use_container_width=True)

else:
    st.warning("Silakan upload gambar telur terlebih dahulu.")

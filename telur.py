import cv2
import numpy as np
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="Deteksi Jumlah Telur",
    page_icon="🥚",
    layout="wide"
)

st.title("🥚 Aplikasi Deteksi dan Penghitung Telur")
st.write(
    "Upload gambar telur, lalu sistem akan mendeteksi jumlah telur "
    "menggunakan kombinasi HSV masking dan template matching."
)


def create_egg_mask(image_rgb):
    """
    Membuat mask warna telur berdasarkan HSV.
    Cocok untuk telur coklat/oranye seperti gambar contoh.
    """
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower_egg = np.array([0, 35, 60])
    upper_egg = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower_egg, upper_egg)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def create_egg_template(width=26, height=22):
    """
    Membuat template oval sederhana yang menyerupai bentuk telur dari atas.
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
    threshold=0.36,
    template_width=26,
    template_height=22,
    peak_window=31
):
    """
    Deteksi telur menggunakan template matching.
    Cocok untuk telur yang saling menempel karena tidak hanya bergantung
    pada contour luar.
    """
    mask = create_egg_mask(image_rgb)

    egg_map = cv2.GaussianBlur(mask, (0, 0), 3).astype(np.float32) / 255.0
    template = create_egg_template(template_width, template_height)

    result = cv2.matchTemplate(
        egg_map,
        template,
        cv2.TM_CCOEFF_NORMED
    )

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
    Menggambar hasil deteksi telur.
    """
    output = image_rgb.copy()

    for idx, (x, y, score) in enumerate(centers, start=1):
        cv2.circle(output, (x, y), 13, (0, 255, 0), 2)
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
        "Sensitivity / Threshold",
        min_value=0.20,
        max_value=0.80,
        value=0.36,
        step=0.01
    )

    template_width = st.slider(
        "Lebar template telur",
        min_value=18,
        max_value=40,
        value=26,
        step=1
    )

    template_height = st.slider(
        "Tinggi template telur",
        min_value=16,
        max_value=36,
        value=22,
        step=1
    )

    peak_window = st.slider(
        "Jarak minimum antar telur",
        min_value=15,
        max_value=45,
        value=31,
        step=2
    )

    st.info(
        "Untuk gambar telur contoh, nilai default biasanya menghasilkan "
        "sekitar 30 telur."
    )


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(image)

    centers, mask, result = detect_eggs_template(
        image_rgb,
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

    with st.expander("Lihat Mask Deteksi Warna"):
        st.image(mask, caption="Mask area warna telur", use_container_width=True)

    with st.expander("Data titik deteksi"):
        data = [
            {
                "No": i,
                "X": x,
                "Y": y,
                "Score": round(score, 3)
            }
            for i, (x, y, score) in enumerate(centers, start=1)
        ]
        st.dataframe(data, use_container_width=True)

else:
    st.warning("Silakan upload gambar telur terlebih dahulu.")

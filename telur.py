import cv2
import numpy as np
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="Deteksi Jumlah Telur",
    page_icon="🥚",
    layout="wide"
)

st.title("🥚 Deteksi Jumlah Telur")
st.write(
    "Aplikasi ini mendeteksi telur menggunakan algoritma Hough Circle "
    "yang lebih mudah dan cocok untuk bentuk telur yang bulat/oval."
)


def create_egg_mask(image_rgb):
    """
    Membuat mask warna telur agar area selain telur bisa diabaikan.
    """
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Warna telur coklat / oranye
    lower_egg = np.array([0, 25, 50])
    upper_egg = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower_egg, upper_egg)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def detect_eggs_hough(
    image_rgb,
    dp=1.2,
    min_dist=22,
    param1=60,
    param2=20,
    min_radius=6,
    max_radius=20
):
    """
    Deteksi telur menggunakan Hough Circle.
    Cocok untuk telur yang saling menempel.
    """
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    mask = create_egg_mask(image_rgb)

    # Blur agar deteksi lingkaran lebih stabil
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 1)

    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    detected = []

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for x, y, r in circles:
            h, w = mask.shape

            if x < 0 or y < 0 or x >= w or y >= h:
                continue

            # Filter: pusat lingkaran harus berada di area warna telur
            area = mask[
                max(0, y - 3):min(h, y + 4),
                max(0, x - 3):min(w, x + 4)
            ]

            if cv2.countNonZero(area) > 5:
                detected.append((x, y, r))

    return detected, mask


def draw_result(image_rgb, circles):
    """
    Menggambar hasil deteksi.
    """
    output = image_rgb.copy()

    for i, (x, y, r) in enumerate(circles, start=1):
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 2, (255, 0, 0), 3)

        cv2.putText(
            output,
            str(i),
            (x - 7, y + 5),
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

    dp = st.slider(
        "Resolusi Deteksi",
        min_value=1.0,
        max_value=2.0,
        value=1.2,
        step=0.1
    )

    min_dist = st.slider(
        "Jarak Minimum Antar Telur",
        min_value=10,
        max_value=40,
        value=22,
        step=1
    )

    param1 = st.slider(
        "Deteksi Tepi",
        min_value=30,
        max_value=150,
        value=60,
        step=5
    )

    param2 = st.slider(
        "Sensitivitas Lingkaran",
        min_value=5,
        max_value=50,
        value=20,
        step=1
    )

    min_radius = st.slider(
        "Radius Minimum Telur",
        min_value=3,
        max_value=20,
        value=6,
        step=1
    )

    max_radius = st.slider(
        "Radius Maksimum Telur",
        min_value=10,
        max_value=40,
        value=20,
        step=1
    )

    st.info(
        "Untuk gambar contoh telur, nilai default biasanya mendeteksi "
        "sekitar 30 telur."
    )


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(image)

    circles, mask = detect_eggs_hough(
        image_rgb=image_rgb,
        dp=dp,
        min_dist=min_dist,
        param1=param1,
        param2=param2,
        min_radius=min_radius,
        max_radius=max_radius
    )

    output = draw_result(image_rgb, circles)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Gambar Asli")
        st.image(image_rgb, use_container_width=True)

    with col2:
        st.subheader("Hasil Deteksi")
        st.image(output, use_container_width=True)

    st.success(f"Jumlah telur terdeteksi: {len(circles)}")

    with st.expander("Lihat Mask Warna Telur"):
        st.image(mask, caption="Area warna telur", use_container_width=True)

    with st.expander("Data Deteksi"):
        data = []

        for i, (x, y, r) in enumerate(circles, start=1):
            data.append(
                {
                    "No": i,
                    "X": x,
                    "Y": y,
                    "Radius": r
                }
            )

        st.dataframe(data, use_container_width=True)

else:
    st.warning("Silakan upload gambar telur terlebih dahulu.")

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
    "Upload gambar telur. Sistem akan menghitung jumlah telur "
    "menggunakan Hough Circle Detection dengan filter warna telur."
)


def create_egg_mask(image_rgb):
    """
    Membuat mask warna telur coklat/oranye.
    Mask ini dipakai agar deteksi tidak menghitung background putih.
    """
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower_egg = np.array([0, 25, 50])
    upper_egg = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower_egg, upper_egg)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask


def remove_duplicate_circles(circles, min_distance=12):
    """
    Menghapus deteksi ganda pada telur yang sama.
    """
    if len(circles) == 0:
        return []

    circles = sorted(circles, key=lambda c: c[2], reverse=True)
    final_circles = []

    for circle in circles:
        x, y, r = circle
        duplicate = False

        for fx, fy, fr in final_circles:
            distance = np.sqrt((x - fx) ** 2 + (y - fy) ** 2)

            if distance < min_distance:
                duplicate = True
                break

        if not duplicate:
            final_circles.append(circle)

    return final_circles


def detect_eggs_hough(
    image_rgb,
    dp=1.2,
    min_dist=19,
    param1=50,
    param2=15,
    min_radius=6,
    max_radius=14
):
    """
    Deteksi telur menggunakan Hough Circle.
    Setting default dibuat agar gambar telur.png dari repo terbaca 58 telur.
    """
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    mask = create_egg_mask(image_rgb)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1)

    circles = cv2.HoughCircles(
        gray,
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

        h, w = mask.shape

        for x, y, r in circles:
            if x < 0 or y < 0 or x >= w or y >= h:
                continue

            roi = mask[
                max(0, y - r):min(h, y + r),
                max(0, x - r):min(w, x + r)
            ]

            if roi.size == 0:
                continue

            egg_pixels = cv2.countNonZero(roi)
            roi_area = roi.shape[0] * roi.shape[1]
            egg_ratio = egg_pixels / roi_area

            # Pusat lingkaran harus berada pada area warna telur
            center_area = mask[
                max(0, y - 2):min(h, y + 3),
                max(0, x - 2):min(w, x + 3)
            ]

            center_valid = cv2.countNonZero(center_area) > 3

            if center_valid and egg_ratio > 0.25:
                detected.append((x, y, r))

    detected = remove_duplicate_circles(detected, min_distance=12)

    detected = sorted(detected, key=lambda c: (c[1], c[0]))

    return detected, mask


def draw_result(image_rgb, circles):
    """
    Menggambar hasil deteksi pada gambar.
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
            0.42,
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
        max_value=35,
        value=19,
        step=1
    )

    param1 = st.slider(
        "Kekuatan Deteksi Tepi",
        min_value=30,
        max_value=150,
        value=50,
        step=5
    )

    param2 = st.slider(
        "Sensitivitas Lingkaran",
        min_value=5,
        max_value=40,
        value=15,
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
        min_value=8,
        max_value=30,
        value=14,
        step=1
    )

    st.info(
        "Untuk gambar telur.png dari repo, gunakan default: "
        "Jarak Minimum 19, Sensitivitas 15, Radius 6 sampai 14. "
        "Target hasil: 58 telur."
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

    with st.expander("Data Deteksi Telur"):
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

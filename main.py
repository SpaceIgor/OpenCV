import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
import folium


def convert_to_lat_lon(x, y, map_center):
    lat = map_center[0] + y * 0.0001
    lon = map_center[1] + x * 0.0001
    return [lat, lon]


feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

cap = cv2.VideoCapture('video1.mp4')

ret, old_frame = cap.read()
if not ret:
    print("Не вдалося завантажити відео.")
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
mask = np.ones_like(old_frame)

tracked_points = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is not None and st is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        tracked_points.append(good_new)

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            frame = cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        # Показ кадру
        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Не вдалося обчислити оптичний потік для кадру.")
        break

cap.release()
cv2.destroyAllWindows()

if len(tracked_points) > 0:
    all_points = np.concatenate(tracked_points)
    x_coords = all_points[:, 0]
    y_coords = all_points[:, 1]

    t = np.arange(len(x_coords))
    spl_x = splrep(t, x_coords, s=0)
    spl_y = splrep(t, y_coords, s=0)

    pred_t = np.linspace(0, len(x_coords) - 1, num=500)
    pred_x = splev(pred_t, spl_x)
    pred_y = splev(pred_t, spl_y)

    plt.figure(figsize=(10, 6))
    plt.plot(pred_x, pred_y, label="Прогнозований шлях дрону", color='blue')
    plt.scatter(x_coords, y_coords, c='red', label="Відстежені точки", s=10)
    plt.title("Прогнозований шлях дрону")
    plt.xlabel("Координата X")
    plt.ylabel("Координата Y")
    plt.legend()

    plt.savefig("drone_path_plot.png")
    plt.close()

    map_center = [37.7749, -122.4194]  # Приклад координат (Сан-Франциско)
    m = folium.Map(location=map_center, zoom_start=14)

    for x, y in zip(pred_x, pred_y):
        lat_lon = convert_to_lat_lon(x, y, map_center)
        folium.CircleMarker(location=lat_lon, radius=5, color='blue', fill=True).add_to(m)

    m.save("drone_path_map.html")
else:
    print("Не вдалося знайти жодних точок для відстеження.")

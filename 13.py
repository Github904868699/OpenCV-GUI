# -*- coding: utf-8 -*-
"""
Circle & Color Detector – GUI Version (完整)
-------------------------------------------
* 实时检测蓝/黑圆形
* 左侧：摄像头选择 + HSV 滑块 + “显示蓝色掩膜” 按钮
* 右侧：带检测框 & FPS 的实时画面
* PyQt5 实现，无键盘快捷键
"""

import sys
import time
from typing import List
from pathlib import Path

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon

def resource_path(rel):
    """
    打包后返回 _MEIPASS\rel，开发环境返回脚本所在目录\rel
    """
    base = getattr(sys, "_MEIPASS", Path(__file__).parent)
    return str(Path(base, rel))



# --------------------- 检测参数 ---------------------
BLUE_LOWER = np.array([66, 69, 77], dtype=np.uint8)
BLUE_UPPER = np.array([128, 203, 199], dtype=np.uint8)
BLACK_S_MAX = 140
BLACK_V_MAX = 85
BLACK_LOWER = np.array([0, 0, 0], dtype=np.uint8)
BLACK_UPPER = np.array([180, BLACK_S_MAX, BLACK_V_MAX], dtype=np.uint8)
MIN_AREA, MAX_AREA = 500, 120_000
MIN_CIRCULARITY = 0.78
ROI_RATIO = 0.5

# --------------------- 工具函数 ---------------------

def list_camera_indices(max_index: int = 8) -> List[int]:
    valid = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            valid.append(idx)
            cap.release()
    return valid


def detect_circles(frame_bgr: np.ndarray) -> List[tuple]:
    results = []
    blur = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 180)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    for c in contours:
        area = cv2.contourArea(c)
        if not (MIN_AREA < area < MAX_AREA):
            continue
        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue
        circularity = 4 * np.pi * area / (peri * peri)
        if circularity < MIN_CIRCULARITY:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        x, y, r = int(x), int(y), int(r)
        roi_r = max(int(r * ROI_RATIO), 1)
        roi = hsv[max(y - roi_r, 0): y + roi_r, max(x - roi_r, 0): x + roi_r]
        if roi.size == 0:
            continue
        mean_h, mean_s, mean_v = np.mean(roi.reshape(-1, 3), axis=0)

        label, color_box = 'unknown', (200, 200, 200)
        if mean_s < BLACK_S_MAX and mean_v < BLACK_V_MAX:
            label, color_box = 'black', (30, 30, 30)
        elif (BLUE_LOWER <= np.array([mean_h, mean_s, mean_v])).all() and \
             (np.array([mean_h, mean_s, mean_v]) <= BLUE_UPPER).all():
            label, color_box = 'blue', (0, 165, 255)
        if label == 'unknown':
            continue

        cv2.circle(frame_bgr, (x, y), r, color_box, 2)
        cv2.putText(frame_bgr, label, (x - r, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box, 2)
        results.append((label, x, y, r))
    return results


def write_result_to_bus(idx: int, label: str, x: int, y: int, r: int):
    """预留的总线/网络写出接口"""
    pass

# --------------------- UI 组件 ---------------------

class HSVSlider(QtWidgets.QWidget):
    valueChanged = QtCore.pyqtSignal(int)
    def __init__(self, text: str, mn: int, mx: int, val: int, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QtWidgets.QLabel(text)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(mn, mx)
        self.slider.setValue(val)
        self.val_lbl = QtWidgets.QLabel(str(val))
        self.val_lbl.setFixedWidth(32)
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        layout.addWidget(self.val_lbl)
        self.slider.valueChanged.connect(self._on_change)
    def _on_change(self, v):
        self.val_lbl.setText(str(v))
        self.valueChanged.emit(v)
    def value(self):
        return self.slider.value()

class MaskWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("蓝色掩膜")
        self.resize(400, 300)
        self.label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        lay = QtWidgets.QVBoxLayout(self); lay.addWidget(self.label)
    def update_mask(self, mask_np):
        h, w = mask_np.shape
        qimg = QtGui.QImage(mask_np.data, w, h, w, QtGui.QImage.Format_Grayscale8)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.label.setPixmap(pix)

# --------------------- 主窗口 ---------------------

class MainWindow(QtWidgets.QWidget):
    FPS_CALC_INTERVAL = 30
    def __init__(self):
        super().__init__(None, QtCore.Qt.Window)

        self.setWindowTitle("蓝色掩膜")
        self.resize(400, 300)

        self.label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label)

        # ❷ 可选：窗口关闭后自动销毁，方便下次重新创建
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("GUI")
        self.resize(1200, 720)
        self.fps = 0.0

        # 布局
        hbox = QtWidgets.QHBoxLayout(self)
        self.ctrl_panel = QtWidgets.QFrame(); self.ctrl_panel.setFixedWidth(300)
        hbox.addWidget(self.ctrl_panel)
        self.video_lbl = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter); self.video_lbl.setMinimumSize(640, 480)
        hbox.addWidget(self.video_lbl, 1)

        cp = QtWidgets.QVBoxLayout(self.ctrl_panel)
        # 摄像头
        cam_group = QtWidgets.QGroupBox("Camera"); cam_layout = QtWidgets.QVBoxLayout(cam_group)
        self.cam_combo = QtWidgets.QComboBox(); cam_layout.addWidget(self.cam_combo)
        cp.addWidget(cam_group)
        # HSV 滑块
        hsv_group = QtWidgets.QGroupBox("HSV Blue"); hsv_layout = QtWidgets.QVBoxLayout(hsv_group)
        self.sliders = {}
        for name, mn, mx, val in [
            ("minH",0,179,int(BLUE_LOWER[0])), ("maxH",0,179,int(BLUE_UPPER[0])),
            ("minS",0,255,int(BLUE_LOWER[1])), ("maxS",0,255,int(BLUE_UPPER[1])),
            ("minV",0,255,int(BLUE_LOWER[2])), ("maxV",0,255,int(BLUE_UPPER[2]))]:
            s = HSVSlider(name, mn, mx, val); s.valueChanged.connect(self.sync_hsv)
            hsv_layout.addWidget(s); self.sliders[name] = s
        cp.addWidget(hsv_group, 1)
        # === HSV Black 组 ===
        black_group = QtWidgets.QGroupBox("HSV Black")
        black_layout = QtWidgets.QVBoxLayout(black_group)
        self.black_sliders = {}
        # 添加 minH_B, minS_B, minV_B, maxH_B 滑块
        for name, mn, mx, val in [
            ("minH_B", 0, 179, 0), ("maxH_B", 0, 179, 180),
            ("minS_B", 0, 255, 0), ("maxS_B", 0, 255, BLACK_S_MAX),
            ("minV_B", 0, 255, 0), ("maxV_B", 0, 255, BLACK_V_MAX)]:
            s = HSVSlider(name, mn, mx, val)
            s.valueChanged.connect(self.sync_black_hsv)  # 绑定黑色 HSV 同步函数
            black_layout.addWidget(s)
            self.black_sliders[name] = s
        cp.addWidget(black_group)
        # 掩膜按钮
        self.mask_btn = QtWidgets.QPushButton("蓝色掩膜")
        self.mask_btn.clicked.connect(lambda: self.toggle_mask('blue'))
        cp.addWidget(self.mask_btn)
        self.mask_btn_black = QtWidgets.QPushButton("黑色掩膜")
        self.mask_btn_black.clicked.connect(lambda: self.toggle_mask('black'))
        cp.addWidget(self.mask_btn_black)
        cp.addStretch(1)


        # 初始化摄像头 & 计时器
        self.capture = None
        self.populate_cameras(); self.cam_combo.currentIndexChanged.connect(self.open_camera)
        self.open_camera()
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self.on_timer); self.timer.start(30)
        self.frame_cnt = 0; self.last_time = time.time()

    # ---------- 摄像头 ----------
    def populate_cameras(self):
        for idx in list_camera_indices():
            self.cam_combo.addItem(f"Camera {idx}", idx)
        if self.cam_combo.count() == 0:
            QtWidgets.QMessageBox.critical(self, "Error", "No camera found"); sys.exit(1)

    def open_camera(self):
        idx = self.cam_combo.currentData()
        if self.capture: 
            self.capture.release()
        self.capture = cv2.VideoCapture(idx)
        # 设置分辨率，可按需修改
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.capture.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", f"Cannot open camera {idx}")

    # ---------- HSV 同步 ----------
    def sync_hsv(self):
        """把滑块数值同步到全局 BLUE_LOWER / BLUE_UPPER 阈值"""
        BLUE_LOWER[:] = [
            self.sliders["minH"].value(),
            self.sliders["minS"].value(),
            self.sliders["minV"].value(),
        ]
        BLUE_UPPER[:] = [
            self.sliders["maxH"].value(),
            self.sliders["maxS"].value(),
            self.sliders["maxV"].value(),
        ]

    # ---------- HSV 黑色同步 ----------
    def sync_black_hsv(self):
        """把滑块数值同步到全局黑色 HSV 阈值"""
        global BLACK_S_MAX, BLACK_V_MAX, BLACK_LOWER, BLACK_UPPER
        BLACK_S_MAX = self.black_sliders["maxS_B"].value()
        BLACK_V_MAX = self.black_sliders["maxV_B"].value()
        BLACK_LOWER[:] = [
            self.black_sliders["minH_B"].value(),
            self.black_sliders["minS_B"].value(),
            self.black_sliders["minV_B"].value(),
        ]
        BLACK_UPPER[:] = [
            self.black_sliders["maxH_B"].value(),
            self.black_sliders["maxS_B"].value(),
            self.black_sliders["maxV_B"].value(),
        ]

    # ---------- 掩膜窗口 ----------
    # def toggle_mask(self):
    #     # 如果不存在或被关闭销毁，就重新创建
    #     if self.mask_win is None:
    #         self.mask_win = MaskWindow()
    #         # 关闭时清空引用
    #         self.mask_win.destroyed.connect(lambda _: setattr(self, 'mask_win', None))

    #     # 显示并置顶
    #     self.mask_win.show()
    #     self.mask_win.raise_()
    #     self.mask_win.activateWindow()
    def toggle_mask(self, which='blue'):
    # which ∈ {'blue', 'black'}
        attr = f"mask_win_{which}"
        if getattr(self, attr, None) is None:
            setattr(self, attr, MaskWindow())
            getattr(self, attr).destroyed.connect(lambda _: setattr(self, attr, None))
        win = getattr(self, attr)
        win.setWindowTitle("蓝色掩膜" if which=='blue' else "黑色掩膜")
        win.show(); 
        win.raise_(); 
        win.activateWindow()

    # ---------- 主循环 ----------
    def on_timer(self):
        if not self.capture or not self.capture.isOpened():
            return

        ok, frame = self.capture.read()
        if not ok:
            return

        # 圆形检测
        results = detect_circles(frame)
        for i, (lab, x, y, r) in enumerate(results):
            write_result_to_bus(i, lab, x, y, r)

        # FPS 计算
        self.frame_cnt += 1
        if self.frame_cnt % self.FPS_CALC_INTERVAL == 0:
            now = time.time()
            self.fps = self.FPS_CALC_INTERVAL / (now - self.last_time)
            self.last_time = now
        cv2.putText(frame, f"{self.fps:.1f} FPS", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 掩膜窗口实时更新
        # if self.mask_win and self.mask_win.isVisible():
        #     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #     mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
        #     self.mask_win.update_mask(mask)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 蓝色掩膜
        if getattr(self, 'mask_win_blue', None) and self.mask_win_blue.isVisible():
            mask_blue = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
            self.mask_win_blue.update_mask(mask_blue)

        # 黑色掩膜：H 不需要管，S/V 只看“最大值”
        if getattr(self, 'mask_win_black', None) and self.mask_win_black.isVisible():
            mask_black = cv2.inRange(hsv, BLACK_LOWER, BLACK_UPPER)
            self.mask_win_black.update_mask(mask_black)


        # 显示到 QLabel
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qimg = QtGui.QImage(frame_rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        self.video_lbl.setPixmap(
            QtGui.QPixmap.fromImage(qimg).scaled(
                self.video_lbl.size(), QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation))

    # ---------- 关闭事件 ----------
    def closeEvent(self, event):
        if self.capture:
            self.capture.release()
        event.accept()

# --------------------- 程序入口 ---------------------

def main():
    icon_path = resource_path("Camera.ico")
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QIcon(icon_path))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()


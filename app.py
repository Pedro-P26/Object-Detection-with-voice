import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox
import datetime
import os
import pyttsx3
import threading

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


class YOLOApp:
    def __init__(self, window):
        self.window = window
        self.window.title("YOLOv8 - Detetor de Objetos e Mãos")

        self.model_obj = YOLO("weights/yolov8s.pt")
        self.model_pose = YOLO("weights/yolov8s-pose.pt")

        self.cap = cv2.VideoCapture(0)
        self.label = tk.Label(window)
        self.label.pack()

        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 170)

        self.detected_objects = {}
        os.makedirs("detected_objects", exist_ok=True)

        self.window.bind("<g>", self.ask_generate_report)
        self.update_frame()
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def speak(self, text):
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except RuntimeError:
            pass

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # ---------- MODELO 1: Deteção de objetos ----------
        results_obj = self.model_obj(frame, stream=True)
        for r in results_obj:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.model_obj.names[cls]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if label not in self.detected_objects:
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    obj_crop = frame[y1:y2, x1:x2]
                    image_path = f"detected_objects/{label}.jpg"
                    cv2.imwrite(image_path, obj_crop)

                    self.detected_objects[label] = {
                        "confidence": conf,
                        "time": timestamp,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "width": width,
                        "height": height,
                        "area": area,
                        "image_path": image_path
                    }

                    message = (
                        f"Objeto detetado: {label}. "
                        f"Confiança: {conf:.2f}. "
                        f"Hora: {timestamp}. "
                        f"Localização de x {x1} a {x2}, e de y {y1} a {y2}. "
                        f"Tamanho: {width} por {height} pixeis. "
                        f"Área: {area} pixeis quadrados."
                    )

                    threading.Thread(target=self.speak, args=(message,), daemon=True).start()

        # ---------- MODELO 2: Deteção de mãos completas ----------
        results_pose = self.model_pose(frame)
        for r in results_pose:
            if r.keypoints is not None:
                keypoints = r.keypoints.xy[0]

                hands = {
                    "hand_left": {"elbow": 7, "wrist": 9},
                    "hand_right": {"elbow": 8, "wrist": 10}
                }

                for label, indices in hands.items():
                    try:
                        elbow = keypoints[indices["elbow"]]
                        wrist = keypoints[indices["wrist"]]

                        if wrist[0] <= 0 or wrist[1] <= 0 or elbow[0] <= 0 or elbow[1] <= 0:
                            continue

                        wx, wy = int(wrist[0]), int(wrist[1])
                        ex, ey = int(elbow[0]), int(elbow[1])

                        dx, dy = wx - ex, wy - ey
                        fx, fy = wx + int(0.6 * dx), wy + int(0.6 * dy)

                        xs = [ex, wx, fx]
                        ys = [ey, wy, fy]

                        x1, y1 = max(min(xs) - 30, 0), max(min(ys) - 30, 0)
                        x2, y2 = min(max(xs) + 30, frame.shape[1]), min(max(ys) + 30, frame.shape[0])

                        conf = 0.99

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        if label not in self.detected_objects:
                            width = x2 - x1
                            height = y2 - y1
                            area = width * height
                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                            obj_crop = frame[y1:y2, x1:x2]
                            image_path = f"detected_objects/{label}.jpg"
                            cv2.imwrite(image_path, obj_crop)

                            self.detected_objects[label] = {
                                "confidence": conf,
                                "time": timestamp,
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                                "width": width,
                                "height": height,
                                "area": area,
                                "image_path": image_path
                            }

                            message = (
                                f"Objeto detetado: {label}. "
                                f"Confiança: {conf:.2f}. "
                                f"Hora: {timestamp}. "
                                f"Localização de x {x1} a {x2}, e de y {y1} a {y2}. "
                                f"Tamanho: {width} por {height} pixeis. "
                                f"Área: {area} pixeis quadrados."
                            )

                            threading.Thread(target=self.speak, args=(message,), daemon=True).start()
                    except:
                        pass

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)
        self.window.after(10, self.update_frame)

    # Ao clicar na tecla "g" vai aparecer uma caixa a questionar se quer gerar um relatório
    def ask_generate_report(self, event=None):
        if not self.detected_objects:
            messagebox.showinfo("Relatório", "Nenhum objeto detetado ainda.")
            return

        answer = messagebox.askyesno("Gerar Relatório", "Deseja gerar um relatório com os objetos detetados?")
        if answer:
            self.generate_report()

    # Gera o relatório com os objetos detetados
    def generate_report(self):
        c = canvas.Canvas("relatorio_objetos.pdf", pagesize=A4)
        width, height = A4
        y = height - 50

        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y, "Relatório de Objetos Detetados (YOLOv8)")
        y -= 30
        c.setFont("Helvetica", 12)

        for label, data in self.detected_objects.items():
            if y < 150:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 12)

            c.drawString(50, y, f"Objeto: {label}")
            y -= 20
            c.drawString(50, y, f"Confiança: {data['confidence']:.2f}")
            y -= 20
            c.drawString(50, y, f"Hora: {data['time']}")
            y -= 20
            c.drawString(50, y, f"Localização: x1={data['x1']}, y1={data['y1']}, x2={data['x2']}, y2={data['y2']}")
            y -= 20
            c.drawString(50, y, f"Tamanho da caixa: {data['width']}x{data['height']} px")
            y -= 20
            c.drawString(50, y, f"Área: {data['area']} px²")
            y -= 20

            try:
                img = ImageReader(data['image_path'])
                c.drawImage(img, 300, y - 10, width=120, height=120, preserveAspectRatio=True)
            except Exception:
                c.drawString(300, y, "[Erro ao carregar imagem]")

            y -= 140

        c.save()
        messagebox.showinfo("Relatório", "Relatório PDF gerado com sucesso: relatorio_objetos.pdf")

    def on_close(self):
        self.cap.release()
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOApp(root)
    root.mainloop()

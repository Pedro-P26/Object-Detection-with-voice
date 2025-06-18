import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox
import datetime
import os
import pyttsx3
import threading

import ollama  # 🚀 Integração com Ollama!

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
            print(f"[TTS] Falando: {text}")
            self.engine.say(text)
            self.engine.runAndWait()
        except RuntimeError as e:
            print("Erro no TTS:", e)

    def generate_description_with_ollama(self, label, conf, x1, y1, x2, y2, width, height, area, timestamp):
        """Gera uma descrição mais natural usando o modelo Llama3 via Ollama."""
        prompt = (
            f"Imagine que você é um especialista em monitoramento de vídeo escrevendo um relatório detalhado em português. "
            f"Descreva de forma clara, humana e amigável a detecção de um objeto do tipo '{label}' com uma confiança de "
            f"{conf:.2f}. Inclua a data e hora ({timestamp}) em que foi detectado e mencione as coordenadas "
            f"({x1}, {y1}) até ({x2}, {y2}). Fale sobre o tamanho da caixa delimitadora ({width}x{height} pixels) "
            f"e a área total ({area} pixels quadrados). Finalize com uma breve conclusão indicando a relevância dessa detecção."
        )
        print(f"[LOG] Enviando prompt para o Ollama:\n{prompt}\n")
        try:
            response = ollama.chat(
                model="llama3",
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            print("[LOG] Resposta JSON completa:", response)
            frase = response.get('message', {}).get('content', '[Sem resposta]')
            print(f"[LOG] Resposta do Ollama:\n{frase}\n")
            return frase
        except Exception as e:
            print("Erro ao chamar Ollama:", e)
            return (
                f"Objeto detetado: {label}. "
                f"Confiança: {conf:.2f}. "
                f"Hora: {timestamp}. "
                f"Localização de x {x1} a {x2}, e de y {y1} a {y2}. "
                f"Tamanho: {width} por {height} pixeis. "
                f"Área: {area} pixeis quadrados."
            )

    def generate_and_speak(self, label, conf, x1, y1, x2, y2, width, height, area, timestamp):
        """Executa a geração da descrição em background e chama o TTS no thread principal."""
        def background_task():
            message = self.generate_description_with_ollama(
                label, conf, x1, y1, x2, y2, width, height, area, timestamp
            )
            print(f"[LOG] Frase gerada para TTS:\n{message}\n")
            self.detected_objects[label]['description'] = message
            self.window.after(0, self.speak, message)

        threading.Thread(target=background_task, daemon=True).start()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

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

                if label not in self.detected_objects and conf >= 0.65:
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
                        "image_path": image_path,
                        "description": ""
                    }

                    self.generate_and_speak(label, conf, x1, y1, x2, y2, width, height, area, timestamp)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)
        self.window.after(10, self.update_frame)

    def split_text(self, text, max_length=90):
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= max_length:
                current_line += (word + " ")
            else:
                lines.append(current_line.strip())
                current_line = word + " "
        if current_line:
            lines.append(current_line.strip())
        return lines

    def ask_generate_report(self, event=None):
        if not self.detected_objects:
            messagebox.showinfo("Relatório", "Nenhum objeto detetado ainda.")
            return

        answer = messagebox.askyesno("Gerar Relatório", "Deseja gerar um relatório com os objetos detetados?")
        if answer:
            self.generate_report()

    def generate_report(self):
        c = canvas.Canvas("relatorio_objetos.pdf", pagesize=A4)
        width, height = A4
        y = height - 50

        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y, "Relatório de Objetos Detetados (YOLOv8)")
        y -= 30
        c.setFont("Helvetica", 12)

        for label, data in self.detected_objects.items():
            if data['confidence'] < 0.65:
                continue

            if y < 200:
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

            description = data.get('description', '[Descrição não disponível]')
            desc_lines = self.split_text(description, max_length=90)
            for idx, line in enumerate(desc_lines):
                prefix = "Descrição: " if idx == 0 else "           "
                c.drawString(50, y, f"{prefix}{line}")
                y -= 20

            y -= 10  # Espaço antes da imagem
            if y < 170:
                c.showPage()
                y = height - 200
                c.setFont("Helvetica", 12)

            try:
                img = ImageReader(data['image_path'])
                c.drawImage(img, 50, y - 120, width=120, height=120, preserveAspectRatio=True)
            except Exception:
                c.drawString(50, y, "[Erro ao carregar imagem]")

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

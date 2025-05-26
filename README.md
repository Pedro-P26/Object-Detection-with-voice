# Detetor de Objetos e Mãos com YOLOv8 e Interface Gráfica

## 🧠 Funcionalidades

- Deteção em tempo real de objetos e mãos via webcam usando dois modelos YOLOv8.
- Feedback por voz com detalhes do objeto (nome, confiança, hora, dimensões).
- Guarda localmente automaticamente os objetos que o programa deteta.
- Gerar relatório em PDF (pressionar tecla **g** para gerar).
- Interface gráfica intuitiva com `Tkinter`.

## 📁 Estrutura do projeto

```
📦 Projeto
├── app.py                  
├── requirements.txt         
├── detected_objects/        
├── weights/
│   ├── yolov8s.pt           
│   └── yolov8s-pose.pt      
├── relatorio_objetos.pdf    
└── README.md
```

## 💬 Requisitos

- Python 3.8 ou superior
- Webcam funcional
- Sistema com som ativado (para feedback de voz)

## 🧪 Instalação

```bash
git clone https://github.com/Pedro-P26/Object-Detection-with-voice.git
cd OBJECT-DETECTION-WITH-VOICE

# Instalar dependências
pip install -r requirements.txt
```

Além disso, é necessário adicionar estas versões do YOLOv8 em `weights/`:
- `yolov8s.pt` → para detetar objetos
- `yolov8s-pose.pt` → para detetar mãos

## 🚀 Execução

```bash
python app.py
```

- A deteção será feita automaticamente pela webcam.
- Pressionar a tecla `g` para gerar um relatório com os objetos detetados.

## 📄 Geração de Relatório PDF

A aplicação gera um relatório PDF com:
- Nome do objeto
- Confiança
- Hora da deteção
- Localização e dimensão da caixa
- Imagem recortada do objeto

## 📦 Dependências principais

- `ultralytics` (YOLOv8)
- `opencv-python`
- `tkinter` (integrado no Python)
- `pyttsx3` (síntese de voz)
- `reportlab` (para criar PDF)

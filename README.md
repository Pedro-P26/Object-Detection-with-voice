# Detetor de Objetos e Mãos com YOLOv8, Voz e Descrição por IA

## 🧠 Funcionalidades

* Deteção em tempo real de objetos e mãos via webcam usando dois modelos YOLOv8.
* Feedback por voz com detalhes do objeto (nome, confiança, hora, dimensões).
* Descrição detalhada e natural dos objetos detetados usando IA (via **Ollama** e **modelo Llama3**).
* Guarda automaticamente imagens dos objetos detetados.
* Geração de relatório em PDF com as informações e imagens dos objetos.
* Interface gráfica intuitiva com `Tkinter`.

## 📁 Estrutura do projeto

```
📦 OBJECT-DETECTION-WITH-VOICE
├── app.py                      
├── requirements.txt            
├── README.md                   
├── reiniciar_ollama.bat        # Script para iniciar o ollama e deixa a porta do ollama desocupada 
├── relatorio_objetos.pdf       
├── detected_objects/         
│   └── *.jpg                   
├── weights/                    
│   ├── yolov8s.pt              
│   ├── yolov8s-pose.pt         
│   └── yolov8n.pt              
```

## 💬 Requisitos

* Python 3.8 ou superior
* Webcam funcional
* Sistema com som ativado
* Servidor **Ollama** instalado e funcional (porta 11434)
* Modelos YOLOv8 (baixar manualmente)

## 🧪 Instalação

```bash
git clone https://github.com/Pedro-P26/Object-Detection-with-voice.git
cd OBJECT-DETECTION-WITH-VOICE

# Instalar dependências
pip install -r requirements.txt
```

Além disso, coloque na pasta `weights/` os seguintes modelos:

* `yolov8s.pt` – para detetar objetos
* `yolov8s-pose.pt` – para detetar mãos

## 🦙 Iniciar o Servidor Ollama

Antes de correr a aplicação, **certifique-se que o servidor Ollama está ativo.**
Pode usar o script incluído para garantir isso:

```bash
reiniciar_ollama.bat
```

Este script:

* Fecha qualquer processo na porta `11434`
* Inicia o servidor Ollama (`ollama serve`)

Certifique-se de que o modelo `llama3` está instalado:

```bash
ollama run llama3
```

> ⚠️ A aplicação irá utilizar este modelo para gerar descrições dos objetos detetados.

## 🚀 Execução da Aplicação

```bash
python app.py
```

* A câmara será ativada automaticamente.
* Quando um novo objeto for detetado com confiança ≥ 65%, será guardado, descrito e falado.
* Pressione a tecla **`g`** para gerar o relatório com os objetos detetados.

## 📄 Relatório PDF

O relatório inclui:

* Nome, confiança e hora da deteção
* Localização, dimensão e área
* Imagem recortada do objeto
* Descrição textual gerada pela IA

O ficheiro final será `relatorio_objetos.pdf`.

## 📦 Dependências principais

* [`ollama`](https://ollama.com/) – comunicação com LLM local
* `ultralytics` – modelo YOLOv8
* `opencv-python` – processamento de vídeo
* `Pillow` – manipulação de imagem
* `pyttsx3` – síntese de voz offline
* `reportlab` – geração de PDF

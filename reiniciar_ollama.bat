@echo off
echo [🔍] A procurar processos a usar a porta 11434...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :11434') do (
    echo [🛑] A terminar processo com PID %%a...
    taskkill /PID %%a /F >nul 2>&1
)
echo [🚀] A iniciar o servidor Ollama...
start "" /B ollama serve
echo [✅] Servidor iniciado. Pressiona qualquer tecla para sair.
pause >nul

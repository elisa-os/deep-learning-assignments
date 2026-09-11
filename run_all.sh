echo "Iniciando Parte 0..." && uv run pa1 0 && \
    echo "Iniciando Parte 1..." && uv run pa1 1 && \
    echo "Iniciando Parte 2..." && uv run pa1 2 && \
    echo "Iniciando Parte 3 (Ablações)..." && uv run pa1-ablation && \
    echo "Iniciando Parte 4 (Mosaico)..." && uv run pa1-mosaic && \
    echo "Iniciando Parte 5 (Intervenção Atrous)..." && uv run pa1 5 && \
    echo "Iniciando Parte 6 (Teste de Estresse)..." && uv run pa1-stress && \
    echo "🎉 TODO O PIPELINE EXECUTADO COM SUCESSO! 🎉"

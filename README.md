# Detector UEBA: Detecção de Anomalias em Logs de Autenticação

Pipeline de detecção de anomalias comportamentais (User and Entity Behavior Analytics - UEBA) baseado em logs de autenticação do Active Directory com limiares dinâmicos.

Desenvolvido no âmbito do Mestrado Profissional em Engenharia Elétrica da Universidade de Brasília (UnB/FT/ENE).

## 🎯 Objetivo
Fornecer uma ferramenta de auditoria comportamental voltada para analistas de Inteligência, Contrainteligência e Segurança Pública, focando na identificação de desvios sutis de conduta e potenciais vazamentos de dados por ameaças internas (insider threats).

## 📂 Estrutura do Repositório

```text
├── config/             # Configurações experimentais (YAML)
├── core/               # Núcleo da arquitetura
│   ├── data/           # Ingestão, adaptação e split cronológico/aleatório
│   ├── features/       # Extração e engenharia de atributos comportamentais
│   ├── models/         # Modelos de detecção (Autoencoder MLP, Random Forest, etc.)
│   └── trainer.py      # Protocolo de treino anti-leakage e calibração de limiares
├── scripts/            # Scripts de execução, inferência e avaliação
│   ├── detectar.py     # Pipeline principal de execução experimental
│   ├── evaluate_performance.py # Avaliação e métricas
│   └── inferir_usual.py        # Inferência em novos eventos
├── tests/              # Testes unitários e de causalidade (pytest)
├── requirements.txt    # Dependências do projeto Python
└── CONTEXTO_DETECTOR_UEBA_01_11_2026.md # Metadados acadêmicos e notas técnicas
```

## 🚀 Instalação e Uso

1. Crie um ambiente virtual Python (versão recomendada: 3.10.x):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # ou no Windows:
   .venv\Scripts\activate
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Executar a suíte de testes:
   ```bash
   pytest tests/
   ```

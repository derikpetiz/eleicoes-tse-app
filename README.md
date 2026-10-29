# 🗳️ Painel de Auditoria Eleitoral & Análise Estatística TSE

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Concluído-success.svg)]()

Aplicação web interativa desenvolvida em **Python** e **Streamlit** para consolidação, simulação contrafactual e auditoria estatística/econométrica dos microdados oficiais dos Boletins de Urna (BUs) disponibilizados pelo Tribunal Superior Eleitoral (TSE).

O painel reúne testes estatísticos avançados, diagnósticos de conformidade estocástica e visualizações operacionais para análise detalhada do comportamento do eleitorado entre o **1º Turno** e o **2º Turno**.

---

## 📌 Índice

- [Funcionalidades Principais](#-funcionalidades-principais)
- [Estrutura das Abas do Painel](#-estrutura-das-abas-do-painel)
- [Modelos Estatísticos & Conceitos Científicos](#-modelos-estatísticos--conceitos-científicos)
- [Pré-requisitos e Tecnologias](#-pré-requisitos-e-tecnologias)
- [Instalação e Execução Local](#-instalação-e-execução-local)
- [Publicação e Deploy no Streamlit Cloud](#-publicação-e-deploy-no-streamlit-cloud)
- [Licença](#-licença)

---

## 🚀 Funcionalidades Principais

* **Filtros Dinâmicos:** Filtragem regional por Região e Estado (UF) em tempo real.
* **Processamento Automatizado de Microdados:** Leitura de arquivos compactados (`.ZIP`), tabelas estáticas (`.CSV`) ou bases otimizadas em formato Parquet (`.PARQUET`).
* **Detecção Automática de Outliers:** Identificação imediata de seções com votação unânime ($0$ votos) e ganhos desproporcionais por município.
* **Simulador Contrafactual:** Ajuste fino via *sliders* para modelar taxas de transferência e migração dos votos de terceiros candidatos, brancos/nulos e ativação de abstêmios.
* **Testes Estatísticos de Conformidade:** Avaliação de aderência dos dígitos finais pela **Lei de Newcomb-Benford (2º Dígito)** e testes não-paramétricos de distribuição.

---

## 📊 Estrutura das Abas do Painel

1. **`1. Resumo & Alertas Principais`**: Visão executiva detalhando seções eleitorais residuais ($\le 10$ votos), alertas de unanimidade ($0$ votos) e ranking dos maiores ganhos de votos por município.
2. **`2. Modelo de Migração de Votos`**: Simulador contrafactual para projeções de cenários alternativos baseados na absorção dos eleitores do 1º Turno.
3. **`3. Voto Deputados vs Presidente`**: Cruzamento ideológico entre votações proporcionais (Deputados) e majoritárias (Presidente) para cálculo do indicador de incongruência ($Z\text{-Score}$).
4. **`4. Lista Detalhada de Anomalias`**: Mapeamento tabular das seções auditadas e gráfico de distribuição frequencial da **Lei de Benford (2º Dígito)** com cálculo do Chi-Quadrado ($\chi^2$).
5. **`5. Auditoria Operacional & Biometria`**: Avaliação das $8,63$ milhões de habilitações manuais por ano de nascimento (sem digital) e análise de latência das seções com votação estendida pós-17h.
6. **`6. Observações do Banco & Diagnóstico 1T vs 2T`**: Relatório sintético comparativo em barras e parecer técnico fundamentado em Econometria Eleitoral e Teoria das Probabilidades.

---

## 🎓 Modelos Estatísticos & Conceitos Científicos

* **Lei dos Grandes Números (LGN):** Avalia a estabilidade da média municipal e a probabilidade estocástica $P(X=0)$ de zerar votos em seções urbanas densas.
* **Lei de Newcomb-Benford (2º Dígito):** Validação logarítmica da frequência natural dos segundos dígitos da contagem de votos.
* **Métrica Z-Score:** Padronização estatística do desvio em relação à média ($Z > 4.0$) para detecção de *outliers* severos.
* **Teste Chi-Quadrado ($\chi^2$):** Teste de hipótese para aderência frequencial observada vs esperada.

---

## 🛠️ Pré-requisitos e Tecnologias

* **Python 3.10+**
* **Streamlit** (Interface Web)
* **Pandas & NumPy** (Manipulação e agregação de dados)
* **SciPy** (Testes de hipótese e estatística inferencial)
* **Plotly** (Gráficos interativos e dashboards)
* **PyArrow / Fastparquet** (Leitura acelerada de arquivos `.parquet`)

---

## 💻 Instalação e Execução Local

1. **Clone este repositório:**
   ```bash
   git clone [https://github.com/derikpetiz/eleicoes-tse-app.git](https://github.com/derikpetiz/eleicoes-tse-app.git)
   cd eleicoes-tse-app

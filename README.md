# 🗳️ Painel de Auditoria Eleitoral & Análise Estatística TSE

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Concluído-success.svg)]()

Aplicação web interativa desenvolvida em **Python** e **Streamlit** para consolidação, simulação contrafactual e auditoria estatística/econométrica dos microdados oficiais dos Boletins de Urna (BUs) disponibilizados pelo Tribunal Superior Eleitoral (TSE).

O painel reúne testes estatísticos avançados, diagnósticos de conformidade estocástica e visualizações operacionais para análise detalhada do comportamento do eleitorado entre o **1º Turno** e o **2º Turno**.

---

## 📌 Índice

- [Descrição do Projeto](#-descrição-do-projeto)
- [Capturas de Tela e Interface](#-capturas-de-tela-e-interface)
- [Principais Insights & Achados da Auditoria](#-principais-insights--achados-da-auditoria)
- [Estrutura das Abas do Painel](#-estrutura-das-abas-do-painel)
- [Modelos Estatísticos & Conceitos Científicos](#-modelos-estatísticos--conceitos-científicos)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Instalação e Execução Local](#-instalação-e-execução-local)
- [Publicação e Deploy](#-publicação-e-deploy)
- [Licença](#-licença)

---

## 📝 Descrição do Projeto

Este projeto foi concebido para auditar e analisar padrões de votação nos microdados do TSE no 2º Turno. A ferramenta permite checar a consistência numérica e operacional dos dados por meio de modelagem econométrica, testes não-paramétricos, análise de congruência de voto cruzado (proporcional vs majoritário) e avaliação de aderência à **Lei de Benford**.

---

## 📸 Capturas de Tela e Interface

Abaixo estão apresentadas as telas consolidadas do dashboard interativo, contemplando o Resumo Executivo, o Mapeamento de Seções Residuais, a Comparação por Turnos e o Teste de Benford (2º Dígito):

<p align="center">
  <img src="https://raw.githubusercontent.com/derikpetiz/eleicoes-tse-app/main/dashboard_consolidado.png" alt="Interface Completa do Painel de Auditoria Eleitoral" width="95%">
</p>

---

## 💡 Principais Insights & Achados da Auditoria (Findings)

* **Concentração em Seções Residuais ($\le 10$ Votos):** Identificou-se uma grande assimetria na distribuição acumulada de votos em seções onde a votação de um dos candidatos permaneceu na faixa residual ($0$ a $10$ votos), acumulando expressiva margem para o adversário em zonas específicas.
* **Volume de Habilitações sem Biometria:** A auditoria aponta que **~8,63 milhões de votos foram liberados manualmente** mediante digitação do ano de nascimento pelo mesário. Este volume supera em quase duas vezes a margem final de diferença nacional (~4,43 milhões de votos).
* **Anomalia Estatística de Seções Unânimes ($0$ Votos):** Sob a **Lei dos Grandes Números**, a probabilidade de um candidato de expressão relevante no 1º turno obter $0$ votos em uma seção urbana densa no 2º turno sob independência aleatória é estatisticamente desprezível ($p < 0,00001$), configurando *outliers* com desvio-padrão $Z\text{-Score} > 4.0$.
* **Aderência à Lei de Newcomb-Benford:** O teste de Chi-Quadrado ($\chi^2$) aplicado ao **2º dígito da votação** por seção permite rastrear discrepâncias estocásticas nas seções retidas ou transmitidas com atraso.

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

## 🛠️ Tecnologias Utilizadas

- **Linguagem:** Python 3.10+
- **Dashboard Web:** Streamlit
- **Manipulação de Dados:** Pandas, NumPy
- **Estatística Inferencial:** SciPy (`scipy.stats`)
- **Visualização de Dados:** Plotly Express & Graph Objects
- **Formato de Dados:** Apache Parquet (`pyarrow` / `fastparquet`)

---

## 💻 Instalação e Execução Local

1. **Clone este repositório:**
   ```bash
   git clone [https://github.com/derikpetiz/eleicoes-tse-app.git](https://github.com/derikpetiz/eleicoes-tse-app.git)
   cd eleicoes-tse-app

# CONTEXTO DE CONTINUIDADE DO PROJETO DETECTOR UEBA
**Data de Atualização:** 01 de Novembro de 2026  
**Status do Projeto:** Pipeline operacional, auditado, com correções metodológicas críticas implementadas e testadas.

---

## 1. Identificação do Projeto e Metadados Acadêmicos
* **Autor:** Allysson de Oliveira Noronha
* **Orientador:** Prof. Dr. Robson de Oliveira Albuquerque (UnB/FT/ENE)
* **Programa:** Mestrado Profissional em Engenharia Elétrica - Universidade de Brasília (UnB)
* **Título da Dissertação:** *Proposta de detecção de Anomalias em Logs de Autenticação com Limiares Dinâmicos e Abordagem UEBA*
* **Prazo Final de Defesa:** Fevereiro de 2027 (~3 meses restantes)
* **Repositório do Texto em LaTeX (unbtex):**
  `C:\Users\allysson.aon\Desktop\TEX\UnBTeX__A_class_for_bachelor__master__and_doctoral_thesis_at_the_University_of_Brasilia__UnB___1_\unbtex-example`
  *(Capítulos 1 a 4 já densamente redigidos com dados confirmatórios de 10 sementes).*
* **Workspace do Código:** `C:\Users\allysson.aon\Det_Anomalias_UEBA`
* **Interpretador Python:** `C:\Users\allysson.aon\Det_Anomalias_UEBA\.venv310\Scripts\python.exe` (Python 3.10.11)
* **Arquivo Consolidado de Códigos:** `C:\Users\allysson.aon\Desktop\codigos_versao_01_11_2026.txt`

---

## 2. Visão do Produto e Problema de Negócio (Mestrado Profissional)
* **Objetivo Real:** Fornecer uma ferramenta de auditoria comportamental (UEBA) para uso **FORA do SOC e FORA da TI**, voltada para analistas de **Inteligência, Contrainteligência e Segurança Pública**.
* **Foco do Alerta:** Identificar desvios sutis de conduta e potenciais vazamentos de dados por ameaças internas (ex.: analista acessando servidores inéditos ou pastas restritas fora do horário habitual dias antes de uma operação policial), e não apenas ataques cibernéticos ruidosos (exploits/bruteforce).
* **Premissa Chave:** Explicabilidade dos desvios com base no histórico de entidades do Active Directory (Quem <-> Quando <-> Onde <-> Quanto).

---

## 3. Principais Descobertas e Correções Realizadas (Sessão 31/10 - 01/11/2026)

### A. Correções Operacionais de Ambiente
1. **Remoção de UTF-8 BOM**: `core/trainer.py` possuía caractere BOM (`\xef\xbb\xbf`) que quebrava o parser AST via `open(..., encoding='utf-8')`. O arquivo foi limpo para UTF-8 puro.
2. **Compatibilidade UTF-8 no Windows (cp1252)**: Adicionado `sys.stdout.reconfigure(encoding='utf-8', errors='replace')` no `scripts/detectar.py` para evitar quebras com prints e emojis no console Windows.
3. **Instalação do `pytest`**: Instalada a versão 9.2.0 no ambiente `.venv310`. A suíte de testes de causalidade (`tests/test_causality.py`) foi executada e todos os 7 testes passaram com 100% de sucesso.

### B. Correção Metodológica Crítica: Desalinhamento de Rótulos no Teste
* **O Bug Encontrado**: No `core/trainer.py`, `_build_sequences` iterava os usuários na ordem de aparição no DataFrame (`groupby('UserID', sort=False)`), gerando escores de anomalia nessa exata sequência. No entanto, no `scripts/evaluate_performance.py`, os rótulos de gabarito `y_test` eram reconstruídos com `groupby('UserID')` (que no pandas tem default `sort=True`, ordem alfabética).
* **Impacto**: Os escores do Usuário A estavam sendo avaliados contra o gabarito do Usuário B, corrompendo todas as métricas de teste (PR-AUC, ROC-AUC, F1, TP, FP).
* **A Correção Aplicada**:
  * Em `scripts/detectar.py` (método `_evaluate_subset`): passa diretamente `y_true=y_test`, `users=users` e `y_pred=y_pred` calculados na memória do `Trainer`.
  * Em `scripts/evaluate_performance.py`: aceita esses arrays diretamente; e se executado de forma avulsa a partir de arquivos em disco, usa `y_true.npy` ou faz `groupby('UserID', sort=False)`.

### C. Calibração Operacional de Limiares (F2-Score vs Taxa de Alertas)
* **O Problema**: A busca por limiares ótimos via F2 puro procurava percentis a partir de `0.0%`. Como F2 prioriza Recall sobre Precision (fator 4x), o otimizador caía em percentis de 0% ou 12%, disparando alertas em 99% das janelas (inundando o analista e quebrando a métrica de *Lift*).
* **A Correção**: Configurado `threshold_search.percentile_min: 80.0` no YAML de execução, forçando o limiar a operar na faixa de percentis condizente com UEBA (rejeitando ruído benigno).
* **Resultado Comprovado no Smoke Test**:
  * Janelas anômalas no teste: 386 de 2.662 (14,50%).
  * Alertas emitidos: 394 (14,8% do tráfego).
  * Verdadeiros Negativos: 1.997 (ruído filtrado com sucesso).
  * Precisão: 0.2919 (o dobro do acaso estatístico, *Lift* positivo de 1.52x).

### D. Validação das Novas Features do `FeatureProcessor`
* O `FeatureProcessor` congelou com sucesso **39 dimensões**.
* As 6 novas features solicitadas foram validadas e estão presentes no `trainer_meta.pkl`:
  * `fail_count_5m`: True
  * `fail_then_success`: True
  * `user_host_new`: True
  * `kerberos_logon_without_tgt`: True
  * `is_ntlm`: True
  * `unique_hosts`: True

---

## 4. Estrutura dos Arquivos Ativos do Detector
O arquivo consolidado com o código-fonte de todos os scripts ativos está disponível em `C:\Users\allysson.aon\Desktop\codigos_versao_01_11_2026.txt`:
1. `core/trainer.py` - Protocolo de treino, anti-leakage e calibração de limiares.
2. `core/temporal_builder.py` - Segmentação de sessões e timeout.
3. `core/utils/config_validator.py` - Validação de integridade dos YAMLs.
4. `core/data/dataset_adapter.py` - Ingestão, normalização do AD e split cronológico/aleatório.
5. `core/data/dataset_inspector.py` - Auditoria estatística das partições.
6. `core/features/feature_registry.py` - Registro dos grupos de features.
7. `core/features/feature_processor.py` - Engenharia de atributos comportamentais UEBA.
8. `core/models/base_model.py` - Interface abstrata dos modelos.
9. `core/models/model_factory.py` - Fábrica dinâmica de algoritmos.
10. `core/models/mlp_single.py` - Detector reconstrutivo neural (MLP Autoencoder).
11. `core/models/random_forest_supervised.py` - Comitê supervisionado (RF).
12. `scripts/detectar.py` - Executor principal da maratona experimental.
13. `scripts/evaluate_performance.py` - Avaliação de métricas corrigida (sem desalinhamento).
14. `scripts/inferir_usual.py` - Inferência de alta fidelidade para datasets novos.
15. `scripts/batch_inferir.py` - Inferência em lote e análise de sensibilidade.

---

## 5. Próximos Passos Mapeados (Roadmap até Fevereiro/2027)

1. **Gestão do Cache (`scripts/detectar.py`)**:
   * *Atenção*: O cache de N1 e N2 usa caminhos baseados em dataset/seed/split_strategy. Ao testar alterações de features, **limpar manualmente** a pasta `data/GLOBAL_CACHE_*` para garantir que o cache não reutilize matrizes antigas.
2. **Adequação Textual da Dissertação (Capítulos 3 e 4 em LaTeX)**:
   * Declarar explicitamente no Capítulo 3 a causalidade das features como **"classificação no instante t do evento recém-ocorrido"** (eliminando o questionamento da avaliação).
   * Atualizar as tabelas do Capítulo 4 com os resultados do pipeline corrigido (quando o Allysson solicitar).
3. **Cenários do EvidenceForge (Opcional/Complementar)**:
   * Mapeado como gerador de cenários realistas de Active Directory (Kerberos/Sysmon) pela Cisco Talos. Pode ser usado para estudos de caso de ameaça interna ou citado na fundamentação teórica.
4. **Git / Versionamento**:
   * A máquina corporativa não possui `git.exe` no PATH. Foi disponibilizado o consolidado em `.txt` como snapshot de segurança.

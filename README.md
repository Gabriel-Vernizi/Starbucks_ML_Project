# Starbucks Offer Predictor

![Status](https://img.shields.io/badge/Status-Work%20in%20Progress-orange)
![Python](https://img.shields.io/badge/Python-3.14%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-0.24%2B-yellow)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-yellow)

Bem-vindo ao repositório do projeto **Starbucks Offer Predictor**. Este projeto tem como objetivo construir um modelo de Machine Learning capaz de prever se um cliente da Starbucks completará ou não uma oferta recebida pelo aplicativo, isolando os casos em que a visualização da oferta de fato influenciou a compra.

## 🛠 Tratamento dos Dados

Para garantir a qualidade dos dados alimentados ao modelo, realizamos uma extensa análise e limpeza inicial utilizando os datasets originais (`portfolio`, `profile` e `transcript`):

* **Valores Ausentes e Anomalias no Perfil (`profile`)**: 
    * Notamos que ~12.8% dos dados não possuíam gênero (`gender`) e renda (`income`).
    * Identificamos um comportamento anômalo na idade: clientes com a idade `118` correspondiam perfeitamente aos perfis com os dados faltantes, funcionando como um *placeholder* (preenchimento padrão do sistema para perfis não completados).
* **Ajuste no Portfólio de Ofertas (`portfolio`)**:
    * A variável de duração (`duration`) estava em dias e foi convertida para horas, padronizando a escala de tempo com os logs de eventos.
    * Realizamos a junção das informações para formar labels descritivos das ofertas (`offer_exp`), como por exemplo `bogo_5_for_5_in_7days`.
* **Feature Engineering**:
    * Criação de variáveis focadas no perfil histórico financeiro e engajamento do usuário (RFM), tais como `total_spent_so_far` (total gasto até o momento), `avg_ticket_so_far` (ticket médio) e `is_high_spender`.
* **Construção da Variável Alvo (`target`)**:
    * O critério de sucesso do negócio foi definido de forma estrita e realista: o modelo considera como sucesso (`1`) apenas quando o cliente **recebeu**, **visualizou** e depois **completou** a oferta dentro do prazo. 
    * Clientes que completaram uma promoção sem terem visto (comportamento orgânico não influenciado pelo app) foram devidamente categorizados como não-sucesso (`0`) sob a ótica de conversão de marketing.

## ⚙️ Preparação do Pipeline

Para garantir reprodutibilidade, facilitar o *deploy* futuro e evitar vazamento de dados (*data leakage*), o pré-processamento foi centralizado em um **Pipeline** integrado do Scikit-Learn, distribuído através de um `ColumnTransformer`:

* **Pipeline Numérico** (`age`, `income`, `reward`, `difficulty`, `duration`, `membership_duration`, `total_spent_so_far`, etc.):
    * `SimpleImputer(strategy='median')`: Garantia de tratamento de possíveis valores nulos restantes pela mediana.
    * `StandardScaler()`: Padronização das distribuições para otimizar convergência e cálculo de distâncias.
* **Pipeline Categórico** (`gender`, `offer_type`):
    * `SimpleImputer(strategy='most_frequent')`: Tratamento de nulos utilizando a moda (valor mais frequente).
    * `OneHotEncoder(handle_unknown='ignore')`: Binarização das categorias (Dummificação).
* **Processamento de Canais (`channels`)**:
    * Implementação de um Transformer customizado (`ChannelsEncoder`) para aplicar encoding nas listas contendo os canais de distribuição (ex: `web`, `mobile`, `social`).
    * Remoção preventiva do canal `email`, uma vez que ele estava presente em 100% das ofertas e não gerava quebra de variância.

---

## ⚠️ Adendo: Desafios e Próximos Passos (WIP)

Atualmente, o projeto está em desenvolvimento e nos deparamos com o desafio de que **o modelo preditivo está performando de forma essencialmente aleatória**. A rede ainda não conseguiu encontrar um padrão forte que separe as classes de quem converte e quem não converte a oferta.

Para solucionar isso, estou focado nas seguintes frentes e alternativas de pesquisa:

1.  **Refinamento do Feature Engineering**: O comportamento do consumidor é altamente temporal. Planejo extrair e adicionar novas *features* cruzadas (Ex: *Taxa histórica de sucesso deste usuário em ofertas do tipo BOGO*, *Tempo decorrido desde a última compra*, e *Frequência de interações nos últimos 10 dias*).
2.  **Reformulação da Variável Alvo (Abordagem de Dois Estágios)**: Talvez manter transações orgânicas na mesma base de dados prejudique o aprendizado do padrão promocional. Avaliaremos segmentar a modelagem e/ou remover da base as transações realizadas sem nenhuma interação do usuário com o app.
3.  **Testes com Algoritmos mais Complexos**: Implementar algoritmos robustos e com melhor lidar com não-linearidade, como **AdaBoost** e **LightGBM**, em substituição ou combinação com o atual Random Forest.
4.  **Refatoração do Pipeline Customizado**: Revisitar o módulo `class_pipeline.py`. Identifiquei classes de imputação customizadas (como o `CustomImputer_for_Gender_Age_Income`) que possuem instâncias hardcoded (ex: o parâmetro numérico de `spending_threshold`). Isso será unificado em um transformador dinâmico para evitar vazamento de dados (*data leakage*) do dataset de treino para o de teste e remover redundâncias de código.
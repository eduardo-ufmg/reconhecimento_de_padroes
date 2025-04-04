# Random KNN feature selection - a fast and stable alternative to Random Forests

## O Problema

É comum, principalmente em problemas de bioinformática, que haja muito mais variáveis do que amostras. Isso é ruim, dado que a condição $ p \gt n $, para *p* variáveis e *n* amostras, leva a
- alto custo computacional &rarr; mais memória ocupada; dados irrelevantes são avaliados
- overfitting &rarr; o modelo pode aprender relações que só existem naquele conjunto de dados

## Seleção de Variáveis

É possível selecionar um conjunto considereravelmente menor de variáveis que contenha somente aquelas que, conforme os critérios empregrados, sejam fundamentalmente relevantes para a classificação.

### Seleção Interna de Variáveis

A seleção das variáveis é parte intríseca do treino do modelo.

#### Árvore de Decisão

- Para cada variável, avalia o ganho de informação em diferentes partições
- Partições com maior ganho ficam mais próximas da raiz

**Principais problemas**
- Pequenas mudanças nos dados podem levar a estruturas completamente diferentes - instabilidade
- Erros são fortemente cumulativos

#### Floresta Aleatória

Agrega multiplas árvores de decisão, com algumas modificações:
- Cada árvore é construída sobre um subconjunto dos dados
- Na construção de cada nó, um subconjunto das variáveis é considerado
- Cada árvore vota conforme seu resultado e a classificação é feita a partir destes

**Principais problemas**
- Mitiga, mas não resolve a instabilidade
- Aumenta o custo computacional

### Filtragem de Variáveis

Analisa, estatisticamente, a relação entre cada variável e as classes do problema. Mantém aquelas que são fortes indicadores de classe.

**Exemplo**: Retirar variáveis cuja variança seja baixa no conjunto completo de dados, visto que estas tendem a não ser relevantes para a classificação.

**Principais problemas**:
- Por avaliar cada variável individualmente, pode remover aquelas que participam de relações úteis para a classificação
- Pelo mesmo motivo, pode manter variáveis redundantes

### Métodos Encapsuladores

Encapsula um modelo em um problema de optimização.
1. Seleciona um subconjunto das variáveis
2. Treina o modelo sobre este subconjunto
3. Avalia o modelo conforme uma métrica apropriada
4. Seleciona o modelo com melhor performance

**Principal problema**: o mais caro computacionalmente

## RKNN-FS

- Para um conjunto de dados com *p* variáveis, *r* classificadores *kNN* são treinados em subconjuntos aleatórios de $m \approx \sqrt{p}$ variáveis
- Cada classificador é treinado sobre todas as amostras. *Bootstraping* não é necessário, já que, por não serem hierárquicos, os classificadores são naturalmente estáveis
- A classificação é dada pelo voto majoritário entre os classificadores

### *Feature Support*

- Cada variável *f* é atribuída a *M* classificadores, que formam o conjunto *C(f)*. Cada classificador recebe o conjunto *F(c)* de variáveis
- Quando um classificador acerta sua predição, as variáveis usadas por ele recebem uma pontuação positiva
- Quando erra, recebem uma pontuação negativa
- A pontuação, por rodada, de cada variável, é a média das pontuações atribuídas por cada classificador $c \in C(f)$
- Variáveis mais relevantes para a classificação têm pontuação maior

### Seleção de Variáveis

#### Remoção geométrica
1. A cada iteração, a metade menos relevante das variáveis é descartada
2. Após todas as iterações, a quantidade de variáveis que resulta na maior precisão é encontrada
3. As variáveis presentes na iteração anterior a esta são passadas para a próxima etapa

#### Remoção linear
1. A cada iteração, um número fixo (*1*, na aplicação citada) de variáveis é removido
2. Após todas as iterações, o conjunto de variáveis mais relevantes é selecionada para o modelo

## Implementação



## Resultados

# Grupo
- Eduardo Henrique Basilio de Carvalho
- João Vitor Braga da Silva Alves

# Etapas
1. Analisar e reproduzir o artigo [Width optimization of RBF kernels for binary classification of support vector machines: A density estimation-based approach](https://www.sciencedirect.com/science/article/pii/S0167865519302156)
2. Formular e implementar uma solução alternativa
3. Comparar os resultados obtidos com a solução alternativa e as obtidas pelo método do artigo
4. Elaborar um relatório com os resultados obtidos e as conclusões

# Cronograma
| Data       | Atividade
| -----------|----------
| 02/Junho   | Análise do artigo
| 09/Junho   | Implementação da solução alternativa
| 15/Junho   | Comparação dos resultados e elaboração do relatório

# Estratégia avaliada
Buscar `h` para o qual a função `scipy.spatial.ConvexHull` forneça o polígono mais "próximo" do triangulo dado por [`(0, 0)`, `(0, 1)`, `(1, 1)`] para a classe 1 e [`(0, 0)`, `(1, 0)`, `(1, 1)`] para a classe 0.

# Desafio inicial
Formular, matemáticamente, o que é um polígono "próximo" de cada triângulo.

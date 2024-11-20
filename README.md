Classificação de Espécies de Flores - Dataset Iris
Este projeto utiliza aprendizado de máquina para classificar espécies de flores do dataset Iris com base em suas características morfológicas (comprimento e largura de sépalas e pétalas).

Objetivo
Treinar modelos de aprendizado de máquina para:

Identificar a espécie de uma flor.
Comparar o desempenho de diferentes algoritmos.
Estrutura do Projeto
main.py: Código principal com implementação dos modelos.
iris.data: Dataset Iris em formato CSV.
README.md: Documentação do projeto.
Modelos Implementados
Support Vector Machine (SVM)
Random Forest
K-Nearest Neighbors (KNN)
Requisitos
Certifique-se de que você tenha os seguintes pacotes instalados: (pip install pandas scikit-learn matplotlib seaborn)
Como Executar
Clone este repositório:
bash
Copiar código
git clone https://github.com/diegocbrito/iris-flower-classification1.git
Navegue até a pasta do projeto:
bash
Copiar código
cd seu-repositorio
Execute o script:
bash
Copiar código
python main.py
O script carregará o dataset, realizará o tratamento de dados, treinará os modelos e exibirá os resultados no terminal.

Resultados Obtidos
SVM:
Validação cruzada: 94.29%
Acurácia: 97.78%
Random Forest:
Validação cruzada: 94.29%
Acurácia: 100%
KNN:
Validação cruzada: 91.43%
Acurácia: 100%
Sobre o Dataset
O dataset Iris contém 150 amostras, divididas em 3 espécies:

Iris-setosa
Iris-versicolor
Iris-virginica
Cada amostra possui 4 atributos:

Comprimento da sépala
Largura da sépala
Comprimento da pétala
Largura da pétala

Licença
Este projeto foi desenvolvido para fins educacionais.

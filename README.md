
# Mangrove Recognizer: Classificação de Macro-habitats de Manguezais

Este projeto é um sistema completo de Machine Learning desenvolvido como projeto final da disciplina de Processamento Digital de Imagens. O objetivo é automatizar a classificação de diferentes tipos de macro-habitats de manguezais a partir de imagens de satélite, utilizando técnicas de PDI e modelos de classificação.

O sistema foi desenvolvido em parceria com o **Laboratório de Peixes e Conservação Marinha (LAPEC - UEPB)** para auxiliar em suas pesquisas, reduzindo o trabalho manual de análise de imagens.

---

## Resultados do Modelo

O modelo principal, baseado em **Random Forest**, alcançou uma **acurácia geral de 93%** no conjunto de dados de teste. A performance detalhada para cada classe de macro-habitat é a seguinte:

| Macro-habitat                       | Precisão | Recall | F1-Score | Amostras (Suporte) |
| ----------------------------------- | :------: | :----: | :------: | :----------------: |
| **I — Manguezal denso**             |   0.94   |  0.97  |   0.96   |         66         |
| **II — Manguezal ralo**             |   0.95   |  0.86  |   0.90   |         21         |
| **III — Ambiente recifal**          |   1.00   |  1.00  |   1.00   |         8          |
| **IV — Bancos de algas e esponjas** |   0.50   |  0.33  |   0.40   |         3          |
| **V — Substrato arenoso com rochas**|   0.88   |  0.93  |   0.90   |         15         |

**Análise:** O modelo demonstra excelente performance para a maioria das classes, especialmente aquelas com mais amostras (`I`, `II`, `V`). A performance na `Classe IV` é baixa, indicando que o modelo tem dificuldade em identificá-la, muito provavelmente devido ao número extremamente baixo de exemplos disponíveis para treinamento.

---

## Como Utilizar (Guia para o LAPEC)

Para classificar uma **nova localidade** a partir de um par de coordenadas (latitude/longitude), siga estes passos:

### Passo 1: Obter a Imagem de Satélite

O objetivo é obter uma imagem de satélite padronizada da sua coordenada.

1.  **Gerar um KML:** Embora o script original usasse um programa para gerar KMLs, você pode criar um manualmente de forma simples. Abra um editor de texto (como o Bloco de Notas), cole o seguinte template e salve o arquivo com a extensão `.kml` (ex: `ponto.kml`).

    ```xml
    <?xml version="1.0" encoding="UTF-8"?>
    <kml xmlns="http://www.opengis.net/kml/2.2">
      <Placemark>
        <name>Ponto de Interesse</name>
        <LookAt>
          <longitude>-44.30</longitude> <!-- SUBSTITUA PELA SUA LONGITUDE -->
          <latitude>-2.50</latitude>   <!-- SUBSTITUA PELA SUA LATITUDE -->
          <altitude>0</altitude>
          <heading>0</heading>
          <tilt>0</tilt>
          <range>500</range> <!-- Altitude da câmera em metros -->
        </LookAt>
        <Point>
          <coordinates>-44.30,-2.50,0</coordinates> <!-- SUBSTITUA LON,LAT -->
        </Point>
      </Placemark>
    </kml>
    ```
    **Importante:** Substitua os valores de latitude e longitude nos locais indicados.

2.  **Abrir no Google Earth Pro:** Dê um duplo clique no arquivo `.kml` que você criou. O Google Earth Pro irá "voar" para a sua coordenada com os parâmetros de câmera padronizados (visão de cima, a 500m de altitude).

3.  **Tirar um Print (Screenshot):** Capture a imagem da tela, garantindo que a área de interesse esteja bem visível. Salve esta imagem como `.png` ou `.jpg` em um local de fácil acesso.

### Passo 2: Executar o Classificador

Com a imagem salva, você pode usar o script de predição que criamos.

1.  **Abra o terminal** na pasta do projeto (`mangrove-recongizer`).
2.  Execute o comando abaixo, substituindo `/caminho/para/sua/imagem.png` pelo caminho real do arquivo que você acabou de salvar.

    ```bash
    python src/predict.py --image /caminho/para/sua/imagem.png
    ```

3.  O terminal irá processar a imagem e, em poucos segundos, exibirá o resultado da classificação e a confiança do modelo.

    ```
    --- RESULTADO DA CLASSIFICAÇÃO ---
    Macro-habitat previsto: I — Manguezal denso
    Confiança do modelo: 95.48%
    -----------------------------------
    ```

---

## Requisitos para Execução

Para executar o projeto, você precisa ter Python 3.9+ instalado, além das seguintes bibliotecas, que podem ser instaladas com o pip:

```bash
- `scikit-learn`, `scikit-image`
- `tensorflow`, `keras`
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `opencv-python`
- `pillow`
- `tqdm`
- `rasterio`
- `joblib`
```

---

## Agradecimentos

Este projeto não seria possível sem o apoio e a colaboração fundamental do **Laboratório de Peixes e Conservação Marinha (LAPEC - UEPB)**.

Um agradecimento especial a:

*   **Dra. Karol Borges**, por fornecer o dataset inicial, o conhecimento de domínio essencial sobre os macro-habitats e por propor o desafio que deu origem a este trabalho.
*   **Mestras Jéssica Sobral e Amanda Pereira** e **Profª. Drª. Tacyana**, pelo apoio contínuo, discussões produtivas e por abrirem as portas do laboratório para esta parceria.

A dedicação de vocês à pesquisa e conservação marinha é uma grande inspiração.


import os
import argparse
import logging
import cv2
import numpy as np
import pandas as pd
import joblib

# Importar as funções de extração de features que já criamos
from feature_extraction import calculate_ndvi, calculate_glcm, calculate_lbp, calculate_color_stats, calculate_morphology

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_image(image, patch_size=64):
    """Aplica o mesmo pré-processamento e extração de patch para uma nova imagem."""
    # 1. Filtro Gaussiano
    gaussian_blurred = cv2.GaussianBlur(image, (5, 5), 1.0)

    # 2. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b, g, r = cv2.split(gaussian_blurred)
    clahe_b = clahe.apply(b)
    clahe_g = clahe.apply(g)
    clahe_r = clahe.apply(r)
    clahe_image = cv2.merge([clahe_b, clahe_g, clahe_r])

    # 3. Extrair patch do centro
    h, w, _ = clahe_image.shape
    center_x, center_y = w // 2, h // 2
    half_patch = patch_size // 2
    patch = clahe_image[center_y - half_patch : center_y + half_patch, center_x - half_patch : center_x + half_patch]
    
    return patch

def extract_features(patch):
    """Extrai todas as features de um único patch."""
    # A imagem já está em BGR, vinda do pré-processamento
    ndvi = calculate_ndvi(patch)
    glcm = calculate_glcm(patch)
    lbp = calculate_lbp(patch)
    color = calculate_color_stats(patch)
    morph = calculate_morphology(patch)
    
    # Concatenar todas as features em um único vetor
    all_features = [ndvi] + glcm + lbp.tolist() + color + morph
    return np.array(all_features).reshape(1, -1)

def main():
    parser = argparse.ArgumentParser(description="Classifica uma única imagem de manguezal.")
    parser.add_argument('--image', type=str, required=True, help="Caminho para a nova imagem a ser classificada.")
    parser.add_argument('--model', type=str, default='models/random_forest.pkl', help="Caminho para o modelo treinado (.pkl).")
    parser.add_argument('--class-map', type=str, default='data/splits/class_map.csv', help="Caminho para o mapeamento de classes.")
    args = parser.parse_args()

    # Validar caminhos
    if not os.path.exists(args.image):
        logging.error(f"Arquivo de imagem não encontrado em: {args.image}")
        return
    if not os.path.exists(args.model):
        logging.error(f"Arquivo do modelo não encontrado em: {args.model}")
        return
    if not os.path.exists(args.class_map):
        logging.error(f"Arquivo de mapa de classes não encontrado em: {args.class_map}")
        return

    # Carregar modelo e mapa de classes
    logging.info(f"Carregando modelo de {args.model}")
    model = joblib.load(args.model)
    class_map_df = pd.read_csv(args.class_map)
    # Criar um dicionário para traduzir o resultado numérico para o nome da classe
    idx_to_class = dict(zip(class_map_df['class_int'], class_map_df['class_str']))

    # Processar a imagem
    logging.info(f"Processando imagem: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        logging.error("Não foi possível ler o arquivo de imagem.")
        return

    # 1. Pré-processamento
    patch = preprocess_image(image)
    if patch.shape[0] != 64 or patch.shape[1] != 64:
        logging.error(f"Falha ao extrair patch 64x64. A imagem é grande o suficiente?")
        return

    # 2. Extração de Features
    logging.info("Extraindo features da imagem...")
    features = extract_features(patch)

    # 3. Predição
    logging.info("Realizando predição...")
    prediction_idx = model.predict(features)[0]
    prediction_proba = model.predict_proba(features)[0]
    predicted_class_name = idx_to_class[prediction_idx]
    confidence = prediction_proba[prediction_idx] * 100

    # 4. Apresentar resultado
    print("\n--- RESULTADO DA CLASSIFICAÇÃO ---")
    print(f"Macro-habitat previsto: {predicted_class_name}")
    print(f"Confiança do modelo: {confidence:.2f}%")
    print("-----------------------------------")

if __name__ == '__main__':
    main()

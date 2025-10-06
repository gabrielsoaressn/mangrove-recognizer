
import os
import argparse
import logging
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import cv2
from tqdm import tqdm

# --- Configuração de Logging ---
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'feature_extraction.log')),
        logging.StreamHandler()
    ]
)

# --- Seed para Reprodutibilidade ---
SEED = 42
np.random.seed(SEED)

def calculate_ndvi(patch):
    """Calcula o NDVI (aproximado) usando canais RGB."""
    # G (canal 1), R (canal 2) - assumindo BGR do OpenCV
    g = patch[:, :, 1].astype(float)
    r = patch[:, :, 2].astype(float)
    # Evitar divisão por zero
    denominator = g + r
    ndvi = np.divide(g - r, denominator, out=np.zeros_like(g), where=denominator!=0)
    return np.mean(ndvi)

def calculate_glcm(patch):
    """Calcula features de textura GLCM."""
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray_patch, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return [contrast, energy, homogeneity, correlation]

def calculate_lbp(patch):
    """Calcula o histograma de features LBP."""
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_patch, n_points, radius, 'uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    # Normalizar o histograma
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def calculate_color_stats(patch):
    """Calcula estatísticas de cor."""
    b, g, r = cv2.split(patch)
    stats = []
    for chan in [b, g, r]:
        stats.extend([
            np.mean(chan),
            np.std(chan),
            np.percentile(chan, 25),
            np.percentile(chan, 75)
        ])
    return stats

def calculate_morphology(patch):
    """Calcula features morfológicas."""
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3,3), np.uint8)
    
    # Abertura e Fechamento
    opening = cv2.morphologyEx(gray_patch, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(gray_patch, cv2.MORPH_CLOSE, kernel)
    
    # Contagem de componentes conectados
    _, thresh = cv2.threshold(gray_patch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(thresh)
    
    return [np.mean(opening), np.mean(closing), num_labels]

def extract_features_from_split(X_data):
    """Extrai todas as features para um conjunto de patches."""
    features_list = []
    for patch in tqdm(X_data, desc="Extraindo Features"):
        # As imagens do .npz estão em RGB, converter para BGR para o OpenCV
        patch_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)

        ndvi = calculate_ndvi(patch_bgr)
        glcm = calculate_glcm(patch_bgr)
        lbp = calculate_lbp(patch_bgr)
        color = calculate_color_stats(patch_bgr)
        morph = calculate_morphology(patch_bgr)
        
        # Concatenar todas as features
        all_features = [ndvi] + glcm + lbp.tolist() + color + morph
        features_list.append(all_features)
    return pd.DataFrame(features_list)

def main():
    parser = argparse.ArgumentParser(description="Extrai features dos patches de imagem.")
    parser.add_argument('--input', type=str, required=True, help="Diretório com os arquivos .npz dos splits.")
    parser.add_argument('--output', type=str, required=True, help="Diretório para salvar os arquivos .csv de features.")
    args = parser.parse_args()

    logging.info("Iniciando extração de features...")
    os.makedirs(args.output, exist_ok=True)

    for split_name in ['train', 'val', 'test']:
        logging.info(f"Processando split: {split_name}")
        split_path = os.path.join(args.input, f'{split_name}.npz')

        if not os.path.exists(split_path):
            logging.error(f"Arquivo de split não encontrado: {split_path}. Pulando.")
            continue

        try:
            data = np.load(split_path)
            X = data['X']
            y = data['y']

            df_features = extract_features_from_split(X)
            df_features['label'] = y

            output_path = os.path.join(args.output, f'features_{split_name}.csv')
            df_features.to_csv(output_path, index=False)
            logging.info(f"Features para o split '{split_name}' salvas em {output_path}")

        except Exception as e:
            logging.error(f"Falha ao processar o split {split_name}. Erro: {e}", exc_info=True)

    logging.info("Extração de features concluída.")

if __name__ == '__main__':
    main()

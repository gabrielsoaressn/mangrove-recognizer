import os
import argparse
import logging
import pandas as pd
import numpy as np
import cv2
import glob
from skimage.transform import rotate
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Configuração de Logging ---
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'data_preparation.log')),
        logging.StreamHandler()
    ]
)

# --- Seed para Reprodutibilidade ---
SEED = 42
np.random.seed(SEED)

def augment_patch(patch):
    """Aplica data augmentation a um patch."""
    augmented_patches = [patch]
    # Rotações
    augmented_patches.append(rotate(patch, angle=90, resize=False, preserve_range=True).astype(patch.dtype))
    augmented_patches.append(rotate(patch, angle=180, resize=False, preserve_range=True).astype(patch.dtype))
    augmented_patches.append(rotate(patch, angle=270, resize=False, preserve_range=True).astype(patch.dtype))
    # Flips
    augmented_patches.append(np.fliplr(patch))
    augmented_patches.append(np.flipud(patch))
    return augmented_patches

def main():
    parser = argparse.ArgumentParser(description="Prepara os dados para o treinamento do modelo de classificação de mangue.")
    parser.add_argument('--csv', type=str, required=True, help="Caminho para o arquivo CSV com coordenadas e classes.")
    parser.add_argument('--images', type=str, required=True, help="Diretório com as imagens.")
    parser.add_argument('--patch-size', type=int, default=64, help="Tamanho do patch a ser extraído (lado).")
    parser.add_argument('--output', type=str, default='data/splits/', help="Diretório para salvar os splits de dados.")
    args = parser.parse_args()

    logging.info("Iniciando a preparação dos dados com a lógica de correspondência 1-para-1...")
    os.makedirs(args.output, exist_ok=True)

    # 1. Carregar CSV
    try:
        coords_df = pd.read_csv(args.csv)
        logging.info(f"Carregado {len(coords_df)} coordenadas de {args.csv}")
    except FileNotFoundError:
        logging.error(f"Arquivo CSV não encontrado em {args.csv}")
        return

    # 2. Listar e ordenar imagens
    image_paths = sorted(glob.glob(os.path.join(args.images, '**/*.png'), recursive=True))
    if not image_paths:
        logging.error(f"Nenhuma imagem .png encontrada em {args.images}")
        return
    logging.info(f"Encontrado {len(image_paths)} imagens.")

    # 3. Verificar correspondência
    if len(coords_df) != len(image_paths):
        logging.error(f"O número de coordenadas ({len(coords_df)}) não corresponde ao número de imagens ({len(image_paths)}). Abortando.")
        logging.error("Certifique-se de que cada linha no CSV corresponde a um arquivo de imagem.")
        return

    # Mapeamento de classes para inteiros
    class_map = {label: i for i, label in enumerate(coords_df['macrohabitat'].unique())}
    logging.info(f"Mapeamento de classes: {class_map}")

    all_patches = []
    all_labels = []
    
    iterator = tqdm(zip(image_paths, coords_df.itertuples()), total=len(image_paths), desc="Processando Imagens")
    for image_path, point_data in iterator:
        try:
            # Carregar imagem com OpenCV (lê em BGR)
            image = cv2.imread(image_path)
            if image is None:
                logging.warning(f"Não foi possível ler a imagem: {image_path}. Pulando.")
                continue
            
            # Converter para RGB para consistência
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Extrair patch do centro da imagem
            h, w, _ = image_rgb.shape
            center_x, center_y = w // 2, h // 2
            half_patch = args.patch_size // 2

            patch = image_rgb[center_y - half_patch : center_y + half_patch, center_x - half_patch : center_x + half_patch]

            if patch.shape != (args.patch_size, args.patch_size, 3):
                logging.warning(f"Patch extraído de {os.path.basename(image_path)} tem tamanho incorreto {patch.shape}. Verifique as dimensões da imagem. Pulando.")
                continue

            # Obter label
            label_str = point_data.macrohabitat
            label_int = class_map[label_str]

            # Aplicar data augmentation
            augmented_patches = augment_patch(patch)
            all_patches.extend(augmented_patches)
            all_labels.extend([label_int] * len(augmented_patches))

        except Exception as e:
            logging.error(f"Falha ao processar a imagem {image_path}. Erro: {e}", exc_info=True)

    if not all_patches:
        logging.error("Nenhum patch foi extraído. Verifique os logs para possíveis erros.")
        return

    X = np.array(all_patches)
    y = np.array(all_labels)
    logging.info(f"Total de patches extraídos e aumentados: {len(X)}")

    # Dividir dataset (70% treino, 15% validação, 15% teste)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y
    )
    val_size_in_remaining = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_in_remaining, random_state=SEED, stratify=y_train_val
    )

    logging.info(f"Tamanho do conjunto de treino: {len(X_train)}")
    logging.info(f"Tamanho do conjunto de validação: {len(X_val)}")
    logging.info(f"Tamanho do conjunto de teste: {len(X_test)}")

    # Salvar splits
    np.savez_compressed(os.path.join(args.output, 'train.npz'), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(args.output, 'val.npz'), X=X_val, y=y_val)
    np.savez_compressed(os.path.join(args.output, 'test.npz'), X=X_test, y=y_test)
    logging.info(f"Splits de dados salvos em {args.output}")

    # Salvar o mapeamento de classes
    class_map_df = pd.DataFrame(list(class_map.items()), columns=['class_str', 'class_int'])
    class_map_df.to_csv(os.path.join(args.output, 'class_map.csv'), index=False)
    logging.info(f"Mapeamento de classes salvo em {os.path.join(args.output, 'class_map.csv')}")

if __name__ == '__main__':
    main()
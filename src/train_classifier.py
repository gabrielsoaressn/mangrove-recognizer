
import os
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from tensorflow.keras.utils import to_categorical

# --- Configuração de Logging ---
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'training.log')),
        logging.StreamHandler()
    ]
)

# --- Seed para Reprodutibilidade ---
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- Callbacks Customizados (Matriz de Confusão) ---
class ConfusionMatrixCallback(Callback):
    def __init__(self, validation_data, output_dir, class_names):
        super().__init__()
        self.validation_data = validation_data
        self.output_dir = output_dir
        self.class_names = class_names
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val_cat = self.validation_data
        y_pred = self.model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_val_cat, axis=1)

        cm = confusion_matrix(y_true, y_pred_classes)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
        ax.set_title(f'Matriz de Confusão - Época {epoch + 1}')
        ax.set_ylabel('Verdadeiro')
        ax.set_xlabel('Previsto')
        plt.savefig(os.path.join(self.output_dir, f'cm_epoch_{epoch + 1}.png'))
        plt.close(fig)

def train_random_forest(features_dir, models_dir):
    logging.info("--- Treinando Modelo Random Forest ---")
    
    # Carregar dados
    df_train = pd.read_csv(os.path.join(features_dir, 'features_train.csv'))
    X_train = df_train.drop('label', axis=1)
    y_train = df_train['label']

    # Instanciar e treinar modelo
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=SEED, n_jobs=-1)
    logging.info("Iniciando o treinamento do Random Forest...")
    rf.fit(X_train, y_train)
    logging.info("Treinamento do Random Forest concluído.")

    # Salvar modelo
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'random_forest.pkl')
    joblib.dump(rf, model_path)
    logging.info(f"Modelo Random Forest salvo em {model_path}")

def train_cnn(splits_dir, models_dir, epochs):
    logging.info("--- Treinando Modelo CNN ---")

    # Carregar dados
    train_data = np.load(os.path.join(splits_dir, 'train.npz'))
    val_data = np.load(os.path.join(splits_dir, 'val.npz'))
    X_train, y_train = train_data['X'], train_data['y']
    X_val, y_val = val_data['X'], val_data['y']

    # Normalizar imagens para [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0

    # One-hot encode labels
    num_classes = len(np.unique(y_train))
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes)

    input_shape = X_train.shape[1:]

    # Construir modelo CNN
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary(print_fn=logging.info)

    # Callbacks
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_path = os.path.join(models_dir, 'cnn_model.h5')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
    model_checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss')
    
    # Carregar nomes das classes para o callback da matriz de confusão
    class_map_path = os.path.join(splits_dir, 'class_map.csv')
    class_names = ['Unknown'] * num_classes
    if os.path.exists(class_map_path):
        class_map_df = pd.read_csv(class_map_path).sort_values('class_int')
        class_names = class_map_df['class_str'].tolist()

    cm_callback = ConfusionMatrixCallback(validation_data=(X_val, y_val_cat), output_dir='results/cnn_training_cms', class_names=class_names)

    logging.info("Iniciando o treinamento da CNN...")
    history = model.fit(X_train, y_train_cat,
                        epochs=epochs,
                        validation_data=(X_val, y_val_cat),
                        batch_size=32,
                        callbacks=[early_stopping, reduce_lr, model_checkpoint, cm_callback])

    logging.info("Treinamento da CNN concluído.")

    # Gerar curvas de aprendizado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history.history['accuracy'], label='Treino Acc')
    ax1.plot(history.history['val_accuracy'], label='Val Acc')
    ax1.set_title('Acurácia do Modelo')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Acurácia')
    ax1.legend()

    ax2.plot(history.history['loss'], label='Treino Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Loss do Modelo')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plot_path = os.path.join('results', 'cnn_learning_curves.png')
    os.makedirs('results', exist_ok=True)
    plt.savefig(plot_path)
    plt.close(fig)
    logging.info(f"Curvas de aprendizado salvas em {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Treina os modelos de classificação.")
    parser.add_argument('--model', type=str, required=True, choices=['rf', 'cnn'], help="Modelo a ser treinado: 'rf' (Random Forest) ou 'cnn'.")
    parser.add_argument('--features', type=str, help="Diretório com as features para o Random Forest.")
    parser.add_argument('--splits', type=str, help="Diretório com os splits de imagem para a CNN.")
    parser.add_argument('--epochs', type=int, default=50, help="Número de épocas para treinar a CNN.")
    parser.add_argument('--models-dir', type=str, default='models', help="Diretório para salvar os modelos treinados.")
    args = parser.parse_args()

    if args.model == 'rf':
        if not args.features:
            parser.error("--features é obrigatório para treinar o Random Forest.")
        train_random_forest(args.features, args.models_dir)
    elif args.model == 'cnn':
        if not args.splits:
            parser.error("--splits é obrigatório para treinar a CNN.")
        train_cnn(args.splits, args.models_dir, args.epochs)

if __name__ == '__main__':
    main()

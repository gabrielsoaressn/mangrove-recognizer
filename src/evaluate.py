
import os
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.utils import to_categorical
import base64
import io
import cv2

# Importar funções de extração de features
from feature_extraction import extract_features_from_split

# --- Configuração de Logging ---
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'evaluation.log')),
        logging.StreamHandler()
    ]
)

# --- Seed para Reprodutibilidade ---
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def fig_to_base64(fig):
    """Converte uma figura Matplotlib para uma string base64."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_html_report(report_data, output_path):
    """Gera um relatório HTML com os resultados da avaliação."""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <title>Relatório de Avaliação de Modelo</title>
        <style>
            body {{ font-family: sans-serif; margin: 2em; background-color: #f4f4f9; color: #333; }}
            h1, h2 {{ color: #444; border-bottom: 2px solid #eee; padding-bottom: 0.5em; }}
            .container {{ background-color: #fff; padding: 2em; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            pre {{ background-color: #e9ecef; padding: 1em; border-radius: 4px; white-space: pre-wrap; }}
            img {{ max-width: 100%; height: auto; border-radius: 4px; display: block; margin: 1em auto; }}
            .grid-container {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 1em; margin-top: 1em; }}
            .grid-item {{ text-align: center; border: 1px solid #ddd; padding: 0.5em; border-radius: 4px; }}
            .grid-item img {{ max-width: 100px; }}
            .correct {{ border-color: #28a745; }}
            .incorrect {{ border-color: #dc3545; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Relatório de Avaliação - {report_data['model_name']}</h1>
            
            <h2>Acurácia Geral</h2>
            <pre>{report_data['accuracy']:.4f}</pre>

            <h2>Relatório de Classificação</h2>
            <pre>{report_data['class_report']}</pre>

            <h2>Matriz de Confusão</h2>
            <img src="data:image/png;base64,{report_data['cm_base64']}">

            <h2>Curvas ROC (One-vs-Rest)</h2>
            <img src="data:image/png;base64,{report_data['roc_base64']}">

            <h2>Exemplos de Previsões no Conjunto de Teste</h2>
            <h3>Acertos</h3>
            <div class="grid-container">
                {report_data['correct_examples_html']}
            </div>
            <h3>Erros</h3>
            <div class="grid-container">
                {report_data['incorrect_examples_html']}
            </div>
        </div>
    </body>
    </html>
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    logging.info(f"Relatório de avaliação salvo em {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Avalia um modelo treinado no conjunto de teste.")
    parser.add_argument('--model', type=str, required=True, help="Caminho para o modelo treinado (.pkl ou .h5).")
    parser.add_argument('--test-data', type=str, required=True, help="Caminho para o arquivo de teste (.npz para CNN, .csv para RF).")
    parser.add_argument('--class-map', type=str, required=True, help="Caminho para o arquivo class_map.csv.")
    parser.add_argument('--output-dir', type=str, default='results', help="Diretório para salvar o relatório de avaliação.")
    args = parser.parse_args()

    logging.info(f"Avaliando modelo: {args.model}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Carregar mapa de classes
    class_map_df = pd.read_csv(args.class_map).sort_values('class_int')
    class_names = class_map_df['class_str'].tolist()
    num_classes = len(class_names)

    report_data = {'model_name': os.path.basename(args.model)}

    # Lógica de avaliação
    if args.model.endswith('.pkl'): # Random Forest
        model = joblib.load(args.model)
        # Para o RF, precisamos extrair as features do conjunto de teste primeiro
        logging.info("Modelo RF detectado. Extraindo features do conjunto de teste...")
        test_npz = np.load(args.test_data)
        X_test_img, y_test = test_npz['X'], test_npz['y']
        X_test_features = extract_features_from_split(X_test_img)
        
        y_pred = model.predict(X_test_features)
        y_pred_proba = model.predict_proba(X_test_features)

    elif args.model.endswith('.h5'): # CNN
        model = tf.keras.models.load_model(args.model)
        test_data = np.load(args.test_data)
        X_test_img, y_test = test_data['X'], test_data['y']
        X_test_img_norm = X_test_img.astype('float32') / 255.0
        
        y_pred_proba = model.predict(X_test_img_norm)
        y_pred = np.argmax(y_pred_proba, axis=1)

    else:
        logging.error("Formato de modelo não suportado. Use .pkl para Scikit-learn ou .h5 para Keras.")
        return

    # 1. Acurácia e Relatório de Classificação
    accuracy = np.mean(y_pred == y_test)
    class_report = classification_report(y_test, y_pred, target_names=class_names)
    report_data['accuracy'] = accuracy
    report_data['class_report'] = class_report
    logging.info(f"Acurácia no teste: {accuracy:.4f}")
    logging.info(f"Relatório de Classificação:\n{class_report}")

    # 2. Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title('Matriz de Confusão no Conjunto de Teste')
    ax.set_ylabel('Verdadeiro')
    ax.set_xlabel('Previsto')
    report_data['cm_base64'] = fig_to_base64(fig_cm)

    # 3. Curva ROC
    y_test_cat = to_categorical(y_test, num_classes=num_classes)
    fig_roc, ax = plt.subplots(figsize=(10, 8))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_test_cat[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'ROC curve for {class_names[i]} (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Curva ROC (One-vs-Rest)')
    ax.legend(loc="lower right")
    report_data['roc_base64'] = fig_to_base64(fig_roc)

    # 4. Exemplos de Acertos e Erros
    correct_indices = np.where(y_pred == y_test)[0]
    incorrect_indices = np.where(y_pred != y_test)[0]
    
    def get_examples_html(indices, result_type):
        html = ""
        for i in indices[:10]: # Limitar a 10 exemplos
            img_b64 = base64.b64encode(cv2.imencode('.png', cv2.cvtColor(X_test_img[i], cv2.COLOR_RGB2BGR))[1]).decode('utf-8')
            true_label = class_names[y_test[i]]
            pred_label = class_names[y_pred[i]]
            html += f'''
            <div class="grid-item {result_type}">
                <img src="data:image/png;base64,{img_b64}">
                <p><b>Verdadeiro:</b> {true_label}<br><b>Previsto:</b> {pred_label}</p>
            </div>'''
        return html

    report_data['correct_examples_html'] = get_examples_html(correct_indices, 'correct')
    report_data['incorrect_examples_html'] = get_examples_html(incorrect_indices, 'incorrect')

    # Gerar relatório final
    output_file = os.path.join(args.output_dir, f"evaluation_report_{os.path.basename(args.model).split('.')[0]}.html")
    generate_html_report(report_data, output_file)

if __name__ == '__main__':
    main()

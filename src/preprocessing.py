
import cv2
import numpy as np
import os
import argparse
import logging
from datetime import datetime
from tqdm import tqdm
from PIL import Image, PngImagePlugin
import matplotlib
matplotlib.use('Agg') # Usar backend não-interativo para evitar erros de GUI
import matplotlib.pyplot as plt
import io
import base64
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# --- Configuração de Logging ---
LOGS_DIR = 'logs'
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'preprocessing.log')),
        logging.StreamHandler()
    ]
)

def get_image_stats(image):
    """Calcula estatísticas básicas para cada canal de uma imagem."""
    if image is None:
        return {}
    stats = {}
    channels = cv2.split(image)
    channel_names = ['Blue', 'Green', 'Red'] # OpenCV BGR order
    if image.ndim == 2: # Grayscale
        channel_names = ['Gray']
        channels = [image]

    for i, chan in enumerate(channels):
        name = channel_names[i]
        stats[name] = {
            'mean': np.mean(chan),
            'std': np.std(chan),
            'min': np.min(chan),
            'max': np.max(chan)
        }
    return stats

def plot_histogram(image, title):
    """Gera um histograma RGB e o retorna como uma imagem em base64."""
    if image is None:
        return ""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots()
    
    channels = cv2.split(image)
    colors = ('b', 'g', 'r')
    ax.set_title(f'Histograma de Cores - {title}')
    ax.set_xlabel("Bins")
    ax.set_ylabel("# de Pixels")
    
    for chan, color in zip(channels, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        ax.plot(hist, color=color)
    
    ax.legend(['Canal Azul', 'Canal Verde', 'Canal Vermelho'])
    ax.set_xlim([0, 256])

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def image_to_base64(image, format='png'):
    """Converte uma imagem (array numpy) para uma string base64."""
    is_success, buffer = cv2.imencode(f".{format}", image)
    if not is_success:
        return ""
    return base64.b64encode(buffer).decode('utf-8')

def process_image(image_path, args):
    """Carrega, valida, processa e calcula métricas para uma única imagem."""
    report_data = {}

    # 1. Carregar imagem com Pillow para preservar metadados
    try:
        pil_img = Image.open(image_path)
        # Preservar metadados EXIF/PNG info
        png_info = pil_img.info
        
        # Converter para formato que OpenCV entende (numpy array)
        # Garantir que é RGB
        original_image_rgb = pil_img.convert('RGB')
        original_image = cv2.cvtColor(np.array(original_image_rgb), cv2.COLOR_RGB2BGR)

    except FileNotFoundError:
        logging.error(f"Arquivo não encontrado: {image_path}")
        return None
    except Exception as e:
        logging.error(f"Não foi possível carregar a imagem {image_path}. Erro: {e}")
        return None

    # 2. Validação
    h, w, c = original_image.shape
    if c != 3:
        logging.warning(f"A imagem {image_path} não é RGB. Pulando.")
        return None
    if original_image.dtype != 'uint8':
        logging.warning(f"A imagem {image_path} não é 8-bit. Pulando.")
        return None
    if h < 224 or w < 224:
        logging.warning(f"A imagem {image_path} tem dimensões ({w}x{h}) menores que o mínimo de 224x224. Pulando.")
        return None

    report_data['filename'] = os.path.basename(image_path)
    report_data['original_stats'] = get_image_stats(original_image)
    report_data['original_hist_b64'] = plot_histogram(original_image, "Original")
    report_data['original_img_b64'] = image_to_base64(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))


    # 3. Aplicar Filtro Gaussiano
    kernel_size = (args.gaussian_kernel, args.gaussian_kernel)
    gaussian_blurred = cv2.GaussianBlur(original_image, kernel_size, 1.0)
    report_data['gaussian_img_b64'] = image_to_base64(cv2.cvtColor(gaussian_blurred, cv2.COLOR_BGR2RGB))

    # 4. Aplicar CLAHE em cada canal
    clahe = cv2.createCLAHE(clipLimit=args.clahe_clip, tileGridSize=(8, 8))
    b, g, r = cv2.split(gaussian_blurred)
    clahe_b = clahe.apply(b)
    clahe_g = clahe.apply(g)
    clahe_r = clahe.apply(r)
    clahe_image = cv2.merge([clahe_b, clahe_g, clahe_r])
    report_data['clahe_img_b64'] = image_to_base64(cv2.cvtColor(clahe_image, cv2.COLOR_BGR2RGB))

    processed_image = clahe_image

    # 5. Opcionalmente: Aplicar Filtro de Mediana
    if args.apply_median:
        processed_image = cv2.medianBlur(processed_image, 3)

    report_data['final_img_b64'] = image_to_base64(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    report_data['final_stats'] = get_image_stats(processed_image)
    report_data['final_hist_b64'] = plot_histogram(processed_image, "Processada")

    # 6. Calcular Métricas de Qualidade (convertendo para RGB para SSIM)
    original_rgb_for_metrics = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    processed_rgb_for_metrics = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    
    report_data['psnr'] = psnr(original_rgb_for_metrics, processed_rgb_for_metrics, data_range=255)
    report_data['ssim'] = ssim(original_rgb_for_metrics, processed_rgb_for_metrics, multichannel=True, channel_axis=-1, data_range=255)
    
    return processed_image, png_info, report_data

def generate_html_report(report_data_list, report_dir):
    """Gera um relatório HTML com os resultados do pré-processamento."""
    if not report_data_list:
        logging.info("Nenhum dado para gerar relatório.")
        return

    html_content = """
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <title>Relatório de Pré-processamento de Imagens</title>
        <style>
            body { font-family: sans-serif; margin: 2em; background-color: #f4f4f9; color: #333; }
            h1 { text-align: center; color: #444; }
            .image-report { border: 1px solid #ddd; border-radius: 8px; margin-bottom: 2em; padding: 1em; background-color: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .image-report h2 { color: #0056b3; border-bottom: 2px solid #eee; padding-bottom: 0.5em; }
            .grid-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1em; margin-top: 1em; }
            .grid-item { text-align: center; }
            .grid-item img { max-width: 100%; height: auto; border-radius: 4px; }
            .grid-item h3 { margin-bottom: 0.5em; font-size: 1em; color: #555; }
            table { width: 100%; border-collapse: collapse; margin-top: 1em; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #e9ecef; }
            .metrics { margin-top: 1em; padding: 1em; background-color: #e9f5ff; border-left: 5px solid #007bff; }
        </style>
    </head>
    <body>
        <h1>Relatório de Pré-processamento de Imagens</h1>
    """

    for data in report_data_list:
        html_content += f"""
        <div class="image-report">
            <h2>{data['filename']}</h2>
            
            <div class="grid-container">
                <div class="grid-item">
                    <h3>Original</h3>
                    <img src="data:image/png;base64,{data['original_img_b64']}">
                </div>
                <div class="grid-item">
                    <h3>Filtro Gaussiano</h3>
                    <img src="data:image/png;base64,{data['gaussian_img_b64']}">
                </div>
                <div class="grid-item">
                    <h3>CLAHE</h3>
                    <img src="data:image/png;base64,{data['clahe_img_b64']}">
                </div>
                <div class="grid-item">
                    <h3>Final</h3>
                    <img src="data:image/png;base64,{data['final_img_b64']}">
                </div>
            </div>

            <div class="grid-container">
                <div class="grid-item">
                    <h3>Histograma Original</h3>
                    <img src="data:image/png;base64,{data['original_hist_b64']}">
                </div>
                <div class="grid-item">
                    <h3>Histograma Final</h3>
                    <img src="data:image/png;base64,{data['final_hist_b64']}">
                </div>
            </div>

            <h3>Estatísticas por Canal</h3>
            <table>
                <tr>
                    <th>Canal</th>
                    <th>Métrica</th>
                    <th>Antes</th>
                    <th>Depois</th>
                </tr>
        """
        for channel in data['original_stats']:
            for metric in data['original_stats'][channel]:
                html_content += f"""
                <tr>
                    <td>{channel}</td>
                    <td>{metric}</td>
                    <td>{data['original_stats'][channel][metric]:.2f}</td>
                    <td>{data['final_stats'][channel][metric]:.2f}</td>
                </tr>
                """
        html_content += """
            </table>

            <div class="metrics">
                <h3>Métricas de Qualidade</h3>
                <p><b>PSNR:</b> {:.2f} dB</p>
                <p><b>SSIM:</b> {:.4f}</p>
            </div>
        </div>
        """.format(data['psnr'], data['ssim'])

    html_content += """
    </body>
    </html>
    """
    
    report_path = os.path.join(report_dir, 'preprocessing_report.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    logging.info(f"Relatório salvo em: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Script para pré-processamento de imagens de manguezais.")
    parser.add_argument('--input', type=str, required=True, help="Diretório com as imagens brutas (data/raw/).")
    parser.add_argument('--output', type=str, required=True, help="Diretório para salvar as imagens processadas (data/preprocessed/).")
    parser.add_argument('--report', type=str, required=True, help="Diretório para salvar o relatório HTML (results/).")
    parser.add_argument('--gaussian-kernel', type=int, default=5, help="Tamanho do kernel Gaussiano (deve ser ímpar).")
    parser.add_argument('--clahe-clip', type=float, default=2.0, help="Limite de contraste para o CLAHE.")
    parser.add_argument('--apply-median', action='store_true', help="Aplica um filtro de mediana 3x3 adicional.")
    parser.add_argument('--verbose', action='store_true', help="Mostra progresso detalhado.")
    
    args = parser.parse_args()

    if args.gaussian_kernel % 2 == 0:
        logging.error("O tamanho do kernel Gaussiano deve ser um número ímpar.")
        return

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.report, exist_ok=True)

    logging.info("Iniciando processo de pré-processamento...")
    logging.info(f"Argumentos: {args}")

    image_paths = []
    for root, _, files in os.walk(args.input):
        for file in files:
            if file.lower().endswith('.png'):
                image_paths.append(os.path.join(root, file))
    
    all_report_data = []
    
    progress_bar = tqdm(image_paths, desc="Processando imagens", disable=not args.verbose)
    for image_path in progress_bar:
        try:
            result = process_image(image_path, args)
            if result:
                processed_image, png_info, report_data = result
                
                # Construir caminho de saída mantendo a subestrutura
                relative_path = os.path.relpath(image_path, args.input)
                output_path = os.path.join(args.output, relative_path)
                
                # Criar subdiretório de saída se não existir
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Adicionar sufixo ao nome do arquivo
                filename, ext = os.path.splitext(output_path)
                final_output_path = f"{filename}_processed{ext}"

                # Salvar com Pillow para manter metadados
                processed_pil_img = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
                
                # Pillow precisa de um objeto PngInfo para salvar metadados
                pnginfo_obj = PngImagePlugin.PngInfo()
                for key, value in png_info.items():
                    if isinstance(key, str):
                         pnginfo_obj.add_text(key, str(value))

                processed_pil_img.save(final_output_path, 'PNG', pnginfo=pnginfo_obj)

                all_report_data.append(report_data)
        except Exception as e:
            logging.error(f"Erro ao processar {image_path}: {e}", exc_info=True)

    if all_report_data:
        generate_html_report(all_report_data, args.report)
    else:
        logging.warning("Nenhuma imagem foi processada com sucesso. O relatório não será gerado.")

    logging.info("Pré-processamento concluído.")

if __name__ == '__main__':
    main()

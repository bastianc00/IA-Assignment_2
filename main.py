import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
import clip
import torchvision.models as models
import torch.hub

class ImageRetrievalEvaluator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.datasets = { # Definición de datasets
            'simple1k': {
                'path': 'simple1K',
                'image_dir': 'images',
                'list_file': 'list_of_images.txt'
            },
            'voc': {
                'path': 'VOC_val',
                'image_dir': 'images',
                'list_file': 'list_of_images.txt'
            },
            'paris': {
                'path': 'Paris_val',
                'image_dir': 'images',
                'list_file': 'list_of_images.txt'
            }
        }
        self.results_dir = 'results' # Directorio para guardar resultados
        os.makedirs(self.results_dir, exist_ok=True)

    def load_model(self, model_name):
        """Carga el modelo especificado"""
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=True).to(self.device)
            model.fc = torch.nn.Identity()
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            dim = 512
            
        elif model_name == 'resnet34':
            model = models.resnet34(pretrained=True).to(self.device)
            model.fc = torch.nn.Identity()
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            dim = 512
            
        elif model_name == 'dinov2':
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            dim = 384
            
        elif model_name == 'clip':
            
            model, preprocess = clip.load("ViT-B/32", self.device)
            model = model.encode_image
            dim = 512
            
        return model, preprocess, dim

    def load_dataset(self, dataset_name):
        """Carga la información del dataset"""
        dataset_info = self.datasets[dataset_name]
        image_paths = []
        classes = []
        #aqui se carga la lista de imágenes y clases del list_of_images.txt
        #simple1k/list_of_images.txt
        if dataset_name == 'simple1k':
            with open(os.path.join(dataset_info['path'], dataset_info['list_file']), 'r+') as f:
                for line in f:
                    parts = line.strip().split()
                    rel_path = parts[0] #240002.jpg
                    class_name = parts[1] #dolphin
                    full_path = os.path.join(dataset_info['path'], dataset_info['image_dir'], rel_path)
                    #aqui se crea la ruta completa de la imagen simple1k/images/dolphin/240002.jpg
                    image_paths.append(full_path)
                    classes.append(class_name)
                    
            return image_paths, classes
        
        elif dataset_name == 'paris':
            with open(os.path.join(dataset_info['path'], dataset_info['list_file']), 'r+') as f:
                for line in f:
                    parts = line.strip().split()
                    rel_path = '/'.join(parts[0].split('/')[1:])
                    class_name = parts[1] 
                    full_path = os.path.join(dataset_info['path'], dataset_info['image_dir'], rel_path)
                    
                    image_paths.append(full_path)
                    classes.append(class_name)
            return image_paths, classes
        
        else:
            with open(os.path.join(dataset_info['path'], dataset_info['list_file']), 'r+') as f:
                for line in f:
                    parts = line.strip().split()
                    rel_path = parts[0] 
                    class_name = parts[1] 
                    full_path = os.path.join(dataset_info['path'], dataset_info['image_dir'], rel_path)
                    image_paths.append(full_path)
                    classes.append(class_name)
            return image_paths, classes

    def extract_features(self, model, preprocess, image_paths):
        """Extrae características para todas las imágenes"""
        features = []
        for path in image_paths:
            image = Image.open(path).convert('RGB')
            image = preprocess(image).unsqueeze(0).to(self.device)
            
            #Pasa la imagen por el modelo y extrae las características
            with torch.no_grad():
                feat = model(image).cpu().numpy().flatten()
                #dim=feat.shape[1] # Dimensión de las características
            features.append(feat)
        return np.array(features)

    def compute_similarity(self, features):
        """Calcula la matriz de similitud coseno"""
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norm_features = features / norms
        sim_matrix = norm_features @ np.transpose(norm_features)
        sim_idx=np.argsort(-sim_matrix, axis = 1)
        return sim_matrix, sim_idx

    def evaluate_retrieval(self, sim_matrix, sim_idx, classes):
        """Evalúa el rendimiento de recuperación"""
        aps = []
        precisions = []
        recalls = []
        
        for i in range(len(classes)):
            query_class = classes[i]
            similarities = sim_matrix[i]
            
            # Ordenar por similitud (excluyendo la consulta misma)
            sorted_indices = np.argsort(similarities)[::-1]
            sorted_indices = sorted_indices[sorted_indices != i]
            
            # Calcular relevancia (1 si misma clase, 0 si no)
            relevant = np.array([1 if classes[j] == query_class else 0 for j in sorted_indices])
            
            # Calcular precisión acumulativa
            cum_relevant = np.cumsum(relevant)
            precision = cum_relevant / (np.arange(len(relevant)) + 1)
            recall = cum_relevant / max(1, sum(relevant))
            
            # Calcular Average Precision
            ap = average_precision_score(relevant, similarities[sorted_indices])
            aps.append(ap)
            
            precisions.append(precision)
            recalls.append(recall)
            
        # Calcular mAP
        mAP = np.mean(aps)
        
        return mAP, aps, precisions, recalls

    def plot_precision_recall(self, precisions, recalls, model_name, dataset_name):
        """Grafica la curva precisión-recall promedio"""
        # Interpolar a 11 puntos de recall estándar
        recall_points = np.linspace(0, 1, 11)
        avg_precision = np.zeros_like(recall_points)
        
        for p, r in zip(precisions, recalls):
            interp_p = np.interp(recall_points, r, p)
            avg_precision += interp_p
            
        avg_precision /= len(precisions)
        
        plt.figure()
        plt.plot(recall_points, avg_precision, marker='o')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall: {model_name} on {dataset_name}')
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, f'pr_curve_{model_name}_{dataset_name}.png'))
        plt.close()

    def save_examples(self, sim_matrix, image_paths, classes, model_name, dataset_name):
        """Guarda ejemplos de mejores y peores resultados"""
        # Seleccionar 5 consultas al azar
        query_indices = np.random.choice(len(image_paths), 5, replace=False)
        
        for i, query_idx in enumerate(query_indices):
            query_class = classes[query_idx]
            similarities = sim_matrix[query_idx]
            
            # Ordenar resultados (excluyendo la consulta)
            sorted_indices = np.argsort(similarities)[::-1]
            sorted_indices = sorted_indices[sorted_indices != query_idx]
            
            # Mejores resultados (top 5)
            best_indices = sorted_indices[:5]
            self.plot_results(query_idx, best_indices, image_paths, classes, 
                            f'best_{i}_{model_name}_{dataset_name}')
            
            # Peores resultados de la misma clase (bottom 5)
            class_indices = [j for j in range(len(classes)) if classes[j] == query_class and j != query_idx]
            if len(class_indices) > 5:
                worst_indices = sorted_indices[-5:]
                self.plot_results(query_idx, worst_indices, image_paths, classes,
                                 f'worst_{i}_{model_name}_{dataset_name}')

    def plot_results(self, query_idx, result_indices, image_paths, classes, filename):
        """Grafica la consulta y los resultados"""
        fig, axes = plt.subplots(1, len(result_indices)+1, figsize=(15, 3))
        
        # Mostrar imagen query
        query_img = Image.open(image_paths[query_idx])
        axes[0].imshow(query_img)
        axes[0].set_title('Query\n' + classes[query_idx])
        axes[0].axis('off')
        
        # Mostrar resultados
        for j, idx in enumerate(result_indices, 1):
            res_img = Image.open(image_paths[idx])
            axes[j].imshow(res_img)
            axes[j].set_title(f'Rank {j}\n' + classes[idx])
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, filename + '.png'))
        plt.close()
        
    def menu(self):
        """Displays the options menu"""
        print("Options Datasets:")
        print("1. Evaluate on simple1k")
        print("2. Evaluate on VOC")
        print("3. Evaluate on Paris")
        print("4. Evaluate on all datasets")
        print("5. Exit")
        choice = input("Select an option: ")
        
        if choice == '1':
            self.run_evaluation('simple1k')
        elif choice == '2': # Toma tiempo este dataset
            self.run_evaluation('voc')
        elif choice == '3':
            self.run_evaluation('paris')
        elif choice == '4': #la suma de todos los datasets
            print("Evaluating on all datasets...")
            for dataset in self.datasets.keys():
                self.run_evaluation(dataset)
        elif choice == '5':
            print("Exiting...")
            exit()
        else:
            print("\nInvalid option.\n")
            self.menu()

    def run_evaluation(self, dataset_name):
        """Ejecuta la evaluación completa"""
        models_to_evaluate = ['resnet18', 'resnet34', 'dinov2', 'clip']
        
        
        print(f"\nEvaluating on dataset: {dataset_name}")
        # Cargar la lista de imágenes y clases
        image_paths, classes = self.load_dataset(dataset_name)  # Retorna una lista de rutas de imágenes y una lista de clases
        
        # Ahora se recorre la lista de modelos a evaluar
        for model_name in models_to_evaluate:
            print(f"\nModel processing: {model_name} \n")
            
            # Cargar modelo y extraer características
            model, preprocess, _ = self.load_model(model_name)
            
            #Revisar en adelante según el sample y el clip puede tener confusión con otras librerías
            
            # Extraer características para todas las imágenes y el paso de la imagen por el modelo
            features = self.extract_features(model, preprocess, image_paths)
             
            # Calcular similitudes
            sim_matrix, sim_idx = self.compute_similarity(features)
            
            """ Revisar en adelante las funciones para ver si cumplen con el objetivo"""
            # Evaluar rendimiento
            mAP, aps, precisions, recalls = self.evaluate_retrieval(sim_matrix, sim_idx, classes)
            print(f"mAP para {model_name} en {dataset_name}: {mAP:.4f}")
            
            # Guardar resultados
            self.plot_precision_recall(precisions, recalls, model_name, dataset_name)
            self.save_examples(sim_matrix, image_paths, classes, model_name, dataset_name)
            
            # Guardar métricas
            document_name = f"metrics_{dataset_name}.txt"
            with open(os.path.join(self.results_dir, document_name), 'a') as f:
                f.write(f"{model_name},{dataset_name},{mAP:.4f}\n")

if __name__ == '__main__':
    evaluator = ImageRetrievalEvaluator()
    evaluator.menu()
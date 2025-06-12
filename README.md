# G-TIGRE

Este repositorio implementa **G-TIGRE** (Generative Time series Imputation with Graph-based REcurrent neural networks), un marco generativo basado en redes adversarias (GANs) y redes neuronales de grafos (GNNs) para la imputación de series temporales multivariantes sobre grafos.

G-TIGRE permite entrenar sin observaciones completas y captura dependencias espaciotemporales usando GNNs en secuencias de grafos. El modelo integra una lectura bidireccional de la serie y realiza múltiples muestreos para escoger la imputación con menor evaluación adversaria.

## Contribuciones principales
- Integración de GANs y GNNs para imputación de MTSI en grafos
- Entrenamiento sin requerir datos totalmente observados
- Captura de relaciones espaciotemporales mediante GNNs
- Representación bidireccional (pasado/futuro) y múltiples imputaciones para mayor robustez
- Comparación con el estado del arte en METR-LA, PEMS-BAY y validación clínica en MIMIC-III

## Arquitectura
G-TIGRE extiende la arquitectura de GAIN. Tanto el generador como el discriminador incluyen módulos GNN, permiten leer la serie en ambas direcciones y emplean una combinación de pérdida adversarial (\(\mathcal{L}_G\)) y de reconstrucción (\(\mathcal{L}_M\)).

## Estructura principal
``` 
├── docker/           # Ficheros para crear el entorno de ejecución
├── scripts/          # Programas para lanzar experimentos y procesar resultados
├── src/
│   ├── data/         # Cargadores y preprocesamiento de datasets (tráfico, MIMIC-III, etc.)
│   ├── experiment/   # Clases para definir y ejecutar experimentos
│   ├── models/       # Definición de modelos GNN y pérdidas GAN
│   └── utils.py      # Utilidades varias
└── requirements.txt  # Dependencias del proyecto
```

### Datasets disponibles
En `src/data` se incluyen cargadores para conjuntos de tráfico (METR-LA, PEMS-BAY), registros clínicos MIMIC-III y otras series eléctricas y de calidad del aire.

## Ejecución básica
Un ejemplo para lanzar una búsqueda aleatoria de hiperparámetros es:
```bash
python scripts/random_search.py --datasets la,bay --models grugcn --iterations 50 --gpu 0
```
Los resultados se guardan en `results/` y los scripts permiten seleccionar modelo, dataset y tipo de experimento.

## Docker
Se proporciona `Dockerfile` y `docker-compose.yaml` para reproducir el entorno. Para construir la imagen y acceder al contenedor:
```bash
chmod +x setup.sh
./setup.sh
```

## Licencia
Este proyecto se distribuye bajo la licencia Apache 2.0. Consulta el archivo `LICENSE` para más información.

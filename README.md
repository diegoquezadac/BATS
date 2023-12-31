# BATS: Bridging Acoustic Transparency in Speech

El reconocimiento de voz se basa en representaciones de señales acústicas, como espectrogramas y MFCCs. Sin embargo, los modelos actuales son en gran medida opacos en cuanto a cómo toman decisiones en este proceso. La naturaleza física de los datos de entrada en el reconocimiento de voz agrega una capa adicional de complejidad, lo que plantea el desafío de mejorar la transparencia y la comprensión de estos modelos para garantizar un reconocimiento de voz más preciso y confiable

## Objetivo

El objetivo principal de esta investigación es incorporar explicabilidad a un modelo robusto dedicado a la tarea de reconocimiento de voz mediante métodos *post-hoc*. Como alternativa, se podría considerar el diseño de un modelo más sencillo y, por lo tanto, más interpretable, que, en un contexto específico de interés, alcance un rendimiento comparable al de los modelos complejos actuales.


## Uso

Para ejecutar el código de este repositorio se necesita `Python 3.10` y las dependencias especificadas en el archivo `requirements.txt`. Se recomienda el uso de un entorno virtual para evitar conflictos con otras versiones de las dependencias.

```bash
conda create -n bats python=3.10
conda activate bats
pip install -r requirements.txt
```

Los metodos de explicabilidad se encuentran en la carpeta
`src/models`. 

se incluyen los jupyter notebooks `slime.ipynb` y `representation_erasure.ipynb` que contienen ejemplos de uso de los metodos de explicabilidad.



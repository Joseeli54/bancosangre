const tf = require('@tensorflow/tfjs');
const fs = require('fs');

// Configuración
const dataFiles = fs.readdirSync('datasetCentroid').filter(file => file.startsWith('data_'));
const diagnoses = ['Anemia', 'Leucopenia', 'Trombocitopenia', 'Healthy'];

// Funciones de preprocesamiento
const encodeFeatures = (data) => {
  const features = data.map(d => [
    d.gender === 'Hombre' ? 1 : 0, // Codificar género
    d.redBloodCells,
    d.hemoglobin,
    d.hematocrit,
    d.whiteBloodCells,
    d.platelets
  ]);
  // Normalizar las características
  const xs = tf.tensor2d(features);
  const min = xs.min(0);
  const max = xs.max(0);
  const normalizedXs = xs.sub(min).div(max.sub(min));
  return { normalizedXs, min, max };
};

const encodeDiagnosis = (diagnosis) => {
  const index = diagnoses.indexOf(diagnosis);
  return tf.oneHot(tf.tensor1d([index], 'int32'), diagnoses.length).reshape([diagnoses.length]);
};

// Función para decodificar el diagnóstico
const decodeDiagnosis = (tensor) => {
  const index = tensor.argMax(-1).dataSync()[0];
  return diagnoses[index];
};

// Cargar y preprocesar un chunk
const loadDataChunk = (filePath) => {
  const rawData = fs.readFileSync(filePath);
  const data = JSON.parse(rawData);
  const { normalizedXs, min, max } = encodeFeatures(data);
  const ys = tf.tensor2d(data.map(d => encodeDiagnosis(d.diagnosis).arraySync()));
  return { normalizedXs, ys, min, max };
};

// Definir el modelo
const model = tf.sequential();
model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [6] })); // Primera capa oculta con más unidades
model.add(tf.layers.dropout({ rate: 0.5 })); // Dropout después de la primera capa oculta
model.add(tf.layers.dense({ units: 32, activation: 'relu' })); // Segunda capa oculta
model.add(tf.layers.dropout({ rate: 0.5 })); // Dropout después de la segunda capa oculta
model.add(tf.layers.dense({ units: 16, activation: 'relu' })); // Tercera capa oculta
model.add(tf.layers.dropout({ rate: 0.5 })); // Dropout después de la tercera capa oculta
model.add(tf.layers.dense({ units: diagnoses.length, activation: 'softmax' })); // Capa de salida

model.compile({
  optimizer: tf.train.adam(),
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy']
});

// Entrenar el modelo por chunks
const trainModelInChunks = async () => {
  let globalMin = null;
  let globalMax = null;

  for (let epoch = 0; epoch < 1; epoch++) { // Cambiar el número de épocas según se requiera
    console.log(`Epoch ${epoch + 1}/3`);
    let epochLoss = 0;
    let epochAcc = 0;
    let batchCount = 0;

    for (const file of dataFiles) {
      console.log(`Entrenando con ${file}...`);
      const { normalizedXs, ys, min, max } = loadDataChunk(`datasetCentroid/${file}`);

      // Actualizar los valores globales de min y max para la normalización
      if (!globalMin || !globalMax) {
        globalMin = min;
        globalMax = max;
      } else {
        globalMin = tf.minimum(globalMin, min);
        globalMax = tf.maximum(globalMax, max);
      }

      const history = await model.fit(normalizedXs, ys, {
        epochs: 1,
        verbose: 0,
        callbacks: {
          onBatchEnd: (batch, logs) => {
            epochLoss += logs.loss;
            epochAcc += logs.acc;
            batchCount++;
          }
        }
      });

      normalizedXs.dispose();
      ys.dispose();
    }

    // Calcular y mostrar el promedio de loss y accuracy por epoch
    console.log(`Epoch ${epoch + 1}: loss = ${(epochLoss / batchCount).toFixed(4)}, accuracy = ${(epochAcc / batchCount).toFixed(4)}`);
  }

  // Guardar el modelo manualmente en formato JSON
  const modelJSON = model.toJSON();
  fs.writeFileSync('model/model.json', JSON.stringify(modelJSON));

  // Obtener los pesos del modelo como un array de Promesas
  const weightPromises = model.getWeights().map(w => w.array());

  // Esperar a que todas las Promesas se resuelvan
  const weightData = await Promise.all(weightPromises);

  // Escribir los pesos en un archivo JSON
  fs.writeFileSync('model/weights.json', JSON.stringify(weightData, null, 2));

  // Guardar globalMin y globalMax en archivos JSON
  const globalMinArray = await globalMin.array();
  const globalMaxArray = await globalMax.array();
  fs.writeFileSync('model/globalMin.json', JSON.stringify(globalMinArray));
  fs.writeFileSync('model/globalMax.json', JSON.stringify(globalMaxArray));

  console.log('Pesos del modelo y valores globales de min y max guardados en formato JSON.');
};

// Ejecutar el entrenamiento
trainModelInChunks();

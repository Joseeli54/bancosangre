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
  const normalizedXs = xs.sub(xs.min(0)).div(xs.max(0).sub(xs.min(0)));
  return normalizedXs;
};

const encodeDiagnosis = (diagnosis) => {
  const index = diagnoses.indexOf(diagnosis);
  return tf.oneHot(tf.tensor1d([index], 'int32'), diagnoses.length).reshape([diagnoses.length]);
};

// Cargar y preprocesar un chunk
const loadDataChunk = (filePath) => {
  const rawData = fs.readFileSync(filePath);
  const data = JSON.parse(rawData);
  const xs = encodeFeatures(data);
  const ys = tf.tensor2d(data.map(d => encodeDiagnosis(d.diagnosis).arraySync()));
  return { xs, ys };
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
  for (let epoch = 0; epoch < 3; epoch++) { // Cambiar el número de épocas según se requiera
    console.log(`Epoch ${epoch + 1}/3`);
    let epochLoss = 0;
    let epochAcc = 0;
    let batchCount = 0;

    for (const file of dataFiles) {
      console.log(`Entrenando con ${file}...`);
      const { xs, ys } = loadDataChunk(`datasetCentroid/${file}`);

      const history = await model.fit(xs, ys, {
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

      xs.dispose();
      ys.dispose();
    }

    // Calcular y mostrar el promedio de loss y accuracy por epoch
    console.log(`Epoch ${epoch + 1}: loss = ${(epochLoss / batchCount).toFixed(4)}, accuracy = ${(epochAcc / batchCount).toFixed(4)}`);
  }

   // Guardar el modelo manualmente en formato JSON
   const modelJSON = model.toJSON();
   fs.writeFileSync('model/model.json', JSON.stringify(modelJSON));
   // Guardar los pesos del modelo en un archivo separado
   const weights = model.getWeights();
   const weightData = weights.map(w => w.dataSync());
   fs.writeFileSync('model/weights.bin', Buffer.from(Float32Array.from(weightData.flat()).buffer));
   console.log('Modelo y pesos guardados en formato JSON y BIN.');
};

// Ejecutar el entrenamiento
trainModelInChunks();

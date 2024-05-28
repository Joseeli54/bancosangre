const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');

// Directorio que contiene los archivos JSON
const dataDir = 'dataset/';

// Función para generar rangos
const generateRange = (min, max, step) => {
  const range = [];
  for (let i = min; i <= max; i += step) {
    range.push(i);
  }
  return range;
};

// Función para cargar datos de un archivo JSON
const loadChunk = (filePath) => {
  const rawData = fs.readFileSync(filePath);
  return JSON.parse(rawData);
};

// Función para preprocesar datos
const diagnoses = ['Anemia', 'Leucopenia', 'Trombocitopenia', 'Healthy'];

const encodeFeatures = (d) => {
  return [
    d.gender === 'Hombre' ? 1 : 0, // Encode gender as 1 for Hombre and 0 for Mujer
    d.redBloodCells,
    d.hemoglobin,
    d.hematocrit,
    d.whiteBloodCells,
    d.platelets
  ];
};

const encodeDiagnosis = (diagnosis) => {
  const index = diagnoses.indexOf(diagnosis);
  return tf.oneHot(tf.tensor1d([index], 'int32'), diagnoses.length).reshape([diagnoses.length]);
};

const decodeDiagnosis = (tensor) => {
  const index = tensor.argMax(-1).dataSync()[0];
  return diagnoses[index];
};

// Definir el modelo
const model = tf.sequential();
model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [6] }));
model.add(tf.layers.dense({ units: diagnoses.length, activation: 'softmax' }));

model.compile({
  optimizer: tf.train.adam(),
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy']
});

// Función para entrenar el modelo en un chunk de datos
const trainOnChunk = async (chunk) => {
  const xs = tf.stack(chunk.map(d => tf.tensor(encodeFeatures(d))));
  const ys = tf.stack(chunk.map(d => encodeDiagnosis(d.diagnosis)));
  
  await model.fit(xs, ys, {
    epochs: 1, // Entrena solo una epoch en cada chunk
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Chunk training: loss = ${logs.loss}, accuracy = ${logs.acc}`);
      }
    }
  });
};

const main = async () => {
  const files = fs.readdirSync(dataDir);
  const numEpochs = 2; // Número total de épocas

  for (let epoch = 0; epoch < numEpochs; epoch++) {
    console.log(`Epoch ${epoch + 1}/${numEpochs}`);

    for (const file of files) {
      console.log(`Loading and training on ${file}`)
      const filePath = path.join(dataDir, file);
      const chunk = loadChunk(filePath);
      await trainOnChunk(chunk);
    }
  }

  // Guardar el modelo manualmente en formato JSON
  const modelJSON = model.toJSON();
  fs.writeFileSync('model/model.json', JSON.stringify(modelJSON));
  // Guardar los pesos del modelo en un archivo separado
  const weights = model.getWeights();
  const weightData = weights.map(w => w.dataSync());
  fs.writeFileSync('model/weights.bin', Buffer.from(Float32Array.from(weightData.flat()).buffer));
  console.log('Modelo y pesos guardados en formato JSON y BIN.');

  // Predecir con el modelo entrenado
  const sampleData = {
    gender: 'Hombre',
    redBloodCells: 4500,
    hemoglobin: 140,
    hematocrit: 40,
    whiteBloodCells: 5000,
    platelets: 2000
  };
  const input = tf.tensor([encodeFeatures(sampleData)]);
  const prediction = model.predict(input);
  console.log(decodeDiagnosis(prediction));
};

main();

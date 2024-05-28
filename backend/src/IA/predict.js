const tf = require('@tensorflow/tfjs');
const fs = require('fs');

// Configuración
const diagnoses = ['Anemia', 'Leucopenia', 'Trombocitopenia', 'Healthy'];

// Función para cargar el modelo
const loadModel = async () => {
  const modelJSON = JSON.parse(fs.readFileSync('model/model.json'));
  const weights = new Float32Array(fs.readFileSync('model/weights.bin').buffer);
  
  const model = await tf.models.modelFromJSON(modelJSON);
  const weightShapes = model.weights.map(w => w.shape);
  const weightSizes = weightShapes.map(shape => tf.util.sizeFromShape(shape));
  const weightTensors = [];

  let offset = 0;
  for (let i = 0; i < weightShapes.length; i++) {
    const size = weightSizes[i];
    const shape = weightShapes[i];
    const values = weights.slice(offset, offset + size);
    offset += size;
    weightTensors.push(tf.tensor(values, shape));
  }

  model.setWeights(weightTensors);
  return model;
};

// Funciones de preprocesamiento
const encodeFeatures = (data) => [
  data.gender === 'Hombre' ? 1 : 0, // Codificar género
  data.redBloodCells,
  data.hemoglobin,
  data.hematocrit,
  data.whiteBloodCells,
  data.platelets
];

const decodeDiagnosis = (tensor) => {
  const index = tensor.argMax(-1).dataSync()[0];
  return diagnoses[index];
};

// Predecir utilizando el modelo
const predict = async (inputData) => {
  const model = tf.sequential();
model.add(tf.layers.dense({units: 16, activation: 'relu', inputShape: [6]}));
model.add(tf.layers.dense({units: diagnoses.length, activation: 'softmax'}));

model.compile({
  optimizer: tf.train.adam(),
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy']
});


  const inputTensor = tf.tensor2d([encodeFeatures(inputData)], [1, 6]);
  const prediction = model.predict(inputTensor);
  const diagnosis = decodeDiagnosis(prediction);
  inputTensor.dispose();
  prediction.dispose();
  return diagnosis;
};

// Ejemplo de uso
const inputData = {
  gender: 'Hombre',
  redBloodCells: 4000,
  hemoglobin: 120,
  hematocrit: 37,
  whiteBloodCells: 6000,
  platelets: 2000
};

predict(inputData).then(diagnosis => {
  console.log(`La predicción para el paciente es: ${diagnosis}`);
});










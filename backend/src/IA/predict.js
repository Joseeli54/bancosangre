const tf = require('@tensorflow/tfjs');
const fs = require('fs');

// Diagnósticos posibles
const diagnoses = ['Anemia', 'Leucopenia', 'Trombocitopenia', 'Healthy'];

// Función para cargar el modelo y los pesos
const loadModel = async () => {
  // Cargar el modelo desde JSON
  const modelJSON = JSON.parse(fs.readFileSync('model/model.json', 'utf8'));
  const model = await tf.models.modelFromJSON(modelJSON);

  // Cargar los pesos del modelo
  const weightsBuffer = fs.readFileSync('model/weights.bin');
  const weightsArray = new Float32Array(weightsBuffer.buffer, weightsBuffer.byteOffset, weightsBuffer.length / Float32Array.BYTES_PER_ELEMENT);
  const weightShapes = model.weights.map(weight => weight.shape);
  const weightNames = model.weights.map(weight => weight.originalName);

  let offset = 0;
  for (let i = 0; i < weightShapes.length; i++) {
    const size = tf.util.sizeFromShape(weightShapes[i]);
    const values = weightsArray.subarray(offset, offset + size);
    model.weights[i].val.assign(tf.tensor(values, weightShapes[i]));
    offset += size;
  }

  return model;
};

// Función para preprocesar los datos de entrada
const encodeFeatures = (data) => {
  const features = [
    data.gender === 'Hombre' ? 1 : 0, // Codificar género
    data.redBloodCells,
    data.hemoglobin,
    data.hematocrit,
    data.whiteBloodCells,
    data.platelets
  ];
  // Normalizar las características
  const xs = tf.tensor2d([features]);
  const normalizedXs = xs.sub(xs.min(0)).div(xs.max(0).sub(xs.min(0)));
  return normalizedXs;
};

// Función para decodificar el diagnóstico
const decodeDiagnosis = (tensor) => {
  const index = tensor.argMax(-1).dataSync()[0];
  return diagnoses[index];
};

// Función para realizar predicciones
const predict = async (inputData) => {
  // Definir el modelo
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [6] })); // Primera capa oculta con más unidades
  model.add(tf.layers.dropout({ rate: 0.5 })); // Dropout después de la primera capa oculta
  model.add(tf.layers.dense({ units: 32, activation: 'relu' })); // Segunda capa oculta
  model.add(tf.layers.dropout({ rate: 0.5 })); // Dropout después de la segunda capa oculta
  model.add(tf.layers.dense({ units: 16, activation: 'relu' })); // Tercera capa oculta
  model.add(tf.layers.dropout({ rate: 0.5 })); // Dropout después de la tercera capa oculta
  model.add(tf.layers.dense({ units: diagnoses.length, activation: 'softmax' })); // Capa de salida

  const inputTensor = encodeFeatures(inputData);
  const prediction = model.predict(inputTensor);
  const diagnosis = decodeDiagnosis(prediction);

  inputTensor.dispose();
  prediction.dispose();

  return diagnosis;
};

// Datos de prueba
const testData = {
  gender: 'Hombre',
  redBloodCells: 4500,
  hemoglobin: 160,
  hematocrit: 450,
  whiteBloodCells: 8000,
  platelets: 2500
};

// Realizar predicción y mostrar resultado
predict(testData).then(diagnosis => {
  console.log(`Predicción: ${diagnosis}`);
}).catch(err => {
  console.error('Error realizando la predicción:', err);
});



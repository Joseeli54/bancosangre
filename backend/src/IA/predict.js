const tf = require('@tensorflow/tfjs');
const fs = require('fs');

// Diagnósticos posibles
const diagnoses = ['Anemia', 'Leucopenia', 'Trombocitopenia', 'Healthy'];

// Función para cargar el modelo y los pesos
const loadModel = async () => {
  // Definir el modelo
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [6] })); // Primera capa oculta con más unidades
  model.add(tf.layers.dropout({ rate: 0.5 })); // Dropout después de la primera capa oculta
  model.add(tf.layers.dense({ units: 32, activation: 'relu' })); // Segunda capa oculta
  model.add(tf.layers.dropout({ rate: 0.5 })); // Dropout después de la segunda capa oculta
  model.add(tf.layers.dense({ units: 16, activation: 'relu' })); // Tercera capa oculta
  model.add(tf.layers.dropout({ rate: 0.5 })); // Dropout después de la tercera capa oculta
  model.add(tf.layers.dense({ units: diagnoses.length, activation: 'softmax' })); // Capa de salida

  // Cargar los pesos del modelo
  const weightsBuffer = fs.readFileSync('IA/model/weights.json');
  const weightData = JSON.parse(weightsBuffer.toString());

  // Convertir los datos de los pesos en tensores
  const weightTensors = weightData.map(w => tf.tensor(w));

  // Asignar los pesos cargados al modelo
  model.setWeights(weightTensors);

  return model;
};

// Función para normalizar las características de entrada
const normalizeFeatures = (features, min, max) => {
  const xs = tf.tensor2d(features);
  return xs.sub(min).div(max.sub(min));
};

// Función para preprocesar los datos de entrada
const encodeFeatures = (data, globalMin, globalMax) => {
  const features = data.map(d => [
    d.gender === 'Hombre' ? 1 : 0, // Codificar género
    d.redBloodCells,
    d.hemoglobin,
    d.hematocrit,
    d.whiteBloodCells,
    d.platelets
  ]);
  // Normalizar las características
  return normalizeFeatures(features, globalMin, globalMax);
};

// Función para decodificar el diagnóstico
const decodeDiagnosis = (tensor) => {
  const index = tensor.argMax(-1).dataSync()[0];
  return diagnoses[index];
};

// Función para realizar predicciones
const  predict = async (inputData) => {
  const model = await loadModel();

  // Cargar los valores min y max
  const globalMinArray = JSON.parse(fs.readFileSync('IA/model/globalMin.json'));
  const globalMaxArray = JSON.parse(fs.readFileSync('IA/model/globalMax.json'));

  const globalMin = tf.tensor(globalMinArray);
  const globalMax = tf.tensor(globalMaxArray);

  const inputTensor = encodeFeatures(inputData, globalMin, globalMax);
  const prediction = model.predict(inputTensor);
  const diagnosis = decodeDiagnosis(prediction);

  inputTensor.dispose();
  prediction.dispose();
  globalMin.dispose();
  globalMax.dispose();

  return diagnosis;
};

// Datos de prueba
const testData = [{
  gender: "Hombre",
  redBloodCells: 3640,
  hemoglobin: 0,
  hematocrit: 20,
  whiteBloodCells: 3733,
  platelets: 3200,
}];

// Realizar predicción y mostrar resultado
predict(testData).then(diagnosis => {
  console.log(`Predicción: ${diagnosis}`);
}).catch(err => {
  console.error('Error realizando la predicción:', err);
});


// Exportar las funciones necesarias
module.exports = {
  loadModel,
  normalizeFeatures,
  encodeFeatures,
  decodeDiagnosis,
  predict
};
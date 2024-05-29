
// Preprocesar datos
const bloodTypes = ['ORH+', 'ORH-', 'ARH+', 'ARH-', 'BRH+', 'BRH-', 'ABRH+', 'ABRH-'];
const diagnoses = ['Anemia', 'Diabetes', 'Healthy'];

const encodeBloodType = (type) => {
  const index = bloodTypes.indexOf(type);
  return tf.oneHot(tf.tensor1d([index], 'int32'), bloodTypes.length).reshape([bloodTypes.length]);
};

const encodeDiagnosis = (diagnosis) => {
  const index = diagnoses.indexOf(diagnosis);
  return tf.oneHot(tf.tensor1d([index], 'int32'), diagnoses.length).reshape([diagnoses.length]);
};

const decodeDiagnosis = (tensor) => {
    const index = tensor.argMax(-1).dataSync()[0];
    return diagnoses[index];
  };
  
//export {bloodTypes, diagnoses, encodeBloodType, encodeDiagnosis, decodeDiagnosis};
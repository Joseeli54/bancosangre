const fs = require('fs');

// Función para generar valores centroides dentro de un rango
const generateCentroids = (min, max, numCentroids) => {
    const step = (max - min) / (numCentroids - 1);
    const centroids = [];
    for (let i = 0; i < numCentroids; i++) {
      centroids.push(Math.round(min + i * step));
    }
    return centroids;
};

// Generar el dataset con valores centroides
const generateCentroidDataset = () => {
  const data = [];

  const genders = ['Hombre', 'Mujer'];
  
  // Definir rangos y centroides para los valores
  const redBloodCellsRange = generateCentroids(2400, 5500, 17);
  const hemoglobinRange = generateCentroids(0, 300, 17);
  const hematocritRange = generateCentroids(20, 60, 17);
  const whiteBloodCellsRange = generateCentroids(2000, 15000, 17);
  const plateletsRange = generateCentroids(500, 5000, 17);

  for (const gender of genders) {
    for (const redBloodCells of redBloodCellsRange) {
      for (const hemoglobin of hemoglobinRange) {
        for (const hematocrit of hematocritRange) {
          for (const whiteBloodCells of whiteBloodCellsRange) {
            for (const platelets of plateletsRange) {
              let diagnosis;
              if (gender === 'Hombre' && redBloodCells < 4350 && hemoglobin < 132 && hematocrit < 38.3) {
                diagnosis = 'Anemia';
              } else if (gender === 'Mujer' && redBloodCells < 3920 && hemoglobin < 116 && hematocrit < 38.3) {
                diagnosis = 'Anemia';
              } else if (whiteBloodCells < 3400) {
                diagnosis = 'Leucopenia';
              } else if (gender === 'Hombre' && platelets < 1350) {
                diagnosis = 'Trombocitopenia';
              } else if (gender === 'Mujer' && platelets < 1570) {
                diagnosis = 'Trombocitopenia';
              } else {
                diagnosis = 'Healthy';
              }
              data.push({ gender, redBloodCells, hemoglobin, hematocrit, whiteBloodCells, platelets, diagnosis });
            }
          }
        }
      }
    }
  }

  return data;
};

// Generar el dataset con valores centroides
const dataset = generateCentroidDataset();

// Guardar el dataset en un archivo JSON
fs.writeFileSync('data_centroids.json', JSON.stringify(dataset, null, 2));

console.log('Dataset generado y guardado en data_centroids.json.');

const fs = require('fs');

// Función para generar valores dentro de un rango con un paso específico
const generateRange = (min, max, step) => {
  const range = [];
  for (let i = min; i <= max; i += step) {
    range.push(i);
  }
  return range;
};

// Función para generar y guardar el dataset en fragmentos
const generateAndSaveDataset = (chunkSize) => {
  const genders = ['Hombre', 'Mujer'];

  const redBloodCellsRange = generateRange(3500, 6000, 150);
  const hemoglobinRange = generateRange(100, 200, 15);
  const hematocritRange = generateRange(32, 55, 0.2);
  const whiteBloodCellsRange = generateRange(2700, 12000, 150);
  const plateletsRange = generateRange(1000, 4200, 150);

  let data = [];
  let chunkIndex = 0;

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

              // Guardar el dataset en partes más pequeñas
              if (data.length >= chunkSize) {
                fs.writeFileSync(`dataset/data_part_${chunkIndex}.json`, JSON.stringify(data, null, 2));
                chunkIndex++;
                data = [];
              }
            }
          }
        }
      }
    }
  }

  // Guardar los datos restantes
  if (data.length > 0) {
    fs.writeFileSync(`dataset/data_part_${chunkIndex}.json`, JSON.stringify(data, null, 2));
  }

  console.log('Dataset dividido en partes y guardado en archivos separados.');
};

// Generar y guardar el dataset en fragmentos de tamaño especificado
const chunkSize = 800000;
generateAndSaveDataset(chunkSize);

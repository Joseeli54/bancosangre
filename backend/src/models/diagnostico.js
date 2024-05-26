const mongoose = require('mongoose');

let Schema = mongoose.Schema;

let diagnosticoSchema = new Schema({
    name: {
        type: String,
    },
    description:{
        type: String,
    },
    id_sangre:{ 
        type: String,
    },
    id_user:{
        type: String,
    },
    createBy: {
        type: String,
    }
});

module.exports = mongoose.model('Diagnostico', diagnosticoSchema)
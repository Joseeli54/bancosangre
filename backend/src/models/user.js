const mongoose = require('mongoose');

var uniqueValidator = require('mongoose-unique-validator');

let rolesValidos = {
    values: ["ADMIN", "USER"],
    message: '{VALUE} no es un role válido'
}

let Schema = mongoose.Schema;

let userSchema = new Schema({
    name: {
        type: String,
        required: [true, 'El name es necesario'],
    },
    apellido: {
        type: String,
        required: [true, 'El apellido es necesario'],
    },
    username: {
        type: String,
        unique: true,
        required: [true, 'El name de usuario es necesario'],
    },
    email: {
        type: String,
        unique: true,
        required: [true, "El correo es necesario"],
    },
    telefono: {
        type: String,
        required: [true, "El numero telefónico es necesario"],
    },
    password: {
        type: String,
        required: [true, "Le contraseña es obligatoria"],
    },
    role: {
        type: String,
        default: 'USER',
        required: [true],
        enum: rolesValidos,
    },
    avatar: {
        type: String,
    },
    avatar_public_id: {
        type: String,
    },
    direccion: {
        type: String,
    },
    type: {
        type: String,
    },
    dateBirth: {
        type: String,
    }
});

// elimina la key password del objeto que retorna al momento de crear un usuario
userSchema.methods.toJSON = function() {
    let user = this;
    let userObject = user.toObject();
    delete userObject.password;

    return userObject;
}

userSchema.plugin(uniqueValidator, {
    message: '{PATH} debe de ser único'
})

module.exports = mongoose.model('User', userSchema)
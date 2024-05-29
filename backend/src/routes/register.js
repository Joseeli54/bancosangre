const express = require('express');
const bcrypt = require('bcrypt');
const User = require('./../models/user');
const app = express();
const path = require('path');
const fs = require('fs-extra');
const transporter = require('./../config/mailer');
const multer = require('./../libs/multerUploads');

app.post('/register', multer.fields([
                        { name: 'avatar', maxCount: 1 }    
                    ]), function (req, res) {

    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, PATCH, DELETE'); // If needed
    res.header('Access-Control-Allow-Headers', 'X-Requested-With,content-type'); // If needed
    res.header('Access-Control-Allow-Credentials', true); // If needed

    let body = req.body;
    var avatar = "";

    if(req.files.avatar !== undefined){
        avatar = body.apiUrl +  "src/uploads/" + req.files.avatar[0].filename;
    }

    let { name, apellido, username, email,
        password, role, direccion, 
        avatar_public_id, type, telefono, dateBirth } = body;

    email = email.toLowerCase();
    username = username.toLowerCase();

    if(name == 'undefined' || apellido == 'undefined'){
        return res.status(400).json({
            ok: false,
            err: {
                message: "El name y apellido son necesarios para registrarte"
            }
        })
    }

    if(username == 'undefined'){
        return res.status(400).json({
            ok: false,
            err: {
                message: "Necesitas el name de usuario para registrarte"
            }
        })
    }

    if(email == 'undefined'){
        return res.status(400).json({
            ok: false,
            err: {
                message: "Necesitas el correo electronico para registrarte"
            }
        })
    }

    if(password == 'undefined'){
        return res.status(400).json({
            ok: false,
            err: {
                message: "Debes colocar una contraseña"
            }
        })
    }

    // send mail with defined transport object
    let info = transporter.sendMail({
                from: '"Registro de Usuario" <joseeli12345@gmail.com>', 
                to: email, 
                subject: "Creacion de Cuenta",
                html: `
                    <p>Estimado(a) `+email+`</p>
                    <p> Gracias por registrarse en nuestro sistema. </p>
                    <p> Su contraseña es la siguiente: `+password+` </p>
                    <p> Saludos cordiales. </p>
                `
    });

    let user = new User({
      name,
      apellido,
      username,
      email,
      password: bcrypt.hashSync(password, 10),
      role,
      avatar,
      avatar_public_id,
      direccion,
      type,
      telefono,
      dateBirth
    });

    let output;
    (async () => {
        output = await data.save();
    })

    user.save()
    .then(userDB => {
        console.log(userDB)

        res.json({
            ok: true,
            user: userDB
        });
    })
    .catch(err => {
        return res.status(400).json({
            ok: false,
            err,
        });
    });
});

module.exports = app;


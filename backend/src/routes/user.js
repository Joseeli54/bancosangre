const express = require('express');
const User = require('./../models/user');
const app = express();
const multer = require('./../libs/multerAvatars');
const path = require('path');
const fs = require('fs-extra');
const bcrypt = require('bcrypt');
const cloudinary = require('cloudinary').v2;

// Configurando el Cors en el servidor.
var cors = require('cors');
app.use(cors({
    origin:['http://localhost:4200',
            'http://127.0.0.1:4200', 
            'http://127.0.0.1:8080'],
    credentials:true
}));

app.get("/users/", function (req, res){

    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, PATCH, DELETE'); // If needed
    res.setHeader('Access-Control-Allow-Headers', 'X-Requested-With,content-type'); // If needed
    res.setHeader('Access-Control-Allow-Credentials', true); // If needed
  
    User.find()
    .then(user => {
        console.log('GET /users')
        return res.status(200).jsonp(user);
    })
    .catch(err => {
        return res.send(500, err.message);
    });
});

app.get("/users/:id", function (req, res){

    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, PATCH, DELETE'); // If needed
    res.setHeader('Access-Control-Allow-Headers', 'X-Requested-With,content-type'); // If needed
    res.setHeader('Access-Control-Allow-Credentials', true); // If needed
    
    User.findById(req.params.id)
    .then(user => {
        if(user !== null){
            console.log(user.password);
            console.log('GET /users/' + req.params.id);
            return res.status(200).jsonp(user);
        }else{
            return res.status(400).jsonp({message: "El usuario no existe"});
        }
    })
    .catch(err =>{
        return res.send(500, err.message);
    });
});

app.get("/users/username/:username", function (req, res){

    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, PATCH, DELETE'); // If needed
    res.setHeader('Access-Control-Allow-Headers', 'X-Requested-With,content-type'); // If needed
    res.setHeader('Access-Control-Allow-Credentials', true); // If needed

    data = { username: req.params.username };
    
    User.findOne(data)
    .then(user => {
        if(user !== null){
            console.log(user.password);
            console.log('GET /users/' + req.params.id);
            return res.status(200).jsonp(user);
        }else{
            return res.status(400).jsonp({message: "El usuario no existe"});
        }
    })
    .catch(err =>{
        return res.send(500, err.message);
    });
});

app.put('/users/username/:username', function(req, res){

    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, PATCH, DELETE'); // If needed
    res.header('Access-Control-Allow-Headers', 'X-Requested-With,content-type'); // If needed
    res.header('Access-Control-Allow-Credentials', true); // If needed

    data = { username: req.params.username };

    User.findOne(data)
    .then(user => {
        if(user !== null) {
            user.name   = req.body.name;
            user.apellido = req.body.apellido;
            user.username = req.body.username;
            user.email = req.body.email;
            user.telefono = req.body.telefono;
            user.direccion = req.body.direccion;
            user.type = req.body.type;

            user.save()
            .then(() => {
                return res.status(200).jsonp(user);
            })
            .catch(err => {
                return res.status(400).json({
                    ok: false,
                    err,
                });
            });
        }else{
            res.status(400).jsonp({message: "No se encontraron los datos del usuario"})
        }
    })
    .catch(err => {
        return res.send(500, err.message);
    });

});

app.delete("/users/:id", function (req, res){

    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, PATCH, DELETE'); // If needed
    res.header('Access-Control-Allow-Headers', 'X-Requested-With,content-type'); // If needed
    res.header('Access-Control-Allow-Credentials', true); // If needed

    User.findById(req.params.id)
    .then(user => {
        if(user !== null){
            user.deleteOne()
            .then(() => {
                return res.status(200).jsonp({message: "Eliminación exitosa", user});
            })
            .catch(err => {
                return res.send(500, err.message);
            });
        }
    })
    .catch(err => {
        return res.send(500, err.message);
    });
});

module.exports = app;




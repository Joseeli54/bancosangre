const express = require('express');
const Diagnostico = require('./../models/diagnostico');
const app = express();

app.get("/diagnosticos/", function (req, res){  
    Diagnostico.find().sort( {code:1 } )   
    .then(diagnostico => {
        console.log('GET /diagnosticos')
        return res.status(200).jsonp(diagnostico);
    })
    .catch(err => {
        return res.send(500, err.message);
    });
});

app.get("/diagnosticos/:id", function (req, res){  
    let id = req.params.id;
    Diagnostico.find({_id: id}).sort( {code:1 } )   
    .then(diagnostico => {
        console.log('GET /diagnosticos')
        return res.status(200).jsonp(diagnostico);
    })
    .catch(err => {
        return res.send(500, err.message);
    });
});

app.post("/diagnosticos/", function (req, res){
    let body = req.body; 
    let {name,
        description,
        id_sangre, 
        id_user,
        createBy} = body;
    
    let diagnostico = new Diagnostico({
                            name,
                            description,
                            id_sangre, 
                            id_user,
                            createBy});

    diagnostico.save().then(diagnosticos => {
        console.log(diagnosticos)

        res.json({
            ok: true,
            sangre: diagnosticos
        });
    })
    .catch(err => {
        return res.status(400).json({
            ok: false,
            err,
        });
    });
    
});

app.put("/diagnosticos/:id", function (req, res){
    let id = req.params.id;
    let body = req.body;
    let {name,
        description,
        id_sangre, 
        id_user,
        createBy} = body;

    let arrayUpdate = { 
                        name,
                        description,
                        id_sangre, 
                        id_user,
                        createBy};

    Diagnostico.updateOne({_id: id}, { $set: arrayUpdate})
    .then(data => {
        console.log(data)
        res.json({
            ok: true,
        });
    })
    .catch(err => {
        return res.status(400).json({
            ok: false,
            err,
        });
    });
});

app.delete("/diagnosticos/:id", function (req, res){
    let id = req.params.id;
    Diagnostico.deleteOne({_id: id})
    .then(data => {
        console.log(data)
        res.json({
            ok: true,
        });
    })
    .catch(err => {
        return res.status(400).json({
            ok: false,
            err,
        });
    });
})


module.exports = app;
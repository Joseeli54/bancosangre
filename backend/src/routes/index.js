const express = require('express')
const app = express()

app.use(require('./login'));
app.use(require('./register'));
app.use(require('./user'));
app.use(require('./mail'));
app.use(require('./sangre'));
app.use(require('./diagnostico'));

//setHeader('Access-Control-Allow-Origin', '*');
//setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, PATCH, DELETE'); // If needed
//setHeader('Access-Control-Allow-Headers', 'X-Requested-With,content-type'); // If needed
//setHeader('Access-Control-Allow-Credentials', true); // If needed
    
module.exports = app;
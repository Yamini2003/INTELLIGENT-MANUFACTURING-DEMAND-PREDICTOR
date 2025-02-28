const mongoose = require('mongoose');
const manufacturerSchema = new mongoose.Schema({
    name: String,
    email: { type: String, unique: true },
    password: String
});
module.exports = mongoose.model('Manufacturer', manufacturerSchema);

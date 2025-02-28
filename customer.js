const mongoose = require('mongoose');
const customerSchema = new mongoose.Schema({
    name: String,
    email: { type: String, unique: true },
    password: String
});
module.exports = mongoose.model('Customer', customerSchema);
const express = require('express');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const Customer = require('../models/Customer');

const router = express.Router();

// Register customer
router.post('/register', async (req, res) => {
    const { name, email, password } = req.body;
    const hashedPassword = await bcrypt.hash(password, 10);

    try {
        const newCustomer = new Customer({ name, email, password: hashedPassword });
        await newCustomer.save();
        res.status(201).json({ message: "Customer registered" });
    } catch (err) {
        res.status(400).json({ error: err.message });
    }
});

// Login customer
router.post('/login', async (req, res) => {
    const { email, password } = req.body;
    const customer = await Customer.findOne({ email });
    if (customer && await bcrypt.compare(password, customer.password)) {
        const token = jwt.sign({ id: customer._id }, process.env.JWT_SECRET);
        res.json({ message: "Login successful", token });
    } else {
        res.status(400).json({ error: "Invalid credentials" });
    }
});

module.exports = router;

const express = require('express');
const multer = require('multer')

const app = express();

const fileFilter = function(req, file, cb) {
    const allowedtypes = ['audio/wav', 'text/plain']
    if(!allowedtypes.includes(file.mimetype)) {
        const error = new Error("Wrong file type")
        error.code = "LIMIT_FILE_TYPES"
        return cb(error, false)
    }
    cb(null, true)
    // comment
}

const MAX_SIZE = 200000000
const upload = multer({
    dest: './uploads/',
    fileFilter,
    limits: {
        fileSize: MAX_SIZE
    }
})

app.post('/upload', upload.single('file'), (req, res) => {
    res.json({ file: req.file});
});

app.use(function(err, req, res, next) {
    if(err.code === "LIMIT_FILE_TYPES") {
        res.status(422).json({ error: "Only wav files are allowed" })
        return
    }
    if(err.code === "LIMIT_FILE_SIZE") {
        res.status(422).json({ error: `File too large. Max Size is ${MAX_SIZE/100000}Mb` })
        return
    }
})

app.listen(3344, () => console.log("Running on localhost:3344"));
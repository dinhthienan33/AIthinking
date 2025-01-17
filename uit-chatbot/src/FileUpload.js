import React, { useState } from 'react';
import { Button, Typography, LinearProgress } from '@material-ui/core';
import axios from 'axios';

function FileUpload() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      setUploadStatus('Please select a file first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    setUploading(true);
    try {
      await axios.post('http://localhost:8000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setUploadStatus('File uploaded successfully!');
    } catch (error) {
      console.error('Error uploading file:', error);
      setUploadStatus('Error uploading file. Please try again.');
    }
    setUploading(false);
  };

  return (
    <div style={{ marginTop: '20px' }}>
      <Typography variant="h6" gutterBottom>
        Upload Knowledge
      </Typography>
      <input
        accept=".pdf,.docx"
        style={{ display: 'none' }}
        id="raised-button-file"
        type="file"
        onChange={handleFileChange}
      />
      <label htmlFor="raised-button-file">
        <Button variant="contained" component="span">
          Choose File
        </Button>
      </label>
      {file && <Typography>{file.name}</Typography>}
      <Button
        variant="contained"
        color="primary"
        onClick={handleUpload}
        disabled={!file || uploading}
        style={{ marginLeft: '10px' }}
      >
        Upload
      </Button>
      {uploading && <LinearProgress style={{ marginTop: '10px' }} />}
      {uploadStatus && <Typography>{uploadStatus}</Typography>}
    </div>
  );
}

export default FileUpload;
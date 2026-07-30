import React, { useState, useRef, useCallback } from 'react'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || '/api'

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef(null)

  const handleFile = (selectedFile) => {
    if (!selectedFile) return
    if (!selectedFile.type.startsWith('image/')) {
      setError('Please upload an image file')
      return
    }
    if (selectedFile.size > 10 * 1024 * 1024) {
      setError('File size must be less than 10MB')
      return
    }
    setFile(selectedFile)
    setPreview(URL.createObjectURL(selectedFile))
    setResult(null)
    setError(null)
  }

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    handleFile(e.dataTransfer.files[0])
  }, [])

  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    setDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
  }, [])

  const handleSubmit = async () => {
    if (!file) return
    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 30000,
      })
      setResult(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Prediction failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const getConfidenceColor = (conf) => {
    if (conf >= 0.8) return '#22c55e'
    if (conf >= 0.5) return '#f59e0b'
    return '#ef4444'
  }

  return (
    <div className="container">
      <div className="header">
        <h1>Brain Tumor Detection</h1>
        <p>Upload an MRI/CT scan to detect tumors using YOLOv11</p>
      </div>

      {error && <div className="error">{error}</div>}

      <div
        className={`upload-zone ${dragOver ? 'dragover' : ''}`}
        onClick={() => fileInputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <div className="upload-icon">📤</div>
        <p>Drag & drop an MRI/CT image here, or click to browse</p>
        <p style={{ fontSize: '0.875rem', color: '#64748b', marginTop: '0.5rem' }}>
          Supports JPG, PNG, JPEG (max 10MB)
        </p>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={(e) => handleFile(e.target.files[0])}
        />
      </div>

      {preview && (
        <div style={{ textAlign: 'center', marginBottom: '1.5rem' }}>
          <button
            onClick={handleSubmit}
            disabled={loading}
            style={{
              padding: '0.75rem 2rem',
              fontSize: '1rem',
              fontWeight: 600,
              background: loading ? '#475569' : 'linear-gradient(90deg, #3b82f6, #8b5cf6)',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              cursor: loading ? 'not-allowed' : 'pointer',
              transition: 'all 0.3s',
            }}
          >
            {loading ? 'Analyzing...' : 'Detect Tumors'}
          </button>
        </div>
      )}

      {loading && (
        <div className="loading">
          <div className="spinner"></div>
          <p>Running YOLOv11 inference...</p>
        </div>
      )}

      {result && (
        <div className="results">
          <div className="image-panel">
            <h3>Original Image</h3>
            <img src={preview} alt="Original" />
          </div>

          <div className="image-panel">
            <h3>Detection Result</h3>
            {result.annotated_image ? (
              <img
                src={`data:image/jpeg;base64,${result.annotated_image}`}
                alt="Detected"
              />
            ) : (
              <div style={{ textAlign: 'center', padding: '4rem', color: '#64748b' }}>
                No tumors detected
              </div>
            )}
          </div>

          <div className="stats-panel" style={{ gridColumn: '1 / -1' }}>
            <h3>Detection Summary</h3>
            <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
              <span className={`badge ${result.detection_count > 0 ? 'badge-danger' : 'badge-success'}`}>
                {result.detection_count > 0 ? `${result.detection_count} Tumor(s) Found` : 'No Tumor Detected'}
              </span>
              <span className="badge" style={{ background: 'rgba(96, 165, 250, 0.2)', color: '#60a5fa' }}>
                {result.processing_time_ms}ms
              </span>
              <span className="badge" style={{ background: 'rgba(167, 139, 250, 0.2)', color: '#a78bfa' }}>
                {result.image_width}x{result.image_height}
              </span>
            </div>

            {result.detections.length > 0 && (
              <div>
                <h4 style={{ marginBottom: '0.75rem', color: '#94a3b8' }}>Detected Objects:</h4>
                {result.detections.map((det, idx) => (
                  <div key={idx} className="detection-item">
                    <div>
                      <div className="class-name">{det.class_name}</div>
                      <div className="confidence-bar">
                        <div
                          className="confidence-fill"
                          style={{
                            width: `${det.confidence * 100}%`,
                            background: getConfidenceColor(det.confidence),
                          }}
                        />
                      </div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontSize: '1.25rem', fontWeight: 700, color: getConfidenceColor(det.confidence) }}>
                        {(det.confidence * 100).toFixed(1)}%
                      </div>
                      <div style={{ fontSize: '0.75rem', color: '#64748b' }}>
                        [{det.bbox.map(v => v.toFixed(0)).join(', ')}]
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default App

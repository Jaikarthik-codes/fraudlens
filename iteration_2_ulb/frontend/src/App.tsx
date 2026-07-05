import { useState } from 'react'

/* ────────────────────────────────────────────
   Preset Transactions
   Each preset mirrors the feature vector your
   Flask model expects. Adjust field names to
   match your backend's expected JSON payload.
   ──────────────────────────────────────────── */
interface Transaction {
  label: string
  description: string
  amount: number
  data: Record<string, number>
}

const PRESET_TRANSACTIONS: Transaction[] = [
  {
    label: 'Midnight Luxury Purchase — $529.00',
    description: 'Real fraud case from ULB dataset',
    amount: 529.00,
    data: {
      Time: 406, V1: -2.3122, V2: 1.9520, V3: -1.6099, V4: 3.9979,
      V5: -0.5222, V6: -1.4265, V7: -2.5374, V8: 1.3917, V9: -2.7701,
      V10: -2.7723, V11: 3.2020, V12: -2.8999, V13: -0.5952, V14: -4.2893,
      V15: 0.3897, V16: -1.1407, V17: -2.8301, V18: -0.0168, V19: 0.4170,
      V20: 0.1269, V21: 0.5172, V22: -0.0350, V23: -0.4652, V24: 0.3202,
      V25: 0.0445, V26: 0.1778, V27: 0.2611, V28: -0.1433, Amount: 529.00,
    },
  },
  {
    label: 'Suspicious Foreign Transfer — $239.93',
    description: 'Real fraud case from ULB dataset',
    amount: 239.93,
    data: {
      Time: 472, V1: -3.0435, V2: -3.1573, V3: 1.0885, V4: 2.2886,
      V5: 1.3598, V6: -1.0648, V7: 0.3256, V8: -0.0678, V9: -0.2710,
      V10: -0.8386, V11: -0.4146, V12: -0.5031, V13: 0.6765, V14: -1.6920,
      V15: 2.0006, V16: 0.6668, V17: 0.5997, V18: 1.7253, V19: 0.2833,
      V20: 2.1023, V21: 0.6617, V22: 0.4355, V23: 1.3760, V24: -0.2938,
      V25: 0.2798, V26: -0.1454, V27: -0.2528, V28: 0.0358, Amount: 239.93,
    },
  },
  {
    label: 'Identity Theft Purchase — $239.93',
    description: 'Real fraud case from ULB dataset',
    amount: 239.93,
    data: {
      Time: 4462, V1: -2.3033, V2: 1.7592, V3: -0.3597, V4: 2.3302,
      V5: -0.8216, V6: -0.0758, V7: 0.5623, V8: -0.3991, V9: -0.2383,
      V10: -1.5254, V11: 2.0329, V12: -6.5601, V13: 0.0229, V14: -1.4701,
      V15: -0.6988, V16: -2.2822, V17: -4.7818, V18: -2.6157, V19: -1.3344,
      V20: -0.4300, V21: -0.2942, V22: -0.9324, V23: 0.1727, V24: -0.0873,
      V25: -0.1561, V26: -0.5426, V27: 0.0396, V28: -0.1530, Amount: 239.93,
    },
  },
  {
    label: 'Card Cloning Attempt — $529.00',
    description: 'Real fraud case from ULB dataset',
    amount: 529.00,
    data: {
      Time: 6986, V1: -4.3980, V2: 1.3584, V3: -2.5928, V4: 2.6798,
      V5: -1.1281, V6: -1.7065, V7: -3.4962, V8: -0.2488, V9: -0.2478,
      V10: -4.8016, V11: 4.8958, V12: -10.9128, V13: 0.1844, V14: -6.7711,
      V15: -0.0073, V16: -7.3581, V17: -12.5984, V18: -5.1315, V19: 0.3083,
      V20: -0.1716, V21: 0.5736, V22: 0.1770, V23: -0.4362, V24: -0.0535,
      V25: 0.2524, V26: -0.6575, V27: -0.8271, V28: 0.8496, Amount: 529.00,
    },
  },
  {
    label: 'High-Value Card Exploit — $59.00',
    description: 'Real fraud case from ULB dataset',
    amount: 59.00,
    data: {
      Time: 7519, V1: 1.2342, V2: 3.0197, V3: -4.3046, V4: 4.7328,
      V5: 3.6242, V6: -1.3577, V7: 1.7134, V8: -0.4964, V9: -1.2829,
      V10: -2.4475, V11: 2.1013, V12: -4.6096, V13: 1.4644, V14: -6.0793,
      V15: -0.3392, V16: 2.5819, V17: 6.7394, V18: 3.0425, V19: -2.7219,
      V20: 0.0091, V21: -0.3791, V22: -0.7042, V23: -0.6568, V24: -1.6327,
      V25: 1.4889, V26: 0.5668, V27: -0.0100, V28: 0.1468, Amount: 59.00,
    },
  },
]

/* ────────────────────────────────────────────
   App Component
   ──────────────────────────────────────────── */
interface PredictionResult {
  prediction: string
  confidence: number
}

export default function App() {
  const [selectedIdx, setSelectedIdx] = useState<number>(-1)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const selected = selectedIdx >= 0 ? PRESET_TRANSACTIONS[selectedIdx] : null

  async function handlePredict() {
    if (!selected) return

    setLoading(true)
    setResult(null)
    setError(null)

    try {
      const response = await fetch('https://fraudlens-api-9xhd.onrender.com/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(selected.data),
      })

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`)
      }

      const json: PredictionResult = await response.json()
      setResult(json)
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'An unexpected error occurred'
      setError(message)
    } finally {
      setLoading(false)
    }
  }

  const isFraud = result?.prediction?.toLowerCase() === 'fraud'
  const confidencePercent = result ? (result.confidence * 100).toFixed(1) : '0'

  return (
    <div className="app-container">
      {/* ── Header ── */}
      <header className="header">
        <div className="header__logo">
          <div className="header__icon" aria-hidden="true">⚡</div>
          <h1 className="header__title">FraudLens</h1>
        </div>
        <p className="header__subtitle">AI-powered transaction fraud detection</p>
      </header>

      {/* ── Main Card ── */}
      <main className="card" id="prediction-card">
        {/* Dropdown */}
        <div className="form-group">
          <label className="form-label" htmlFor="transaction-select">
            Select Transaction
          </label>
          <select
            id="transaction-select"
            className="form-select"
            value={selectedIdx}
            onChange={(e) => {
              setSelectedIdx(Number(e.target.value))
              setResult(null)
              setError(null)
            }}
          >
            <option value={-1} disabled>
              Choose a preset transaction…
            </option>
            {PRESET_TRANSACTIONS.map((t, i) => (
              <option key={i} value={i}>
                {t.label}
              </option>
            ))}
          </select>
        </div>

        {/* Transaction Summary */}
        {selected && (
          <div className="transaction-details" id="transaction-details">
            <div className="detail-item detail-item--full">
              <span className="detail-key">Description</span>
              <span className="detail-value">{selected.description}</span>
            </div>
            <div className="detail-item">
              <span className="detail-key">Amount</span>
              <span className="detail-value">
                ${selected.amount.toLocaleString('en-US', { minimumFractionDigits: 2 })}
              </span>
            </div>
            <div className="detail-item">
              <span className="detail-key">Features</span>
              <span className="detail-value">
                {Object.keys(selected.data).length} dimensions
              </span>
            </div>
          </div>
        )}

        {/* Predict Button */}
        <button
          id="predict-button"
          className={`btn-predict ${loading ? 'btn-predict--loading' : ''}`}
          disabled={!selected || loading}
          onClick={handlePredict}
        >
          {loading ? (
            <>
              <span className="spinner" />
              Analyzing…
            </>
          ) : (
            'Predict'
          )}
        </button>

        {/* Result */}
        {result && (
          <div className="result" id="prediction-result">
            <div className={`result__badge ${isFraud ? 'result__badge--fraud' : 'result__badge--legit'}`}>
              <span className="result__icon" aria-hidden="true">
                {isFraud ? '🚨' : '✅'}
              </span>
              <span className={`result__verdict ${isFraud ? 'result__verdict--fraud' : 'result__verdict--legit'}`}>
                {isFraud ? 'Fraud' : 'Legitimate'}
              </span>
            </div>

            <p className="result__confidence">
              Confidence: <strong>{confidencePercent}%</strong>
            </p>

            <div className="confidence-bar">
              <div
                className={`confidence-bar__fill ${isFraud ? 'confidence-bar__fill--fraud' : 'confidence-bar__fill--legit'}`}
                style={{ width: `${confidencePercent}%` }}
              />
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="error" id="prediction-error" role="alert">
            ⚠ {error}
          </div>
        )}
      </main>

      {/* ── Footer ── */}
      <footer className="footer">
        FraudLens <span className="footer__dot" /> Fraud Detection System
      </footer>
    </div>
  )
}

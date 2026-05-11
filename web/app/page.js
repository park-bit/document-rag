"use client";

import { useState, useEffect, useRef } from "react";

const API = typeof window !== "undefined" 
  ? `http://${window.location.hostname}:8000` 
  : "http://127.0.0.1:8000";

export default function Home() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("System Ready");
  
  // Separate loading states
  const [isUploading, setIsUploading] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [isExtracting, setIsExtracting] = useState(false);
  
  const [isIndexed, setIsIndexed] = useState(false);
  const [topK, setTopK] = useState(4);

  const [question, setQuestion] = useState("");
  const [queryAnswer, setQueryAnswer] = useState(null);

  const [certificateData, setCertificateData] = useState(null);
  const [filledForm, setFilledForm] = useState(null);
  const [formFields, setFormFields] = useState("document_title, author, key_topics, main_conclusion");

  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  // Reset indexed state when file changes
  useEffect(() => {
    if (file) {
      setIsIndexed(false);
      setStatus("Document Selected");
    }
  }, [file]);

  async function uploadFile() {
    if (!file) return;
    setIsUploading(true);
    setStatus("Neural Indexing...");
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch(`${API}/upload`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setIsIndexed(true);
      setStatus(`Indexed: ${data.num_chunks} Neural Chunks`);
    } catch (e) {
      setError(e.message);
      setStatus("Engine Error");
    } finally {
      setIsUploading(false);
    }
  }

  async function askQuestion() {
    if (!question.trim()) return;
    setIsQuerying(true);
    setStatus("Generating Insights...");
    try {
      const res = await fetch(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: topK }),
      });
      const data = await res.json();
      setQueryAnswer(data);
      setStatus("Insights Generated");
    } catch (e) {
      setError(e.message);
    } finally {
      setIsQuerying(false);
    }
  }

  async function analyzeCertificate() {
    setIsScanning(true);
    setStatus("Neural Scanning...");
    try {
      const res = await fetch(`${API}/analyze/certificate`, { method: "POST" });
      const data = await res.json();
      setCertificateData(data.parsed);
      setStatus("Analysis Complete");
    } catch (e) {
      setError(e.message);
    } finally {
      setIsScanning(false);
    }
  }

  async function runFormFill() {
    setIsExtracting(true);
    setStatus("Extracting Entities...");
    try {
      const fields = formFields.split(",").map(f => f.trim()).filter(Boolean);
      const res = await fetch(`${API}/fill-form`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fields, top_k: topK }),
      });
      const data = await res.json();
      setFilledForm(data.result);
      setStatus("Extraction Complete");
    } catch (e) {
      setError(e.message);
    } finally {
      setIsExtracting(false);
    }
  }

  const isLoadingGlobal = isUploading || isQuerying || isScanning || isExtracting;

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "60px 24px" }}>
      <header style={{ marginBottom: 60 }} className="animate-fade">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
          <div>
            <h1 className="hero-text" style={{ fontSize: 48, lineHeight: 1.1, marginBottom: 12 }}>
              DocMeant
            </h1>
            <p style={{ color: "var(--text-muted)", fontSize: 16, maxWidth: 500 }}>
              Sophisticated document intelligence. Powered by local silicon.
            </p>
          </div>
          <div className="status-badge">
            {status}
          </div>
        </div>
      </header>

      <div style={{ display: "grid", gridTemplateColumns: "320px 1fr", gap: 32 }}>
        {/* Control Panel */}
        <aside className="animate-fade" style={{ animationDelay: "0.1s" }}>
          <div className="glass-card" style={{ marginBottom: 24 }}>
            <span className="label-text">Neural Source</span>
            <div 
              onClick={() => fileInputRef.current.click()}
              style={{ 
                border: "1px dashed var(--accent-secondary)", 
                borderRadius: 8, 
                padding: 30, 
                textAlign: "center",
                cursor: "pointer",
                background: file ? "rgba(255,255,255,0.02)" : "transparent"
              }}
            >
              <input 
                type="file" 
                ref={fileInputRef}
                onChange={(e) => setFile(e.target.files[0])} 
                style={{ display: "none" }}
              />
              <div style={{ fontSize: 13, fontWeight: 500, color: file ? "#fff" : "var(--text-muted)" }}>
                {file ? file.name : "Select Document"}
              </div>
            </div>
            
            <button 
              type="button"
              onClick={uploadFile} 
              disabled={isUploading || isIndexed || !file} 
              className="btn-primary"
              style={{ width: "100%", marginTop: 20 }}
            >
              {isUploading ? "Initializing..." : isIndexed ? "Neural Link Active" : "Process Document"}
            </button>
          </div>

          <div className="glass-card">
            <span className="label-text">Intelligence Config</span>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: 13, color: "var(--text-muted)" }}>Context Window (K)</span>
              <input 
                type="number" 
                value={topK} 
                onChange={(e) => setTopK(Number(e.target.value))} 
                className="input-field"
                style={{ width: 60, textAlign: "center", padding: "6px" }}
              />
            </div>
          </div>
        </aside>

        {/* Neural Ops */}
        <main style={{ display: "flex", flexDirection: "column", gap: 24 }}>
          {/* Smart Scan */}
          <div className="glass-card animate-fade" style={{ animationDelay: "0.2s" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
              <div>
                <h2 style={{ fontSize: 20, marginBottom: 4 }}>Neural Document Scan</h2>
                <p style={{ fontSize: 13, color: "var(--text-muted)" }}>Autonomous identification of document metadata.</p>
              </div>
              <button 
                type="button" 
                onClick={analyzeCertificate} 
                disabled={isScanning || !isIndexed} 
                className="btn-primary" 
                style={{ padding: "8px 20px" }}
              >
                {isScanning ? "Scanning..." : "Execute Scan"}
              </button>
            </div>

            {certificateData ? (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                {Object.entries(certificateData).map(([k, v]) => (
                  <div key={k} style={{ background: "#000", padding: "12px 16px", borderRadius: 8, border: "1px solid var(--glass-border)" }}>
                    <div className="label-text" style={{ marginBottom: 4, opacity: 0.6 }}>{k.replace(/_/g, ' ')}</div>
                    <div style={{ fontSize: 14, fontWeight: 600, color: v ? "#fff" : "var(--error)" }}>{v || "N/A"}</div>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ color: "var(--text-muted)", fontSize: 13, textAlign: "center", padding: "20px 0" }}>
                {!isIndexed ? "Index a document to enable neural scanning." : "Ready for autonomous scan."}
              </div>
            )}
          </div>

          {/* Q&A Section */}
          <div className="glass-card animate-fade" style={{ animationDelay: "0.3s" }}>
            <h2 style={{ fontSize: 20, marginBottom: 8 }}>Neural Query Console</h2>
            <p style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 20 }}>Direct natural language interface with the document brain.</p>
            <div style={{ display: "flex", gap: 12, marginBottom: 20 }}>
              <input 
                placeholder="Query the document..." 
                value={question} 
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    askQuestion();
                  }
                }}
                className="input-field"
              />
              <button type="button" onClick={askQuestion} disabled={isQuerying || !isIndexed} className="btn-primary" style={{ minWidth: 100 }}>
                {isQuerying ? "Thinking..." : "Query"}
              </button>
            </div>

            {queryAnswer && (
              <div className="animate-fade">
                <div style={{ background: "#000", padding: 20, borderRadius: 12, border: "1px solid var(--glass-border)", marginBottom: 20 }}>
                  <p style={{ fontSize: 15, lineHeight: 1.6, color: "#fff" }}>{queryAnswer.answer}</p>
                </div>
                <div>
                  <span className="label-text">Context Verification</span>
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    {queryAnswer.sources?.map((s, i) => (
                      <div key={i} style={{ fontSize: 12, color: "var(--text-muted)", padding: "10px 14px", background: "#000", borderRadius: 8, border: "1px solid var(--glass-border)" }}>
                        <span style={{ color: "#fff", fontWeight: 700, marginRight: 8 }}>P{s.page}</span> {s.excerpt}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Entity Extraction */}
          <div className="glass-card animate-fade" style={{ animationDelay: "0.4s" }}>
            <h2 style={{ fontSize: 20, marginBottom: 8 }}>Entity Target Extraction</h2>
            <p style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 20 }}>Specify target fields for deep neural extraction.</p>
            <div style={{ marginBottom: 20 }}>
              <input 
                value={formFields} 
                onChange={(e) => setFormFields(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    runFormFill();
                  }
                }}
                className="input-field"
              />
            </div>
            <button 
              type="button" 
              onClick={runFormFill} 
              disabled={isExtracting || !isIndexed} 
              className="btn-primary" 
              style={{ width: "100%", background: "var(--accent-secondary)", color: "#fff" }}
            >
              {isExtracting ? "Extracting..." : "Execute Target Extraction"}
            </button>

            {filledForm && (
              <div style={{ marginTop: 20, background: "#000", padding: 16, borderRadius: 8, border: "1px solid var(--glass-border)" }}>
                <pre style={{ color: "var(--success)", fontSize: 13, fontFamily: "monospace", overflowX: "auto" }}>
                  {JSON.stringify(filledForm, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </main>
      </div>

      <footer style={{ marginTop: 80, padding: "40px 0", borderTop: "1px solid var(--glass-border)", textAlign: "center" }}>
        <p style={{ color: "var(--text-muted)", fontSize: 12, letterSpacing: "0.05em" }}>
          Engine: PyMuPDF - FAISS - Local Neural VRAM - OCR
        </p>
      </footer>

      {error && (
        <div style={{ position: "fixed", bottom: 40, right: 40, background: "var(--error)", color: "white", padding: "12px 24px", borderRadius: 8, fontSize: 14, fontWeight: 600, boxShadow: "0 10px 30px rgba(239, 68, 68, 0.3)" }}>
          {error}
        </div>
      )}
    </div>
  );
}

"use client";

import { useState, useEffect } from "react";

const API = "http://127.0.0.1:8000";

export default function Home() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("Ready");
  const [loading, setLoading] = useState(false);
  const [topK, setTopK] = useState(4);

  const [question, setQuestion] = useState("");
  const [queryAnswer, setQueryAnswer] = useState(null);

  const [certificateData, setCertificateData] = useState(null);
  const [filledForm, setFilledForm] = useState(null);
  const [formFields, setFormFields] = useState("person_name, date_of_birth, certificate_type, issue_date");

  const [error, setError] = useState(null);

  async function uploadFile() {
    if (!file) return alert("Please select a file first.");
    setLoading(true);
    setStatus("Uploading & Indexing...");
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
      setStatus(`Success: Document indexed (${data.num_chunks} chunks)`);
    } catch (e) {
      setError(e.message);
      setStatus("Upload failed");
    } finally {
      setLoading(false);
    }
  }

  async function askQuestion() {
    if (!question.trim()) return;
    setLoading(true);
    setStatus("Analyzing...");
    try {
      const res = await fetch(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: topK }),
      });
      const data = await res.json();
      setQueryAnswer(data);
      setStatus("Analysis complete");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function analyzeCertificate() {
    setLoading(true);
    setStatus("Extracting certificate data...");
    try {
      const res = await fetch(`${API}/analyze/certificate`, { method: "POST" });
      const data = await res.json();
      setCertificateData(data.parsed);
      setStatus("Extraction complete");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function runFormFill() {
    setLoading(true);
    setStatus("Filling form...");
    try {
      const fields = formFields.split(",").map(f => f.trim()).filter(Boolean);
      const res = await fetch(`${API}/fill-form`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fields, top_k: topK }),
      });
      const data = await res.json();
      setFilledForm(data.result);
      setStatus("Form extraction complete");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: 1000, margin: "40px auto", padding: "0 20px" }}>
      <header style={{ marginBottom: 40, textAlign: "center" }}>
        <h1 style={{ fontSize: 42, fontWeight: 800, marginBottom: 8, background: "linear-gradient(to right, #3b82f6, #8b5cf6)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
          DocMind Local RAG
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: 18 }}>
          Secure, private, and local document intelligence.
        </p>
      </header>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: 24 }}>
        {/* Sidebar */}
        <aside>
          <div className="card">
            <label className="label">Step 1: Upload Document</label>
            <input 
              type="file" 
              onChange={(e) => setFile(e.target.files[0])} 
              style={{ marginBottom: 12, padding: 8 }}
            />
            <button onClick={uploadFile} disabled={loading || !file} style={{ width: "100%" }}>
              {loading && status.includes("Upload") ? "Processing..." : "Upload & Index"}
            </button>
            <div style={{ marginTop: 12, fontSize: 13, color: error ? "#f87171" : "var(--text-secondary)" }}>
              {status}
            </div>
          </div>

          <div className="card">
            <label className="label">Configuration</label>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <span style={{ fontSize: 14 }}>Top K Chunks:</span>
              <input 
                type="number" 
                value={topK} 
                onChange={(e) => setTopK(Number(e.target.value))} 
                style={{ width: 60, padding: "4px 8px" }}
              />
            </div>
          </div>
        </aside>

        {/* Main Content */}
        <main>
          {/* Certificate Analysis */}
          <div className="card animate-fade-in">
            <h2 style={{ marginBottom: 16, fontSize: 20 }}>Smart Extraction</h2>
            <div style={{ display: "flex", gap: 12, marginBottom: 20 }}>
              <button onClick={analyzeCertificate} disabled={loading} style={{ background: "#8b5cf6" }}>
                Auto-Analyze Certificate
              </button>
            </div>

            {certificateData && (
              <div className="grid" style={{ background: "rgba(255,255,255,0.03)", padding: 16, borderRadius: 8 }}>
                {Object.entries(certificateData).map(([k, v]) => (
                  <div key={k}>
                    <div className="label" style={{ marginBottom: 2 }}>{k.replace(/_/g, ' ')}</div>
                    <div style={{ fontSize: 14, fontWeight: 500 }}>{v || "N/A"}</div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Q&A Section */}
          <div className="card animate-fade-in" style={{ animationDelay: "0.1s" }}>
            <h2 style={{ marginBottom: 16, fontSize: 20 }}>Ask the Document</h2>
            <div style={{ display: "flex", gap: 12 }}>
              <input 
                placeholder="e.g. What is the issue date of this document?" 
                value={question} 
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && askQuestion()}
              />
              <button onClick={askQuestion} disabled={loading}>Ask</button>
            </div>

            {queryAnswer && (
              <div style={{ marginTop: 20 }}>
                <div style={{ background: "rgba(59, 130, 246, 0.1)", padding: 20, borderRadius: 12, borderLeft: "4px solid var(--accent-color)" }}>
                  <p style={{ fontSize: 16, color: "#fff" }}>{queryAnswer.answer}</p>
                </div>
                <div style={{ marginTop: 16 }}>
                  <h4 className="label">Sources</h4>
                  {queryAnswer.sources?.map((s, i) => (
                    <div key={i} style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 4, padding: "4px 8px", background: "rgba(255,255,255,0.02)", borderRadius: 4 }}>
                      Page {s.page}: ...{s.excerpt}...
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Custom Form Fill */}
          <div className="card animate-fade-in" style={{ animationDelay: "0.2s" }}>
            <h2 style={{ marginBottom: 16, fontSize: 20 }}>Custom Field Extraction</h2>
            <div style={{ marginBottom: 12 }}>
              <label className="label">Desired Fields (comma separated)</label>
              <input 
                value={formFields} 
                onChange={(e) => setFormFields(e.target.value)}
              />
            </div>
            <button onClick={runFormFill} disabled={loading} style={{ background: "#10b981" }}>
              Extract Fields
            </button>

            {filledForm && (
              <pre style={{ marginTop: 16, background: "rgba(0,0,0,0.3)", padding: 12, borderRadius: 8, fontSize: 13, overflowX: "auto" }}>
                {JSON.stringify(filledForm, null, 2)}
              </pre>
            )}
          </div>
        </main>
      </div>

      {error && (
        <div style={{ position: "fixed", bottom: 20, right: 20, background: "#ef4444", color: "white", padding: "12px 24px", borderRadius: 8, boxShadow: "0 4px 12px rgba(0,0,0,0.3)" }}>
          {error}
        </div>
      )}
    </div>
  );
}

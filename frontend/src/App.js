import React from 'react';
import './App.css';
import ChunkingInterface from './ChunkingInterface';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>AWS LLM RAGA Project</h1>
        <p>Semantic Search and Document Analysis</p>
      </header>

      <main className="App-main">
        {/* Chunking Interface Section */}
        <section className="chunking-section">
          <ChunkingInterface />
        </section>
      </main>
    </div>
  );
}

export default App;

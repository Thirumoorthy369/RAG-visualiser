/* ================================================================
   RAG Workflow Visualizer — Engine
   All simulation logic, animations, cosine similarity, graph viz.
   ================================================================ */

// ═══════════════════════════════════════════════════════════════════
//  SIMULATED KNOWLEDGE BASE
// ═══════════════════════════════════════════════════════════════════

const DOCUMENTS = [
  {
    title: "TypeScript Overview",
    text: "TypeScript is a strongly typed programming language that builds on JavaScript, giving you better tooling at any scale. TypeScript adds optional static typing and class-based object-oriented programming to the language. It is designed for the development of large applications and transcompiles to JavaScript. TypeScript is developed and maintained by Microsoft. It was first made public in October 2012.",
    page: 1,
  },
  {
    title: "Generics in TypeScript",
    text: "Generics allow creating reusable components that work over a variety of types rather than a single one. Users can consume these components and use their own types. A generic function uses a type variable T that captures the type provided by the user. For example: function identity<T>(arg: T): T { return arg; }. Generics preserve type information, unlike using 'any', which loses type safety. They are widely used in arrays, promises, and utility types like Partial<T> and Record<K,V>.",
    page: 2,
  },
  {
    title: "Enums in TypeScript",
    text: "Enums allow a developer to define a set of named constants. TypeScript provides both numeric and string-based enums. Numeric enums auto-increment from an initial value: enum Direction { Up = 0, Down, Left, Right }. String enums require each member to be initialized with a string literal: enum Color { Red = 'RED', Green = 'GREEN', Blue = 'BLUE' }. Enums are compiled to JavaScript objects, making them available at runtime. Const enums are inlined at compile time for performance.",
    page: 3,
  },
  {
    title: "Interfaces in TypeScript",
    text: "Interfaces define the shape of an object in TypeScript. They are a powerful way to define contracts within your code. An interface can include properties, methods, and index signatures. Interfaces support optional properties using the ? modifier. They can be extended using the extends keyword: interface Animal { name: string } interface Dog extends Animal { breed: string }. Unlike type aliases, interfaces can be merged (declaration merging). Interfaces are erased at compile time and have zero runtime cost.",
    page: 4,
  },
  {
    title: "Strict Null Checks",
    text: "The strictNullChecks compiler option makes TypeScript treat null and undefined as distinct types. When enabled, you cannot assign null or undefined to a variable unless you explicitly include them in the type annotation: let x: string | null = null. This catches common bugs where developers forget to handle null values. The non-null assertion operator (!) tells the compiler a value is definitely not null: element!.textContent. Optional chaining (?.) and nullish coalescing (??) are related features that make null-safe code more concise.",
    page: 5,
  },
];

// Fake embedding dimension
const EMBED_DIM = 8;

// Predefined "embeddings" for each document (architecturally accurate simulation)
const DOC_EMBEDDINGS = [
  [ 0.82,  0.15, -0.33,  0.71, -0.12,  0.44,  0.09, -0.58],
  [ 0.11,  0.91, -0.07,  0.22,  0.65, -0.31,  0.77, -0.14],
  [-0.25,  0.08,  0.88, -0.19,  0.33,  0.72, -0.41,  0.16],
  [ 0.55, -0.22,  0.14,  0.84, -0.45,  0.07,  0.63, -0.37],
  [ 0.33,  0.47, -0.62,  0.11,  0.79, -0.18, -0.29,  0.86],
];

// Query → embedding mapping (keyed by keyword)
const QUERY_VECTORS = {
  typescript: [ 0.78,  0.20, -0.28,  0.66, -0.08,  0.39,  0.12, -0.52],
  generics:   [ 0.15,  0.88, -0.11,  0.18,  0.71, -0.27,  0.73, -0.10],
  enums:      [-0.20,  0.12,  0.85, -0.15,  0.29,  0.68, -0.38,  0.21],
  interface:  [ 0.52, -0.18,  0.18,  0.80, -0.41,  0.11,  0.59, -0.33],
  strict:     [ 0.29,  0.43, -0.58,  0.15,  0.75, -0.14, -0.25,  0.82],
  null:       [ 0.31,  0.45, -0.60,  0.13,  0.77, -0.16, -0.27,  0.84],
  default:    [ 0.45,  0.35, -0.22,  0.40,  0.30,  0.15,  0.25, -0.18],
};

// Simulated LLM answers
const ANSWERS = {
  typescript: "TypeScript is a strongly typed programming language developed by Microsoft that builds on JavaScript. It adds optional static typing, class-based object-oriented programming, and compiles (transcompiles) to plain JavaScript. It was designed for building large-scale applications and was first released in October 2012.\n\nKey benefits include:\n• **Static type checking** catches errors at compile time\n• **Better tooling** — IntelliSense, refactoring, navigation\n• **Scales** to large codebases with interfaces, generics, and modules\n• **Fully compatible** with existing JavaScript code",
  generics: "Generics in TypeScript allow you to create **reusable components** that work across multiple types while preserving full type safety.\n\nInstead of using `any` (which loses type info), generics use **type variables** like `T`:\n\n```\nfunction identity<T>(arg: T): T {\n  return arg;\n}\n```\n\nThe type `T` is captured when you call the function:\n• `identity<string>(\"hello\")` → returns `string`\n• `identity<number>(42)` → returns `number`\n\nGenerics are used extensively in utility types like `Partial<T>`, `Record<K,V>`, `Promise<T>`, and array types.",
  enums: "TypeScript enums define a **set of named constants**, available in two flavors:\n\n**Numeric enums** auto-increment from an initial value:\n```\nenum Direction { Up = 0, Down, Left, Right }\n```\n\n**String enums** require explicit initialization:\n```\nenum Color { Red = 'RED', Green = 'GREEN', Blue = 'BLUE' }\n```\n\nKey facts:\n• Compiled to JavaScript objects → available at **runtime**\n• `const enum` variants are **inlined** at compile time for better performance\n• Can be used in switch statements and as union-like types",
  interface: "Interfaces in TypeScript define the **shape** (contract) of an object:\n\n```\ninterface User {\n  name: string;\n  age?: number; // optional\n}\n```\n\nKey features:\n• **Optional properties** with `?` modifier\n• **Extends** other interfaces: `interface Dog extends Animal { breed: string }`\n• **Declaration merging** — same-name interfaces auto-merge\n• **Zero runtime cost** — erased during compilation\n\nInterfaces vs Type aliases:\n• Interfaces can be merged; types cannot\n• Interfaces are generally preferred for object shapes\n• Types are better for unions and primitives",
  strict: "**Strict null checks** (`strictNullChecks: true`) make TypeScript treat `null` and `undefined` as **distinct types**, preventing one of the most common categories of bugs.\n\nWithout it: any variable can be `null` silently\nWith it: you must explicitly opt in:\n```\nlet x: string | null = null; // OK\nlet y: string = null; // ❌ Error!\n```\n\nRelated tools:\n• **Non-null assertion** `!` → `element!.textContent`\n• **Optional chaining** `?.` → `obj?.prop?.nested`\n• **Nullish coalescing** `??` → `value ?? defaultValue`\n\nThis catches bugs at compile time instead of runtime crashes.",
};


// ═══════════════════════════════════════════════════════════════════
//  STATE
// ═══════════════════════════════════════════════════════════════════

let chunks = [];
let vectorStore = [];
let queryEmbedding = null;
let similarities = [];
let topResults = [];
let rerankedResults = [];
let currentAnswer = "";
let showTech = false;

// ═══════════════════════════════════════════════════════════════════
//  DOM REFS
// ═══════════════════════════════════════════════════════════════════

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);
const pipeline = $("#pipeline");
const fileInput = $("#fileInput");
const uploadZone = $("#uploadZone");
const btnIngest = $("#btnIngest");
const btnAsk = $("#btnAsk");
const queryInput = $("#queryInput");
const toggleTech = $("#toggleTech");
const toggleGraph = $("#toggleGraph");
const graphSection = $("#graphSection");
const graphCanvas = $("#graphCanvas");

// ═══════════════════════════════════════════════════════════════════
//  UTILITIES
// ═══════════════════════════════════════════════════════════════════

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function cosineSimilarity(a, b) {
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

function tokenCount(text) {
  return Math.ceil(text.split(/\s+/).length * 1.3);
}

function getQueryVector(query) {
  const q = query.toLowerCase();
  for (const [key, vec] of Object.entries(QUERY_VECTORS)) {
    if (q.includes(key)) return { keyword: key, vec };
  }
  return { keyword: "default", vec: QUERY_VECTORS.default };
}

function getAnswer(query) {
  const q = query.toLowerCase();
  if (q.includes("generic")) return ANSWERS.generics;
  if (q.includes("enum"))    return ANSWERS.enums;
  if (q.includes("interface")) return ANSWERS.interface;
  if (q.includes("strict") || q.includes("null")) return ANSWERS.strict;
  return ANSWERS.typescript;
}

function vecHTML(vec, showAll = false) {
  const vals = showAll ? vec : vec.slice(0, EMBED_DIM);
  return vals.map((v) => {
    const cls = v >= 0 ? "vec-pos" : "vec-neg";
    return `<span class="vec-val ${cls}">${v.toFixed(3)}</span>`;
  }).join(" ");
}

// ═══════════════════════════════════════════════════════════════════
//  STAGE RENDERER
// ═══════════════════════════════════════════════════════════════════

function addStage(num, icon, title, subtitle, color, bodyHTML) {
  // Arrow
  const arrow = document.createElement("div");
  arrow.className = "stage-arrow";
  arrow.innerHTML = "⬇";
  pipeline.appendChild(arrow);

  // Stage card
  const stage = document.createElement("div");
  stage.className = "stage";
  stage.innerHTML = `
    <div class="stage-card active" style="--stage-color:${color}">
      <div class="stage-header">
        <div class="stage-number" style="background:${color}">${num}</div>
        <div>
          <div class="stage-title">${icon} ${title}</div>
          <div class="stage-subtitle">${subtitle}</div>
        </div>
      </div>
      <div class="stage-progress"><div class="stage-progress-fill"></div></div>
      <div class="stage-body">${bodyHTML}</div>
    </div>
  `;
  pipeline.appendChild(stage);

  // Animate in
  requestAnimationFrame(() => {
    arrow.classList.add("visible");
    stage.classList.add("visible");
    stage.scrollIntoView({ behavior: "smooth", block: "center" });
  });

  return stage;
}

async function animateProgress(stage, durationMs = 1200) {
  const fill = stage.querySelector(".stage-progress-fill");
  if (!fill) return;
  fill.style.transition = `width ${durationMs}ms ease`;
  fill.style.width = "100%";
  await sleep(durationMs);
}

// ═══════════════════════════════════════════════════════════════════
//  1️⃣  DOCUMENT INGESTION FLOW
// ═══════════════════════════════════════════════════════════════════

async function runIngestion() {
  pipeline.innerHTML = "";
  chunks = [];
  vectorStore = [];
  btnIngest.disabled = true;
  btnIngest.innerHTML = '<span class="spinner"></span>Running …';

  // ── Stage 1: Data Collection ─────────────────────────────────
  const totalChars = DOCUMENTS.reduce((s, d) => s + d.text.length, 0);
  const s1body = `
    <div class="metrics">
      <div class="metric" style="--stage-color:#3b82f6">
        <div class="metric-value">${DOCUMENTS.length}</div>
        <div class="metric-label">Pages</div>
      </div>
      <div class="metric" style="--stage-color:#3b82f6">
        <div class="metric-value">${totalChars.toLocaleString()}</div>
        <div class="metric-label">Characters</div>
      </div>
      <div class="metric" style="--stage-color:#3b82f6">
        <div class="metric-value">${tokenCount(DOCUMENTS.map(d=>d.text).join(" "))}</div>
        <div class="metric-label">~Tokens</div>
      </div>
    </div>
    <div class="edu-tip">
      <span class="tip-icon">💡</span>
      <span>In production, a document loader (like PyPDF2, Unstructured, or LangChain loaders) extracts raw text from PDFs, DOCX, HTML, etc. The text must be cleaned and normalised before processing.</span>
    </div>
    <div class="preview-box">${DOCUMENTS[0].text.slice(0, 200)}…\n\n[Page 2] ${DOCUMENTS[1].text.slice(0, 120)}…\n\n… (${DOCUMENTS.length} pages total)</div>
  `;
  const s1 = addStage(1, "📄", "Data Collection", "Extract raw text from uploaded document", "#3b82f6", s1body);
  await animateProgress(s1, 1000);
  await sleep(400);

  // ── Stage 2: Chunking ────────────────────────────────────────
  DOCUMENTS.forEach((doc, i) => {
    chunks.push({
      id: `chunk_${i + 1}`,
      text: doc.text,
      tokenCount: tokenCount(doc.text),
      metadata: { page: doc.page, title: doc.title, source: "TypeScript Notes" },
      embedding: null,
    });
  });

  let chunkCards = chunks.map((c) => `
    <div class="chunk-card">
      <div class="chunk-id">${c.id}</div>
      <div class="chunk-tokens">📏 ${c.tokenCount} tokens · Page ${c.metadata.page}</div>
      <div class="chunk-text">${c.text.slice(0, 110)}…</div>
    </div>
  `).join("");

  const s2body = `
    <div class="metrics">
      <div class="metric" style="--stage-color:#8b5cf6">
        <div class="metric-value">${chunks.length}</div>
        <div class="metric-label">Chunks</div>
      </div>
      <div class="metric" style="--stage-color:#8b5cf6">
        <div class="metric-value">~${Math.round(chunks.reduce((s,c)=>s+c.tokenCount,0)/chunks.length)}</div>
        <div class="metric-label">Avg Tokens</div>
      </div>
      <div class="metric" style="--stage-color:#8b5cf6">
        <div class="metric-value">300</div>
        <div class="metric-label">Max Size</div>
      </div>
    </div>
    <div class="edu-tip">
      <span class="tip-icon">💡</span>
      <span><strong>Why chunking?</strong> LLMs have context windows (token limits). Large documents must be split into smaller, semantically meaningful chunks. Overlap between chunks ensures no information is lost at boundaries. Common strategies: fixed-size, sentence-based, or recursive splitting.</span>
    </div>
    <div class="chunk-grid">${chunkCards}</div>
  `;
  const s2 = addStage(2, "✂️", "Preprocessing & Chunking", "Split document into semantic chunks", "#8b5cf6", s2body);
  await animateProgress(s2, 1200);
  await sleep(400);

  // ── Stage 3: Embedding Generation ────────────────────────────
  chunks.forEach((c, i) => {
    c.embedding = DOC_EMBEDDINGS[i];
  });

  let vecRows = chunks.map((c) => `
    <div class="vector-row">
      <span class="vec-label">${c.id}:</span>
      ${vecHTML(c.embedding)}
    </div>
  `).join("");

  const s3body = `
    <div class="metrics">
      <div class="metric" style="--stage-color:#f59e0b">
        <div class="metric-value">${chunks.length}</div>
        <div class="metric-label">Vectors</div>
      </div>
      <div class="metric" style="--stage-color:#f59e0b">
        <div class="metric-value">${EMBED_DIM}</div>
        <div class="metric-label">Dimensions</div>
      </div>
      <div class="metric" style="--stage-color:#f59e0b">
        <div class="metric-value">float32</div>
        <div class="metric-label">Dtype</div>
      </div>
    </div>
    <div class="edu-tip">
      <span class="tip-icon">💡</span>
      <span><strong>How embeddings work:</strong> An embedding model (e.g. all-MiniLM-L6-v2 with 384 dims, or OpenAI text-embedding-3-small with 1536 dims) converts each text chunk into a dense numeric vector. Texts with similar meaning will have vectors that are close together in this high-dimensional space. This is what enables semantic search.</span>
    </div>
    <p style="font-size:.85rem;color:var(--text2);margin:8px 0">Each chunk → ${EMBED_DIM}-dimensional vector (real models use 384–1536 dims)</p>
    <div class="tech-detail">${vecRows}</div>
  `;
  const s3 = addStage(3, "🧮", "Embedding Generation", "Convert text chunks into vector representations", "#f59e0b", s3body);
  await animateProgress(s3, 1400);
  await sleep(400);

  // ── Stage 4: Vector Database ─────────────────────────────────
  vectorStore = chunks.map((c) => ({ ...c }));

  const s4body = `
    <div class="metrics">
      <div class="metric" style="--stage-color:#10b981">
        <div class="metric-value">${vectorStore.length}</div>
        <div class="metric-label">Vectors Stored</div>
      </div>
      <div class="metric" style="--stage-color:#10b981">
        <div class="metric-value">Indexed</div>
        <div class="metric-label">Status</div>
      </div>
      <div class="metric" style="--stage-color:#10b981">
        <div class="metric-value">Flat IP</div>
        <div class="metric-label">Index Type</div>
      </div>
    </div>
    <div class="edu-tip">
      <span class="tip-icon">💡</span>
      <span><strong>Vector databases</strong> (FAISS, Pinecone, Weaviate, ChromaDB) store embeddings and support fast similarity search. IndexFlatIP uses inner product on normalised vectors — equivalent to cosine similarity. Production systems use approximate methods (HNSW, IVF) for millions of vectors.</span>
    </div>
    <p style="color:var(--green);font-weight:600;margin-top:8px">✅ All ${vectorStore.length} chunks indexed and ready for retrieval!</p>
  `;
  const s4 = addStage(4, "📦", "Vector Database", "Store and index embeddings for fast retrieval", "#10b981", s4body);
  await animateProgress(s4, 800);

  // Done
  btnIngest.innerHTML = "✅ Ingestion Complete";
  btnIngest.disabled = false;
  queryInput.disabled = false;
  btnAsk.disabled = false;
  $$(".sq").forEach((b) => (b.disabled = false));
}


// ═══════════════════════════════════════════════════════════════════
//  2️⃣  QUESTION ANSWERING FLOW
// ═══════════════════════════════════════════════════════════════════

async function runQA(query) {
  if (!vectorStore.length) return;

  // Remove any previous QA stages (keep ingestion stages 1-4)
  const allStages = pipeline.querySelectorAll(".stage, .stage-arrow");
  let ingestionCount = 0;
  allStages.forEach((el) => {
    // Count stage cards (not arrows)
    if (el.classList.contains("stage")) ingestionCount++;
    // Remove everything after the 4th stage
    if (ingestionCount > 4) el.remove();
    // Also remove arrows after 4th stage
    if (ingestionCount > 4 && el.classList.contains("stage-arrow")) el.remove();
  });
  // Clean approach: remove all QA stages
  pipeline.querySelectorAll("[data-qa]").forEach((el) => el.remove());

  btnAsk.disabled = true;
  btnAsk.innerHTML = '<span class="spinner"></span>Thinking …';

  // ── Stage 5: Query Embedding ─────────────────────────────────
  const { keyword, vec: qVec } = getQueryVector(query);
  queryEmbedding = qVec;

  const s5body = `
    <p style="font-size:.92rem;margin-bottom:10px"><strong>Query:</strong> "${query}"</p>
    <div class="metrics">
      <div class="metric" style="--stage-color:#ef4444">
        <div class="metric-value">${EMBED_DIM}D</div>
        <div class="metric-label">Vector Dims</div>
      </div>
      <div class="metric" style="--stage-color:#ef4444">
        <div class="metric-value">${keyword}</div>
        <div class="metric-label">Detected Topic</div>
      </div>
    </div>
    <div class="edu-tip">
      <span class="tip-icon">💡</span>
      <span>The same embedding model used for chunks is used for the query. This ensures both live in the <strong>same vector space</strong>, making similarity comparison meaningful.</span>
    </div>
    <div class="tech-detail">
      <div class="vector-row">
        <span class="vec-label">Query:</span>
        ${vecHTML(qVec)}
      </div>
    </div>
  `;
  const s5 = addStage(5, "🔍", "Query Embedding", "Convert question into the same vector space", "#ef4444", s5body);
  s5.setAttribute("data-qa", "1");
  await animateProgress(s5, 800);
  await sleep(300);

  // ── Stage 6: Top-K Retrieval ─────────────────────────────────
  similarities = vectorStore.map((chunk) => ({
    ...chunk,
    similarity: cosineSimilarity(qVec, chunk.embedding),
  }));
  similarities.sort((a, b) => b.similarity - a.similarity);
  topResults = similarities.slice(0, 3);

  const simRows = similarities.map((s, i) => {
    const pct = Math.max(5, Math.round(s.similarity * 100));
    const isTop = i < 3;
    return `
      <div class="sim-row ${isTop ? "top" : ""}">
        <span class="sim-rank">#${i + 1}</span>
        <span class="sim-label">${s.metadata.title}</span>
        <div class="sim-bar-track">
          <div class="sim-bar-fill" style="width:${pct}%"></div>
        </div>
        <span class="sim-score">${s.similarity.toFixed(4)}</span>
      </div>
    `;
  }).join("");

  const mathDetail = `
    <div class="tech-detail">
      <p style="font-size:.82rem;color:var(--text2);margin:8px 0"><strong>Cosine Similarity Formula:</strong></p>
      <div class="preview-box">cos(A, B) = (A · B) / (|A| × |B|)

${similarities.map((s) => `${s.id}: cos = ${s.similarity.toFixed(6)}`).join("\n")}</div>
    </div>
  `;

  const s6body = `
    <div class="metrics">
      <div class="metric" style="--stage-color:#ef4444">
        <div class="metric-value">${vectorStore.length}</div>
        <div class="metric-label">Vectors Searched</div>
      </div>
      <div class="metric" style="--stage-color:#ef4444">
        <div class="metric-value">Top 3</div>
        <div class="metric-label">Retrieved</div>
      </div>
      <div class="metric" style="--stage-color:#ef4444">
        <div class="metric-value">${topResults[0].similarity.toFixed(3)}</div>
        <div class="metric-label">Best Score</div>
      </div>
    </div>
    <div class="edu-tip">
      <span class="tip-icon">💡</span>
      <span><strong>Semantic search:</strong> Unlike keyword search, vector search finds passages by <em>meaning</em>. A query about "generic types" matches chunks about generics even if the exact words differ. Cosine similarity measures the angle between vectors — 1.0 = identical, 0 = unrelated.</span>
    </div>
    ${simRows}
    ${mathDetail}
  `;
  const s6 = addStage(6, "🔍", "Top-K Retrieval", "Find most similar chunks via cosine similarity", "#ef4444", s6body);
  s6.setAttribute("data-qa", "1");
  await animateProgress(s6, 1000);
  await sleep(400);

  // ── Stage 7: Reranking ───────────────────────────────────────
  // Simulate cross-encoder scores (boost the most relevant, shuffle slightly)
  rerankedResults = topResults.map((r, i) => ({
    ...r,
    crossEncoderScore: r.similarity * (1 + (Math.random() * 0.15 - 0.03)),
    originalRank: i + 1,
  }));
  rerankedResults.sort((a, b) => b.crossEncoderScore - a.crossEncoderScore);

  const rerankRows = rerankedResults.map((r, i) => {
    const pct = Math.max(5, Math.round(r.crossEncoderScore * 100));
    return `
      <div class="sim-row top">
        <span class="sim-rank">#${i + 1}</span>
        <span class="sim-label">${r.metadata.title} <span style="color:var(--text3);font-size:.7rem">(was #${r.originalRank})</span></span>
        <div class="sim-bar-track">
          <div class="sim-bar-fill" style="width:${pct}%"></div>
        </div>
        <span class="sim-score">${r.crossEncoderScore.toFixed(4)}</span>
      </div>
    `;
  }).join("");

  const s7body = `
    <div class="edu-tip">
      <span class="tip-icon">💡</span>
      <span><strong>Why reranking?</strong> Bi-encoder retrieval (Stage 6) is fast but coarse — it embeds query and chunks separately. A <strong>cross-encoder</strong> reads the (query, chunk) pair <em>together</em> through a transformer, producing a much more accurate relevance score. This is slower but used on the small Top-K set for precision.</span>
    </div>
    <p style="font-size:.85rem;color:var(--text2);margin:8px 0">Cross-encoder re-scoring Top-3 results:</p>
    ${rerankRows}
  `;
  const s7 = addStage(7, "🏆", "Reranking", "Cross-encoder refines retrieval precision", "#ec4899", s7body);
  s7.setAttribute("data-qa", "1");
  await animateProgress(s7, 900);
  await sleep(300);

  // ── Stage 8: Prompt Construction ─────────────────────────────
  const contextBlock = rerankedResults.map((r, i) =>
    `[${i + 1}] (source: ${r.metadata.title}, page: ${r.metadata.page})\n${r.text}`
  ).join("\n\n");

  const fullPrompt =
    `SYSTEM:\nYou are a helpful assistant. Answer using ONLY the context below. Cite sources with [1], [2], etc.\n\n` +
    `CONTEXT:\n${contextBlock}\n\n` +
    `QUESTION:\n${query}\n\nANSWER:`;

  const promptTokens = tokenCount(fullPrompt);

  const s8body = `
    <div class="metrics">
      <div class="metric" style="--stage-color:#6366f1">
        <div class="metric-value">${rerankedResults.length}</div>
        <div class="metric-label">Context Chunks</div>
      </div>
      <div class="metric" style="--stage-color:#6366f1">
        <div class="metric-value">~${promptTokens}</div>
        <div class="metric-label">Prompt Tokens</div>
      </div>
    </div>
    <div class="edu-tip">
      <span class="tip-icon">💡</span>
      <span><strong>Prompt engineering for RAG:</strong> The retrieved chunks are injected as "context" into the prompt. The system message instructs the LLM to answer <em>only</em> from this context and cite sources. This is how RAG <strong>reduces hallucination</strong> — the model is grounded in real documents.</span>
    </div>
    <div class="tech-detail">
      <div class="prompt-display"><span class="hl-system">SYSTEM:\nYou are a helpful assistant. Answer using ONLY the context below. Cite sources with [1], [2], etc.</span>\n\n<span class="hl-context">CONTEXT:\n${contextBlock.replace(/</g, "&lt;").replace(/>/g, "&gt;").slice(0, 600)}…</span>\n\n<span class="hl-question">QUESTION:\n${query}</span>\n\nANSWER:</div>
    </div>
  `;
  const s8 = addStage(8, "📝", "Prompt Construction", "Assemble system + context + question", "#6366f1", s8body);
  s8.setAttribute("data-qa", "1");
  await animateProgress(s8, 700);
  await sleep(300);

  // ── Stage 9: LLM Generation ──────────────────────────────────
  currentAnswer = getAnswer(query);

  const s9body = `
    <p style="font-size:.85rem;color:var(--text2);margin-bottom:10px">
      <span class="spinner"></span> Generating response…
    </p>
    <div class="typing-area cursor-blink" id="typingArea"></div>
  `;
  const s9 = addStage(9, "🤖", "LLM Answer Generation", "Model generates grounded response", "#22d3ee", s9body);
  s9.setAttribute("data-qa", "1");

  // Typing effect
  const typingArea = document.getElementById("typingArea");
  await typeText(typingArea, currentAnswer, 12);
  typingArea.classList.remove("cursor-blink");

  // Remove spinner
  const spinnerP = s9.querySelector(".stage-body > p");
  if (spinnerP) spinnerP.innerHTML = "✅ Response generated";

  await animateProgress(s9, 600);
  await sleep(300);

  // ── Stage 10: Evaluation & Feedback ──────────────────────────
  const confidence = Math.round(70 + Math.random() * 25);
  const relevance = Math.round(65 + Math.random() * 30);

  const s10body = `
    <div class="metrics">
      <div class="metric" style="--stage-color:#a3e635">
        <div class="metric-value">${confidence}%</div>
        <div class="metric-label">Confidence</div>
      </div>
      <div class="metric" style="--stage-color:#a3e635">
        <div class="metric-value">${relevance}%</div>
        <div class="metric-label">Relevance</div>
      </div>
      <div class="metric" style="--stage-color:#a3e635">
        <div class="metric-value">${rerankedResults.length}</div>
        <div class="metric-label">Sources Used</div>
      </div>
    </div>
    <div class="edu-tip">
      <span class="tip-icon">💡</span>
      <span><strong>Evaluation in production:</strong> Real RAG systems measure <em>faithfulness</em> (is the answer supported by context?), <em>relevance</em> (does it answer the question?), and <em>groundedness</em>. User feedback is logged and used to fine-tune the retrieval and generation pipeline over time.</span>
    </div>
    <div class="feedback-row">
      <span style="color:var(--text2);font-size:.85rem">Was this helpful?</span>
      <button class="fb-btn" onclick="this.classList.add('selected')">👍 Helpful</button>
      <button class="fb-btn" onclick="this.classList.add('selected')">👎 Not Helpful</button>
      <span class="fb-score" id="fbNote"></span>
    </div>
  `;
  const s10 = addStage(10, "✅", "Evaluation & Feedback", "Measure quality + collect user feedback", "#a3e635", s10body);
  s10.setAttribute("data-qa", "1");
  await animateProgress(s10, 500);

  // Done
  btnAsk.innerHTML = "❓ Ask Another Question";
  btnAsk.disabled = false;

  // Update graph if visible
  if (!graphSection.classList.contains("hidden")) {
    drawGraph();
  }
}


// ═══════════════════════════════════════════════════════════════════
//  TYPING EFFECT
// ═══════════════════════════════════════════════════════════════════

async function typeText(el, text, speed = 15) {
  el.textContent = "";
  for (let i = 0; i < text.length; i++) {
    el.textContent += text[i];
    if (i % 3 === 0) {
      await sleep(speed);
      el.scrollTop = el.scrollHeight;
    }
  }
  // Replace plain text with formatted markdown (bold, code, lists)
  el.innerHTML = text
    .replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, '<code style="background:var(--bg3);padding:1px 5px;border-radius:3px;font-family:var(--mono);font-size:.85em">$1</code>')
    .replace(/```\n?([\s\S]*?)```/g, '<pre style="background:var(--bg);padding:10px;border-radius:6px;margin:6px 0;font-family:var(--mono);font-size:.82rem;overflow-x:auto">$1</pre>')
    .replace(/^• /gm, "• ")
    .replace(/\n/g, "<br>");
}


// ═══════════════════════════════════════════════════════════════════
//  GRAPH VISUALIZATION
// ═══════════════════════════════════════════════════════════════════

function drawGraph() {
  const ctx = graphCanvas.getContext("2d");
  const W = graphCanvas.width;
  const H = graphCanvas.height;
  ctx.clearRect(0, 0, W, H);

  if (!queryEmbedding || !vectorStore.length) {
    ctx.fillStyle = "#64748b";
    ctx.font = "14px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Run both Ingestion and QA to see the graph", W / 2, H / 2);
    return;
  }

  const topIds = new Set(rerankedResults.map((r) => r.id));

  // Layout: query in center-left, chunks in arc on right
  const qx = 180, qy = H / 2;

  const nodes = vectorStore.map((chunk, i) => {
    const angle = -Math.PI / 3 + (Math.PI * 2 / 3) * (i / (vectorStore.length - 1 || 1));
    const radius = 200;
    return {
      x: W / 2 + 120 + Math.cos(angle) * radius,
      y: H / 2 + Math.sin(angle) * radius,
      chunk,
      sim: similarities.find((s) => s.id === chunk.id)?.similarity || 0,
      isTop: topIds.has(chunk.id),
    };
  });

  // Draw edges
  nodes.forEach((node) => {
    const thickness = Math.max(0.5, node.sim * 6);
    const alpha = Math.max(0.08, node.sim * 0.7);
    ctx.beginPath();
    ctx.moveTo(qx, qy);
    ctx.lineTo(node.x, node.y);
    ctx.strokeStyle = node.isTop
      ? `rgba(59,130,246,${alpha})`
      : `rgba(100,116,139,${alpha * 0.5})`;
    ctx.lineWidth = node.isTop ? thickness * 1.5 : thickness;
    ctx.stroke();

    // Similarity label on edge
    if (node.isTop) {
      const mx = (qx + node.x) / 2;
      const my = (qy + node.y) / 2;
      ctx.fillStyle = "#94a3b8";
      ctx.font = "11px 'JetBrains Mono', monospace";
      ctx.textAlign = "center";
      ctx.fillText(node.sim.toFixed(3), mx, my - 6);
    }
  });

  // Draw chunk nodes
  nodes.forEach((node) => {
    const r = node.isTop ? 28 : 18;
    ctx.beginPath();
    ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
    ctx.fillStyle = node.isTop ? "#3b82f6" : "#1e293b";
    ctx.fill();
    ctx.strokeStyle = node.isTop ? "#60a5fa" : "#334155";
    ctx.lineWidth = node.isTop ? 2 : 1;
    ctx.stroke();

    // Label
    ctx.fillStyle = "#e2e8f0";
    ctx.font = `${node.isTop ? 11 : 10}px Inter, sans-serif`;
    ctx.textAlign = "center";
    ctx.fillText(node.chunk.metadata.title.split(" ").slice(0, 2).join(" "), node.x, node.y + r + 16);
  });

  // Draw query node
  ctx.beginPath();
  ctx.arc(qx, qy, 34, 0, Math.PI * 2);
  ctx.fillStyle = "#0e7490";
  ctx.fill();
  ctx.strokeStyle = "#22d3ee";
  ctx.lineWidth = 3;
  ctx.stroke();

  ctx.fillStyle = "#fff";
  ctx.font = "bold 13px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Query", qx, qy + 5);
}


// ═══════════════════════════════════════════════════════════════════
//  EVENT LISTENERS
// ═══════════════════════════════════════════════════════════════════

// File upload
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) {
    uploadZone.classList.add("active");
    btnIngest.disabled = false;
    uploadZone.querySelector("span:not(.upload-hint)").textContent =
      `📄 ${fileInput.files[0].name} selected`;
  }
});

// Ingestion button
btnIngest.addEventListener("click", () => {
  runIngestion();
});

// Ask button
btnAsk.addEventListener("click", () => {
  const q = queryInput.value.trim();
  if (q) runQA(q);
});

// Enter key on query
queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    const q = queryInput.value.trim();
    if (q && vectorStore.length) runQA(q);
  }
});

// Suggested queries
$$(".sq").forEach((btn) => {
  btn.addEventListener("click", () => {
    queryInput.value = btn.dataset.q;
    if (vectorStore.length) runQA(btn.dataset.q);
  });
});

// Technical details toggle
toggleTech.addEventListener("change", () => {
  showTech = toggleTech.checked;
  document.body.classList.toggle("show-tech", showTech);
});

// Graph mode toggle
toggleGraph.addEventListener("change", () => {
  const on = toggleGraph.checked;
  graphSection.classList.toggle("hidden", !on);
  if (on) drawGraph();
});

// Feedback buttons (event delegation)
document.addEventListener("click", (e) => {
  if (e.target.classList.contains("fb-btn")) {
    // Deselect siblings
    e.target.parentElement.querySelectorAll(".fb-btn").forEach((b) => b.classList.remove("selected"));
    e.target.classList.add("selected");
    const note = e.target.closest(".feedback-row")?.querySelector("#fbNote");
    if (note) note.textContent = "✓ Feedback recorded — used for fine-tuning";
  }
});

/* Display recorded inference only. DOM text nodes keep model output literal. */
"use strict";

const gallery = document.querySelector("#gallery");
const status = document.querySelector("#status");
const summary = document.querySelector("#summary");
const state = { fixtures: null, results: null, runs: {}, variant: "clean", category: "all", language: "all" };

function element(tag, className, content) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content !== undefined) node.textContent = content;
  return node;
}

function percent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function metric(label, value, className = "") {
  const node = element("div");
  node.append(element("strong", className, value), document.createTextNode(label));
  return node;
}

function sourceLink(source) {
  const link = element("a", "", `${source.name} · ${source.set} #${source.collector_number}`);
  link.href = source.scryfall_url;
  return link;
}

function renderExample(sample, variant, record) {
  const article = element("article", "example");
  const heading = element("div", "example-header");
  const title = element("div");
  const h3 = element("h3", "", sample.title);
  h3.id = `sample-${sample.id}`;
  article.setAttribute("aria-labelledby", h3.id);
  const credit = element("p", "print-credit");
  credit.append(sourceLink(sample.source));
  title.append(h3, credit);
  heading.append(title, element("span", "category", sample.category));

  const comparison = element("div", "comparison");
  const picture = element("div");
  picture.append(element("p", "field-label", "01 / Input picture"));
  const container = element("div", "crop-container");
  const image = element("img", "crop");
  image.src = `assets/${variant.image}`;
  image.alt = `${variant.label} of ${sample.title.toLowerCase()} on ${sample.source.name}`;
  image.loading = "lazy";
  image.width = sample.crop_bounds_xyxy[2] - sample.crop_bounds_xyxy[0];
  image.height = sample.crop_bounds_xyxy[3] - sample.crop_bounds_xyxy[1];
  container.append(image);
  picture.append(container, element("p", "transform", variant.transform));

  const output = element("div");
  output.append(element("p", "field-label", "02 / Recognized text"));
  if (record) {
    const inference = record.inference;
    output.append(element("p", `observed${inference.text ? "" : " empty"}`, inference.text || "[No text recognized]"));
    const metrics = element("div", "metrics");
    metrics.append(
      metric("Model confidence", percent(inference.confidence)),
      metric("Character error rate", percent(record.character_error_rate), record.exact_match ? "good" : "error"),
      metric("Reference match", record.exact_match ? "Exact" : `${record.character_edits} edit${record.character_edits === 1 ? "" : "s"}`, record.exact_match ? "good" : "error")
    );
    output.append(metrics, element("p", "chosen-method", `Method: ${inference.method}`));
  } else {
    output.append(element("p", "no-result", "No recorded inference is available for this crop."));
  }
  const truth = element("p", "truth");
  truth.append(element("span", "", "Printed reference"), document.createTextNode(sample.ground_truth));
  output.append(truth);
  comparison.append(picture, output);
  article.append(heading, comparison);

  const details = element("details");
  details.append(element("summary", "", "Method, candidates & provenance"));
  const content = element("div", "details-content");
  content.append(element("p", "", `Language: ${sample.language}; font / treatment: ${sample.font}`));
  content.append(element("p", "", `Reference: ${sample.transcription}`));
  content.append(element("p", "", `Original scan crop [left, top, right, bottom]: ${sample.crop_bounds_xyxy.join(", ")}`));
  if (record) {
    content.append(element("p", "", `Actual method: ${record.inference.method}`));
    content.append(element("p", "", state.results.confidence_note));
    content.append(element("p", "", "Complete recorded inference (including preprocessing candidates):"));
    content.append(element("pre", "", JSON.stringify(record.inference, null, 2)));
  }
  const source = element("p");
  source.append(document.createTextNode(`Scan artist: ${sample.source.artist}. `), sourceLink(sample.source));
  content.append(source);
  content.append(element("p", "", `Source image SHA-256: ${sample.source.source_image_sha256}`));
  details.append(content);
  article.append(details);
  return article;
}

function render() {
  if (!state.fixtures) return;
  const records = new Map((state.results?.records || []).map(record => [`${record.sample_id}/${record.variant_id}`, record]));
  const samples = state.fixtures.samples.filter(sample =>
    (state.category === "all" || sample.category === state.category) &&
    (state.language === "all" || sample.language === state.language)
  );
  gallery.replaceChildren(...samples.map(sample => {
    const variant = sample.variants.find(item => item.id === state.variant);
    return renderExample(sample, variant, records.get(`${sample.id}/${variant.id}`));
  }));
  const selected = samples.map(sample => records.get(`${sample.id}/${state.variant}`)).filter(Boolean);
  if (!selected.length) {
    summary.textContent = `${samples.length} examples · inference not recorded`;
    return;
  }
  const edits = selected.reduce((sum, item) => sum + item.character_edits, 0);
  const characters = selected.reduce((sum, item) => sum + item.reference_characters, 0);
  const exact = selected.filter(item => item.exact_match).length;
  summary.textContent = `${selected.length} examples · ${exact} exact · ${percent(edits / Math.max(characters, 1))} CER`;
}

for (const input of document.querySelectorAll("input[type=radio]")) {
  input.addEventListener("change", event => {
    if (event.target.name === "model") {
      state.results = state.runs[event.target.value];
      updateRunDetails();
    } else {
      state[event.target.name] = event.target.value;
    }
    render();
  });
}

document.querySelector("#language-filter").addEventListener("change", event => {
  state.language = event.target.value;
  render();
});

function initializeLanguages() {
  const select = document.querySelector("#language-filter");
  select.replaceChildren();
  const languages = [...new Set(state.fixtures.samples.map(sample => sample.language))].sort();
  const displayNames = typeof Intl.DisplayNames === "function" ? new Intl.DisplayNames(["en"], { type: "language" }) : null;
  if (languages.length > 1) {
    const all = element("option", "", "All available");
    all.value = "all";
    select.append(all);
  }
  for (const language of languages) {
    const option = element("option", "", `${displayNames?.of(language) || language} (${language})`);
    option.value = language;
    select.append(option);
  }
  state.language = select.value;
  document.querySelector("#coverage-note").textContent = state.fixtures.language_note;
}

async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json();
}

function updateRunDetails() {
  const modelName = state.results.records[0]?.inference.model.name || "unnamed model";
  document.querySelector("#checkpoint").textContent = `Recorded checkpoint: ${modelName}`;
  document.querySelector("#metric-note").textContent = state.results.metric_note;
  const details = document.querySelector("#run-details");
  details.replaceChildren();
  const date = new Date(state.results.generated_at);
  details.append(document.createTextNode(`Recorded ${Number.isNaN(date.valueOf()) ? state.results.generated_at : date.toLocaleDateString("en-GB", { year: "numeric", month: "short", day: "numeric", timeZone: "UTC" })} · passes: ${state.results.passes}. Recognition model SHA-256: `));
  details.append(element("code", "", state.results.recognition_model_sha256));
}

async function initialize() {
  const [fixtures, results, baseline] = await Promise.allSettled([
    loadJson("assets/fixtures.json"), loadJson("assets/results.json"), loadJson("assets/results-baseline.json")
  ]);
  if (fixtures.status === "rejected") {
    status.textContent = "The crop manifest could not be loaded. Serve this directory over HTTP or open the raw JSON links below.";
    summary.textContent = "Gallery unavailable";
    return;
  }
  state.fixtures = fixtures.value;
  initializeLanguages();
  if (results.status === "fulfilled") {
    state.results = results.value;
    state.runs.mtg = results.value;
    if (baseline.status === "fulfilled") {
      state.runs.base = baseline.value;
      document.querySelector("#model-controls").hidden = baseline.value.recognition_model_sha256 === results.value.recognition_model_sha256;
    }
    updateRunDetails();
  } else {
    status.textContent = "Predictions have not been recorded for this build. The real input crops and printed references are shown; run scripts/gallery_export.py to produce inference results.";
  }
  render();
}

initialize();

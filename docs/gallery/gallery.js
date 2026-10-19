/* Display recorded inference only. DOM text nodes keep model output literal. */
"use strict";

const gallery = document.querySelector("#gallery");
const status = document.querySelector("#status");
const summary = document.querySelector("#summary");
const state = { catalog: null, samples: [], runs: new Map(), model: "paddle", variant: "clean", category: "all", language: "en" };

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

function runFor(sample) {
  return state.runs.get(`${sample.bundle}/${state.model}`);
}

function renderExample(sample, variant, record, run) {
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
  image.src = new URL(variant.image, sample.manifestUrl).href;
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
    output.append(element("p", "no-result", "This model has no recorded run for this language."));
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
    content.append(element("p", "", run.data.confidence_note));
    content.append(element("p", "", "Complete recorded inference (including preprocessing candidates):"));
    content.append(element("pre", "", JSON.stringify(record.inference, null, 2)));
  }
  const source = element("p");
  if (sample.source.artist) source.append(document.createTextNode(`Scan artist: ${sample.source.artist}. `));
  source.append(sourceLink(sample.source));
  content.append(source);
  content.append(element("p", "", `Source image SHA-256: ${sample.source.source_image_sha256}`));
  details.append(content);
  article.append(details);
  return article;
}

function languageSamples() {
  return state.samples.filter(sample => state.language === "all" || sample.language === state.language);
}

function updateModelAvailability() {
  const samples = languageSamples();
  const inputs = [...document.querySelectorAll("input[name=model]")];
  for (const input of inputs) {
    input.disabled = !samples.some(sample => state.runs.has(`${sample.bundle}/${input.value}`));
  }
  const current = inputs.find(input => input.value === state.model);
  if (!current || current.disabled) {
    const available = inputs.find(input => !input.disabled);
    if (available) state.model = available.value;
  }
  for (const input of inputs) input.checked = input.value === state.model;
}

function updateRunDetails(samples) {
  const active = [...new Map(samples.map(sample => [`${sample.bundle}/${state.model}`, runFor(sample)]).filter(([, run]) => run)).values()];
  const names = [...new Set(active.map(run => run.data.records[0]?.inference.model.name || "unnamed model"))];
  document.querySelector("#checkpoint").textContent = names.length ? `Recorded checkpoint: ${names.join("; ")}` : "No recorded checkpoint for these filters";
  const details = document.querySelector("#run-details");
  details.replaceChildren();
  if (!active.length) return;
  document.querySelector("#metric-note").textContent = active[0].data.metric_note;
  const weights = [...new Set(active.map(run => run.data.recognition_model_sha256))];
  details.append(document.createTextNode("Recognition model SHA-256: "), element("code", "", weights.join("; ")));
  for (const run of active) {
    const link = element("a", "", `${run.label} · recorded ${run.data.generated_at.slice(0, 10)} · ${run.data.passes} passes`);
    link.href = run.resultsPath;
    details.append(element("br"), link);
  }
}

function render() {
  if (!state.catalog) return;
  updateModelAvailability();
  const samples = languageSamples().filter(sample => state.category === "all" || sample.category === state.category);
  const selected = [];
  gallery.replaceChildren(...samples.map(sample => {
    const variant = sample.variants.find(item => item.id === state.variant);
    const run = runFor(sample);
    const record = run?.records.get(`${sample.id}/${variant.id}`);
    if (record) selected.push(record);
    return renderExample(sample, variant, record, run);
  }));
  updateRunDetails(samples);
  if (!samples.length) {
    summary.textContent = "No examples with these filters";
  } else if (!selected.length) {
    summary.textContent = `${samples.length} examples · inference not recorded`;
  } else {
    const edits = selected.reduce((sum, item) => sum + item.character_edits, 0);
    const characters = selected.reduce((sum, item) => sum + item.reference_characters, 0);
    const exact = selected.filter(item => item.exact_match).length;
    const coverage = selected.length === samples.length ? `${selected.length} examples` : `${selected.length}/${samples.length} recorded`;
    summary.textContent = `${coverage} · ${exact} exact · ${percent(edits / Math.max(characters, 1))} CER`;
  }
}

for (const input of document.querySelectorAll("input[type=radio]")) {
  input.addEventListener("change", event => {
    state[event.target.name] = event.target.value;
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
  const languages = [...new Set(state.samples.map(sample => sample.language))].sort();
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
  select.value = languages.includes(state.catalog.default_language) ? state.catalog.default_language : languages[0];
  state.language = select.value;
  document.querySelector("#coverage-note").textContent = state.catalog.language_note;
}

async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json();
}

async function initialize() {
  try {
    state.catalog = await loadJson("assets/catalog.json");
    state.model = state.catalog.default_model;
    const bundles = await Promise.allSettled(state.catalog.bundles.map(async bundle => {
      const files = await Promise.allSettled([loadJson(bundle.fixtures), ...bundle.runs.map(run => loadJson(run.results))]);
      if (files[0].status !== "fulfilled") throw new Error(`Could not load ${bundle.fixtures}`);
      const manifestUrl = new URL(bundle.fixtures, document.baseURI).href;
      state.samples.push(...files[0].value.samples.map(sample => ({ ...sample, bundle: bundle.id, manifestUrl })));
      for (const [index, run] of bundle.runs.entries()) {
        const result = files[index + 1];
        if (result.status !== "fulfilled") throw new Error(`Could not load ${run.results}`);
        state.runs.set(`${bundle.id}/${run.model}`, { label: bundle.label, resultsPath: run.results,
          data: result.value, records: new Map(result.value.records.map(record => [`${record.sample_id}/${record.variant_id}`, record])) });
      }
    }));
    const failure = bundles.find(bundle => bundle.status === "rejected");
    if (failure) status.textContent = "Some recorded examples could not be loaded. The raw artifact links below remain available.";
    initializeLanguages();
    document.querySelector("#model-controls").hidden = false;
    render();
  } catch (error) {
    status.textContent = "The gallery could not be loaded. Serve this directory over HTTP or open the raw JSON links below.";
    summary.textContent = "Gallery unavailable";
  }
}

initialize();

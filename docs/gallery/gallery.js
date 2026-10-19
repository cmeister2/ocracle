/* Display literal, recorded inference. No model runs in the browser. */
"use strict";

const sampleIds = [
  "historic-flavor", "modern-flavor", "italic-reminder", "mixed-reminder",
  "goudy-title", "matrix-title", "beleren-title", "latin-fr-m21-182"
];
const gallery = document.querySelector("#gallery");
const status = document.querySelector("#status");

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function percent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function link(text, href) {
  const node = element("a", "", text);
  node.href = href;
  return node;
}

function renderVariant(sample, variant, record) {
  const section = element("section", "variant");
  section.dataset.variant = variant.id;
  const label = element("h3", "variant-label", variant.id === "clean" ? "Original" : "Severe blur");
  label.id = `${sample.id}-${variant.id}`;
  section.setAttribute("aria-labelledby", label.id);
  const image = element("img", "crop");
  image.src = new URL(variant.image, sample.manifestUrl).href;
  image.alt = `${variant.label}: ${sample.title.toLowerCase()} on ${sample.source.name}`;
  // Both variants share the native crop's display dimensions for comparison.
  // The blurred file is still the exact recorded 30% resolution input.
  image.width = sample.crop_bounds_xyxy[2] - sample.crop_bounds_xyxy[0];
  image.height = sample.crop_bounds_xyxy[3] - sample.crop_bounds_xyxy[1];
  const inference = record.inference;
  const observed = element("p", "observed", inference.text || "[No text recognized]");
  const error = record.exact_match ? "0 edits" : `${record.character_edits} edit${record.character_edits === 1 ? "" : "s"}, ${percent(record.character_error_rate)} CER`;
  const meta = element("p", "result-meta", `Score ${inference.confidence.toFixed(3)} · ${error} · ${inference.method}`);
  section.append(label, image, observed, meta);
  if (!record.exact_match) {
    const reference = element("p", "reference");
    reference.append(element("span", "", "Printed: "), document.createTextNode(sample.ground_truth));
    section.append(reference);
  }
  return section;
}

function renderExample(sample, run) {
  const article = element("article", "example");
  article.dataset.sampleId = sample.id;
  const heading = element("h2", "", sample.title);
  heading.id = `sample-${sample.id}`;
  article.setAttribute("aria-labelledby", heading.id);
  const variants = element("div", "variants");
  const records = {};
  for (const id of ["clean", "blur"]) {
    const variant = sample.variants.find(item => item.id === id);
    const record = run.records.get(`${sample.id}/${id}`);
    if (!variant || !record) throw new Error(`Missing recorded example: ${sample.id}/${id}`);
    records[id] = record;
    variants.append(renderVariant(sample, variant, record));
  }
  const details = element("details", "provenance");
  details.append(element("summary", "", "Source and recorded results"));
  const source = element("p");
  source.append(link(`${sample.source.name} · ${sample.source.set} #${sample.source.collector_number}`, sample.source.scryfall_url));
  details.append(source,
    element("p", "", `Language: ${sample.language}. Font/treatment: ${sample.font}.`),
    element("p", "", `Printed reference: ${sample.ground_truth}`),
    element("p", "", sample.transcription),
    element("p", "", `Native crop [left, top, right, bottom]: ${sample.crop_bounds_xyxy.join(", ")}.`),
    element("p", "", `Source image SHA-256: ${sample.source.source_image_sha256}`),
    element("p", "", `Recognition model SHA-256: ${run.data.recognition_model_sha256}`),
    element("p", "", run.data.confidence_note));
  for (const variant of sample.variants) {
    details.append(element("p", "", `${variant.label}: ${variant.transform}`));
  }
  const raw = element("p");
  raw.append(link("Raw inference JSON", run.path));
  details.append(raw, element("pre", "", JSON.stringify(records, null, 2)));
  article.append(heading, variants, details);
  return article;
}

async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json();
}

async function initialize() {
  try {
    const catalog = await loadJson("assets/catalog.json");
    const bundles = await Promise.all(catalog.bundles.map(async bundle => {
      const recorded = bundle.runs.find(run => run.model === catalog.default_model);
      if (!recorded) throw new Error(`Missing default-model results: ${bundle.id}`);
      const [fixtures, data] = await Promise.all([loadJson(bundle.fixtures), loadJson(recorded.results)]);
      const manifestUrl = new URL(bundle.fixtures, document.baseURI).href;
      return {
        samples: fixtures.samples.map(sample => ({ ...sample, manifestUrl })),
        run: { path: recorded.results, data, records: new Map(data.records.map(record => [`${record.sample_id}/${record.variant_id}`, record])) }
      };
    }));
    const examples = sampleIds.map(id => {
      const bundle = bundles.find(item => item.samples.some(sample => sample.id === id));
      if (!bundle) throw new Error(`Missing sample: ${id}`);
      return renderExample(bundle.samples.find(sample => sample.id === id), bundle.run);
    });
    gallery.replaceChildren(...examples);
    status.textContent = "";
  } catch (error) {
    status.textContent = "Examples could not be loaded. The recorded JSON links below remain available.";
  }
}

initialize();

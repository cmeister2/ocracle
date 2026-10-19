//! Configuration integrity checks that do not require neural model downloads.

use std::fs;

use ocracle::CardOcr;
use sha2::{Digest, Sha256};

#[test]
fn rejects_swapped_character_indices_before_loading_registered_weights() {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join("models")).unwrap();
    let config = br#"{"alphabet":"ab","input_height":64,"model_name":"fixture","language":"fr"}"#;
    let config_path = root.path().join("models/config.json");
    fs::write(&config_path, config).unwrap();
    let registry = serde_json::json!({
        "schema_version": 1,
        "profiles": [{
            "id": "fr",
            "language": "fr",
            "model": "models/absent.rten",
            "configuration": "models/config.json",
            "sha256": "not-loaded-in-this-test",
            "configuration_sha256": Sha256::digest(config).iter()
                .map(|byte| format!("{byte:02x}")).collect::<String>()
        }]
    });
    fs::write(
        root.path().join("models/profiles.json"),
        serde_json::to_vec(&registry).unwrap(),
    )
    .unwrap();

    let valid_config_error = CardOcr::load_profile("fr", root.path(), None)
        .err()
        .unwrap();
    assert!(
        valid_config_error
            .to_string()
            .contains("read recognition model")
    );

    // Same class count and language, but different class meanings. A weight
    // digest alone would not detect this otherwise plausible configuration.
    fs::write(
        &config_path,
        String::from_utf8_lossy(config).replace("\"ab\"", "\"ba\""),
    )
    .unwrap();
    let altered_config_error = CardOcr::load_profile("fr", root.path(), None)
        .err()
        .unwrap();
    assert!(
        altered_config_error
            .to_string()
            .contains("configuration SHA-256 mismatch")
    );
}

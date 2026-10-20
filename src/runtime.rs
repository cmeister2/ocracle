//! Float-tensor inference shared by the default and optional CPU runtimes.

use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use rten::{Dimension, Model, RunOptions, ThreadPool};
use rten_tensor::{NdTensorView, Tensor};

use crate::InferenceBackend;

/// Validated inference settings shared by an engine's detector and recognizer.
pub(crate) struct RuntimeOptions {
    /// Selected model execution backend.
    pub(crate) backend: InferenceBackend,
    /// Optional worker count for inference.
    pub(crate) threads: Option<usize>,
    /// Explicit RTen pool reused by every model in this engine.
    rten_pool: Option<Arc<ThreadPool>>,
    /// Caller-owned native environment, bypassing the compatibility cache.
    #[cfg(all(feature = "onnxruntime", not(target_arch = "wasm32")))]
    pub(crate) ort_environment: Option<ort::environment::Environment>,
}

impl RuntimeOptions {
    /// Validate worker settings and create a private RTen pool when requested.
    ///
    /// `None` preserves each backend's existing default thread policy.
    /// No process environment variables are changed.
    ///
    /// # Errors
    ///
    /// Returns an error unless a supplied count is in `1..=i32::MAX`.
    pub(crate) fn new(backend: InferenceBackend, threads: Option<usize>) -> Result<Self> {
        ensure!(
            threads.is_none_or(|count| count > 0 && i32::try_from(count).is_ok()),
            "runtime thread count must be in 1..=i32::MAX"
        );
        let rten_pool = if backend == InferenceBackend::Rten {
            threads.map(|count| Arc::new(ThreadPool::with_num_threads(count)))
        } else {
            None
        };
        Ok(Self {
            backend,
            threads,
            rten_pool,
            #[cfg(all(feature = "onnxruntime", not(target_arch = "wasm32")))]
            ort_environment: None,
        })
    }
}

/// A reusable model with a single float input and output.
pub(crate) enum RuntimeModel {
    /// Portable Rust inference using the default or an explicitly shared pool.
    Rten {
        /// Loaded neural network.
        model: Box<Model>,
        /// Engine-owned pool, or `None` to retain the process-global policy.
        thread_pool: Option<Arc<ThreadPool>>,
    },
    /// Optional CPU session, serialized because ORT requires mutable session runs.
    #[cfg(all(feature = "onnxruntime", not(target_arch = "wasm32")))]
    OnnxRuntime(Box<std::sync::Mutex<ort::session::Session>>),
}

impl RuntimeModel {
    /// Load one model with the explicitly selected inference backend.
    ///
    /// Models loaded with the same settings share their explicit RTen pool.
    ///
    /// # Errors
    ///
    /// Returns errors for unavailable optional runtimes,
    /// missing native libraries, malformed models or unsupported model I/O.
    pub(crate) fn load(bytes: Vec<u8>, options: &RuntimeOptions) -> Result<Self> {
        match options.backend {
            InferenceBackend::Rten => {
                let model = Model::load(bytes).context("load RTen model")?;
                ensure!(
                    model.input_ids().len() == 1 && model.output_ids().len() == 1,
                    "runtime model must have exactly one input and output"
                );
                Ok(Self::Rten {
                    model: Box::new(model),
                    thread_pool: options.rten_pool.clone(),
                })
            }
            InferenceBackend::OnnxRuntime => {
                #[cfg(all(feature = "onnxruntime", not(target_arch = "wasm32")))]
                {
                    let environment = match &options.ort_environment {
                        Some(environment) => environment.clone(),
                        None => default_onnx_environment()?,
                    };
                    load_onnx(bytes, options.threads, &environment)
                }
                #[cfg(not(all(feature = "onnxruntime", not(target_arch = "wasm32"))))]
                {
                    anyhow::bail!(
                        "ONNX Runtime requires the `onnxruntime` Cargo feature and a native target"
                    )
                }
            }
        }
    }

    /// Return fixed dimensions and `None` for each symbolic input dimension.
    ///
    /// # Errors
    ///
    /// Returns errors for missing input shape metadata, non-tensor input types
    /// or a poisoned session lock.
    pub(crate) fn input_dimensions(&self) -> Result<Vec<Option<usize>>> {
        match self {
            Self::Rten { model, .. } => Ok(model
                .input_shape(0)
                .context("model has no input shape")?
                .into_iter()
                .map(|dimension| match dimension {
                    Dimension::Fixed(size) => Some(size),
                    Dimension::Symbolic(_) => None,
                })
                .collect()),
            #[cfg(all(feature = "onnxruntime", not(target_arch = "wasm32")))]
            Self::OnnxRuntime(session) => {
                let session = session
                    .lock()
                    .map_err(|_| anyhow::anyhow!("ONNX Runtime session lock poisoned"))?;
                let input = session.inputs().first().context("model has no input")?;
                let ort::value::ValueType::Tensor { shape, .. } = input.dtype() else {
                    anyhow::bail!("model input must be a tensor");
                };
                Ok(shape
                    .iter()
                    .map(|&size| usize::try_from(size).ok())
                    .collect())
            }
        }
    }

    /// Run an NCHW float input and return an owned, dynamically ranked output.
    ///
    /// # Errors
    ///
    /// Returns runtime errors for incompatible inputs, inference failure or
    /// non-float output, and errors for a poisoned optional session lock.
    pub(crate) fn run(&self, input: NdTensorView<'_, f32, 4>) -> Result<Tensor<f32>> {
        match self {
            Self::Rten { model, thread_pool } => model
                .run_one(
                    input.into(),
                    thread_pool
                        .as_ref()
                        .map(|pool| RunOptions::default().with_thread_pool(Some(Arc::clone(pool)))),
                )?
                .try_into()
                .context("model must output a float tensor"),
            #[cfg(all(feature = "onnxruntime", not(target_arch = "wasm32")))]
            Self::OnnxRuntime(session) => {
                use rten_tensor::prelude::*;

                let data = input
                    .data()
                    .context("ONNX Runtime input must be contiguous")?;
                let tensor = ort::value::TensorRef::from_array_view((input.shape(), data))?;
                let mut session = session
                    .lock()
                    .map_err(|_| anyhow::anyhow!("ONNX Runtime session lock poisoned"))?;
                let outputs = session.run(ort::inputs![tensor])?;
                ensure!(outputs.len() == 1, "model must return one output");
                let (shape, values) = outputs[0]
                    .try_extract_tensor::<f32>()
                    .context("model must output a float tensor")?;
                let shape: Vec<usize> = shape
                    .iter()
                    .map(|&size| usize::try_from(size))
                    .collect::<std::result::Result<_, _>>()
                    .context("invalid output dimension")?;
                let elements = shape
                    .iter()
                    .try_fold(1_usize, |size, &dim| size.checked_mul(dim))
                    .context("output shape overflows")?;
                ensure!(elements == values.len(), "output shape and data disagree");
                // Own the output before releasing the session lock. Inputs are
                // borrowed without a copy; ORT output ownership cannot be moved
                // into RTen's allocation through the safe public APIs.
                Ok(Tensor::from_data(shape.as_slice(), values.to_vec()))
            }
        }
    }
}

/// Retain the default environment for callers using the backend-only loaders.
///
/// ORT allows only one environment to be created, even after its last handle is
/// dropped. Keep a successful initialization alive for repeated engine loads.
/// Explicit environment loaders bypass this cache and retain caller ownership.
#[cfg(all(feature = "onnxruntime", not(target_arch = "wasm32")))]
static DEFAULT_ONNX_ENVIRONMENT: std::sync::Mutex<Option<ort::environment::Environment>> =
    std::sync::Mutex::new(None);

/// Initialize or reuse the process-lifetime default native environment.
///
/// # Errors
///
/// Returns errors for a missing/incompatible library, a poisoned initialization
/// lock or an environment already created outside the default loader.
#[cfg(all(feature = "onnxruntime", not(target_arch = "wasm32")))]
fn default_onnx_environment() -> Result<ort::environment::Environment> {
    let mut cached = DEFAULT_ONNX_ENVIRONMENT
        .lock()
        .map_err(|_| anyhow::anyhow!("ONNX Runtime environment lock poisoned"))?;
    if let Some(environment) = &*cached {
        return Ok(environment.clone());
    }
    let library = std::env::var_os("ORT_DYLIB_PATH")
        .filter(|path| !path.is_empty())
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            if cfg!(target_os = "windows") {
                "onnxruntime.dll".into()
            } else if cfg!(any(target_os = "macos", target_os = "ios")) {
                "libonnxruntime.dylib".into()
            } else {
                "libonnxruntime.so".into()
            }
        });
    // Eagerly load via the fallible API so a missing library returns an error.
    // Cache only successes so a missing-library error does not prevent retries.
    let environment = ort::init_from(&library)
        .with_context(|| {
            format!(
                "load ONNX Runtime native library {}; set ORT_DYLIB_PATH to an installed library",
                library.display()
            )
        })?
        .build()
        .context(
            "create ONNX Runtime environment; if the application already created one, \
             use CardOcr::load_with_onnxruntime_environment or \
             CardOcr::load_profile_with_onnxruntime_environment",
        )?;
    *cached = Some(environment.clone());
    Ok(environment)
}

/// Create one independent CPU session in an existing native environment.
///
/// # Errors
///
/// Returns errors for an invalid ONNX graph, invalid session options or a model
/// that does not have one float input and output.
#[cfg(all(feature = "onnxruntime", not(target_arch = "wasm32")))]
fn load_onnx(
    bytes: Vec<u8>,
    threads: Option<usize>,
    environment: &ort::environment::Environment,
) -> Result<RuntimeModel> {
    use ort::{
        session::{Session, builder::GraphOptimizationLevel},
        value::{TensorElementType, ValueType},
    };

    let mut builder = Session::builder(environment)?
        .with_no_environment_execution_providers()
        .map_err(session_option_error)?
        .with_independent_thread_pool()
        .map_err(session_option_error)?
        .with_parallel_execution(false)
        .map_err(session_option_error)?
        .with_inter_threads(1)
        .map_err(session_option_error)?
        .with_intra_op_spinning(false)
        .map_err(session_option_error)?
        .with_inter_op_spinning(false)
        .map_err(session_option_error)?
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(session_option_error)?;
    if let Some(threads) = threads {
        builder = builder
            .with_intra_threads(threads)
            .map_err(session_option_error)?;
    }
    let session = builder
        .commit_from_memory(&bytes)
        .context("load ONNX Runtime model (ONNX format required)")?;
    ensure!(
        session.inputs().len() == 1 && session.outputs().len() == 1,
        "runtime model must have exactly one input and output"
    );
    for outlet in session.inputs().iter().chain(session.outputs()) {
        ensure!(
            matches!(
                outlet.dtype(),
                ValueType::Tensor {
                    ty: TensorElementType::Float32,
                    ..
                }
            ),
            "runtime model input and output must be float32 tensors"
        );
    }
    Ok(RuntimeModel::OnnxRuntime(Box::new(std::sync::Mutex::new(
        session,
    ))))
}

/// Retain an option diagnostic while dropping its non-Send recovery builder.
#[cfg(all(feature = "onnxruntime", not(target_arch = "wasm32")))]
fn session_option_error(error: ort::Error<ort::session::builder::SessionBuilder>) -> anyhow::Error {
    anyhow::anyhow!("configure ONNX Runtime CPU session: {error}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rten_tensor::{NdTensor, prelude::*};

    /// ONNX IR8/opset13 Identity graph with float32 shape [1,1,2,2].
    const IDENTITY_ONNX: &[u8] = b"\x08\x08\x3a\x68\x0a\x19\x0a\x05\x69\x6e\x70\x75\x74\x12\x06\x6f\x75\x74\x70\x75\x74\x22\x08\x49\x64\x65\x6e\x74\x69\x74\x79\x12\x08\x69\x64\x65\x6e\x74\x69\x74\x79\x5a\x1f\x0a\x05\x69\x6e\x70\x75\x74\x12\x16\x0a\x14\x08\x01\x12\x10\x0a\x02\x08\x01\x0a\x02\x08\x01\x0a\x02\x08\x02\x0a\x02\x08\x02\x62\x20\x0a\x06\x6f\x75\x74\x70\x75\x74\x12\x16\x0a\x14\x08\x01\x12\x10\x0a\x02\x08\x01\x0a\x02\x08\x01\x0a\x02\x08\x02\x0a\x02\x08\x02\x42\x04\x0a\x00\x10\x0d";

    /// Load the tiny graph using the same validated settings as public engines.
    ///
    /// # Errors
    ///
    /// Returns errors from worker validation or runtime initialization.
    fn load_identity(backend: InferenceBackend, threads: Option<usize>) -> Result<RuntimeModel> {
        RuntimeModel::load(
            IDENTITY_ONNX.to_vec(),
            &RuntimeOptions::new(backend, threads)?,
        )
    }

    #[test]
    fn rten_identity_preserves_shape_and_pixels() -> Result<()> {
        let model = load_identity(InferenceBackend::Rten, None)?;
        assert!(matches!(
            model,
            RuntimeModel::Rten {
                thread_pool: None,
                ..
            }
        ));
        assert_eq!(
            model.input_dimensions()?,
            [Some(1), Some(1), Some(2), Some(2)]
        );
        let input = NdTensor::from_data([1, 1, 2, 2], vec![-1., 0., 0.5, 1.]);
        let output = model.run(input.view())?;
        assert_eq!(output.shape(), [1, 1, 2, 2]);
        assert_eq!(output.into_data(), input.into_data());
        Ok(())
    }

    #[test]
    fn explicit_rten_pool_is_shared_reused_and_independent_of_other_engines() -> Result<()> {
        let options = RuntimeOptions::new(InferenceBackend::Rten, Some(2))?;
        let first = RuntimeModel::load(IDENTITY_ONNX.to_vec(), &options)?;
        let second = RuntimeModel::load(IDENTITY_ONNX.to_vec(), &options)?;
        let independent = load_identity(InferenceBackend::Rten, Some(1))?;
        let (
            RuntimeModel::Rten {
                thread_pool: Some(first_pool),
                ..
            },
            RuntimeModel::Rten {
                thread_pool: Some(second_pool),
                ..
            },
            RuntimeModel::Rten {
                thread_pool: Some(independent_pool),
                ..
            },
        ) = (&first, &second, &independent)
        else {
            panic!("explicit RTen settings must create owned inference pools");
        };
        assert!(Arc::ptr_eq(first_pool, second_pool));
        assert!(!Arc::ptr_eq(first_pool, independent_pool));
        drop(options);
        assert_eq!(Arc::strong_count(first_pool), 2);
        std::thread::scope(|scope| -> Result<()> {
            let workers: Vec<_> = [&first, &second, &independent]
                .into_iter()
                .enumerate()
                .map(|(index, model)| {
                    scope.spawn(move || -> Result<()> {
                        for offset in 0..3 {
                            let input = NdTensor::from_data(
                                [1, 1, 2, 2],
                                vec![index as f32, offset as f32, -1., 0.5],
                            );
                            let output = model.run(input.view())?;
                            assert_eq!(output.shape(), [1, 1, 2, 2]);
                            assert_eq!(output.into_data(), input.into_data());
                        }
                        Ok(())
                    })
                })
                .collect();
            for worker in workers {
                worker.join().unwrap()?;
            }
            Ok(())
        })
    }

    #[test]
    fn invalid_threads_fail_before_any_runtime_loading() {
        for backend in [InferenceBackend::Rten, InferenceBackend::OnnxRuntime] {
            for count in [0, i32::MAX as usize + 1] {
                let result = load_identity(backend, Some(count));
                assert!(result.err().unwrap().to_string().contains("thread count"));
            }
        }
    }

    #[cfg(not(feature = "onnxruntime"))]
    #[test]
    fn optional_runtime_requires_explicit_feature() {
        let result = load_identity(InferenceBackend::OnnxRuntime, None);
        assert!(result.err().unwrap().to_string().contains("Cargo feature"));
    }

    #[cfg(all(feature = "onnxruntime", not(target_arch = "wasm32")))]
    #[test]
    fn supplied_native_runtime_matches_rten_and_serializes_shared_calls() -> Result<()> {
        if std::env::var_os("ORT_DYLIB_PATH").is_none() {
            eprintln!("Native runtime test requires ORT_DYLIB_PATH; skipped");
            return Ok(());
        }
        // Concurrent first loads must initialize only one environment. Dropping
        // every model must still allow subsequent engines to reuse it.
        std::thread::scope(|scope| -> Result<()> {
            let workers: Vec<_> = (0..4)
                .map(|_| scope.spawn(|| load_identity(InferenceBackend::OnnxRuntime, Some(1))))
                .collect();
            for worker in workers {
                let model = worker.join().unwrap()?;
                let input = NdTensor::from_data([1, 1, 2, 2], vec![-1., 0., 0.5, 1.]);
                assert_eq!(model.run(input.view())?.into_data(), input.into_data());
            }
            Ok(())
        })?;
        let model = load_identity(InferenceBackend::OnnxRuntime, Some(1))?;
        assert_eq!(
            model.input_dimensions()?,
            [Some(1), Some(1), Some(2), Some(2)]
        );
        let model = std::sync::Arc::new(model);
        std::thread::scope(|scope| -> Result<()> {
            let workers: Vec<_> = (0..4)
                .map(|offset| {
                    let model = model.clone();
                    scope.spawn(move || -> Result<()> {
                        let input =
                            NdTensor::from_data([1, 1, 2, 2], vec![offset as f32, -1., 0.5, 1.]);
                        let output = model.run(input.view())?;
                        assert_eq!(output.shape(), [1, 1, 2, 2]);
                        assert_eq!(output.into_data(), input.into_data());
                        Ok(())
                    })
                })
                .collect();
            for worker in workers {
                worker.join().unwrap()?;
            }
            Ok(())
        })
    }

    #[cfg(all(feature = "onnxruntime", not(target_arch = "wasm32")))]
    #[test]
    fn missing_native_library_returns_error_in_fresh_process() -> Result<()> {
        if std::env::var_os("OCRACLE_MISSING_ORT_CHILD").is_some() {
            let result = load_identity(InferenceBackend::OnnxRuntime, Some(1));
            let error = result
                .err()
                .context("missing library unexpectedly loaded")?;
            ensure!(
                error.to_string().contains("set ORT_DYLIB_PATH"),
                "missing-library diagnostic lost its setup guidance"
            );
            return Ok(());
        }
        // ORT caches the first loaded native library globally. A subprocess
        // isolates the missing-library case without mutating this test process.
        let directory = tempfile::tempdir()?;
        let status = std::process::Command::new(std::env::current_exe()?)
            .args([
                "--exact",
                "runtime::tests::missing_native_library_returns_error_in_fresh_process",
                "--nocapture",
            ])
            .env("OCRACLE_MISSING_ORT_CHILD", "1")
            .env(
                "ORT_DYLIB_PATH",
                directory.path().join("absent-onnxruntime.so"),
            )
            .status()?;
        ensure!(status.success(), "missing-library subprocess test failed");
        Ok(())
    }
}

//! Headless GPU/readback test for the Vulkan legacy render-pass fallback.
//! Run with BLADE_VULKAN_FORCE_LEGACY_RENDER_PASS=1 to test on newer GPUs too.
use blade_graphics as gpu;
use std::sync::atomic::{AtomicBool, Ordering};

struct SmokeLogger;
static LOGGER: SmokeLogger = SmokeLogger;
static HAD_GPU_ERROR: AtomicBool = AtomicBool::new(false);

impl log::Log for SmokeLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Info
    }
    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!("{}: {}", record.level(), record.args());
            if record.level() == log::Level::Error {
                HAD_GPU_ERROR.store(true, Ordering::Relaxed);
            }
        }
    }
    fn flush(&self) {}
}

const SHADER: &str = r#"
@vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let positions = array(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    return vec4(positions[index], 0.25, 1.0);
}
struct Colors {
    @location(0) first: vec4<f32>,
    @location(1) second: vec4<f32>,
}
@fragment fn fs() -> Colors {
    return Colors(vec4(0.25, 0.5, 0.75, 1.0), vec4(0.75, 0.25, 0.5, 1.0));
}
"#;

fn target(
    context: &gpu::Context,
    format: gpu::TextureFormat,
    samples: u32,
    size: gpu::Extent,
) -> (gpu::Texture, gpu::TextureView) {
    let texture = context.create_texture(gpu::TextureDesc {
        name: "legacy smoke target",
        format,
        size,
        dimension: gpu::TextureDimension::D2,
        array_layer_count: 1,
        mip_level_count: 1,
        sample_count: samples,
        usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::COPY,
        external: None,
    });
    let view = context.create_texture_view(
        texture,
        gpu::TextureViewDesc {
            name: "legacy smoke view",
            format,
            dimension: gpu::ViewDimension::D2,
            subresources: &gpu::TextureSubresources::default(),
            aspects: None,
        },
    );
    (texture, view)
}

#[cfg_attr(test, test)]
fn main() {
    log::set_logger(&LOGGER).unwrap();
    log::set_max_level(log::LevelFilter::Info);
    let context = unsafe {
        gpu::Context::init(gpu::ContextDesc {
            // Also check device creation with presentation enabled, as in BladeRender.
            presentation: true,
            validation: true,
            ..Default::default()
        })
        .expect("Initializing Vulkan")
    };
    println!("Adapter: {:?}", context.device_information());
    let shader = context.create_shader(gpu::ShaderDesc { source: SHADER });
    let size = gpu::Extent {
        width: 16,
        height: 16,
        depth: 1,
    };
    let color_format = gpu::TextureFormat::Rgba32Float;
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "legacy smoke",
        buffer_count: 1,
        extra: Default::default(),
    });
    let readback = context.create_buffer(gpu::BufferDesc {
        name: "legacy smoke readback",
        size: 16 * 16 * 36,
        memory: gpu::Memory::Shared,
    });
    for samples in [1, 4] {
        for depth_format in [
            gpu::TextureFormat::Depth32Float,
            gpu::TextureFormat::Depth32FloatStencil8Uint,
        ] {
            let mut pipeline = context.create_render_pipeline(gpu::RenderPipelineDesc {
                name: "legacy smoke pipeline",
                data_layouts: &[],
                vertex: shader.at("vs"),
                vertex_fetches: &[],
                primitive: Default::default(),
                depth_stencil: Some(gpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: true,
                    depth_compare: gpu::CompareFunction::Always,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                fragment: Some(shader.at("fs")),
                color_targets: &[color_format.into(), color_format.into()],
                multisample_state: gpu::MultisampleState {
                    sample_count: samples,
                    ..Default::default()
                },
            });
            // Recreate views at the same size to exercise imageless framebuffer reuse.
            for _ in 0..2 {
                let color = target(&context, color_format, samples, size);
                let second_color = target(&context, color_format, samples, size);
                let depth = target(&context, depth_format, samples, size);
                let resolved_color = target(&context, color_format, 1, size);
                let second_resolved_color = target(&context, color_format, 1, size);
                let resolved_depth = target(&context, depth_format, 1, size);
                encoder.start();
                for texture in [
                    color.0,
                    second_color.0,
                    depth.0,
                    resolved_color.0,
                    second_resolved_color.0,
                    resolved_depth.0,
                ] {
                    encoder.init_texture(texture);
                }
                let color_finish = if samples == 1 {
                    gpu::FinishOp::Store
                } else {
                    gpu::FinishOp::ResolveTo {
                        view: resolved_color.1,
                        mode: gpu::ResolveMode::Average,
                        store_original: true,
                    }
                };
                let depth_finish = if samples == 1 {
                    gpu::FinishOp::Store
                } else {
                    gpu::FinishOp::ResolveTo {
                        view: resolved_depth.1,
                        mode: gpu::ResolveMode::Sample0,
                        store_original: true,
                    }
                };
                let second_color_finish = if samples == 1 {
                    gpu::FinishOp::Store
                } else {
                    gpu::FinishOp::ResolveTo {
                        view: second_resolved_color.1,
                        mode: gpu::ResolveMode::Average,
                        store_original: true,
                    }
                };
                let targets = [
                    gpu::RenderTarget {
                        view: color.1,
                        init_op: gpu::InitOp::Clear(gpu::TextureColor::TransparentBlack),
                        finish_op: color_finish,
                    },
                    gpu::RenderTarget {
                        view: second_color.1,
                        init_op: gpu::InitOp::Clear(gpu::TextureColor::TransparentBlack),
                        finish_op: second_color_finish,
                    },
                ];
                {
                    let mut pass = encoder.render(
                        "legacy draw",
                        gpu::RenderTargetSet {
                            colors: &targets,
                            depth_stencil: Some(gpu::DepthStencilRenderTarget {
                                view: depth.1,
                                depth_init_op: gpu::InitOp::Clear(1.0),
                                depth_finish_op: depth_finish,
                                stencil_init_op: gpu::InitOp::Clear(7),
                                stencil_finish_op: depth_finish,
                            }),
                        },
                    );
                    pass.with(&pipeline).draw(0, 3, 0, 1);
                }
                // A second compatible pass checks that LOAD preserves the output.
                let targets = targets.map(|target| gpu::RenderTarget {
                    init_op: gpu::InitOp::Load,
                    ..target
                });
                drop(encoder.render(
                    "legacy load",
                    gpu::RenderTargetSet {
                        colors: &targets,
                        depth_stencil: Some(gpu::DepthStencilRenderTarget {
                            view: depth.1,
                            depth_init_op: gpu::InitOp::Load,
                            depth_finish_op: depth_finish,
                            stencil_init_op: gpu::InitOp::Load,
                            stencil_finish_op: depth_finish,
                        }),
                    },
                ));
                let output = if samples == 1 {
                    color.0
                } else {
                    resolved_color.0
                };
                encoder.transfer("legacy readback").copy_texture_to_buffer(
                    output.into(),
                    readback.into(),
                    size.width * 16,
                    size,
                );
                let second_output = if samples == 1 {
                    second_color.0
                } else {
                    second_resolved_color.0
                };
                encoder
                    .transfer("legacy second color readback")
                    .copy_texture_to_buffer(
                        second_output.into(),
                        readback.at(16 * 16 * 16),
                        size.width * 16,
                        size,
                    );
                if depth_format == gpu::TextureFormat::Depth32Float {
                    let depth_output = if samples == 1 {
                        depth.0
                    } else {
                        resolved_depth.0
                    };
                    encoder
                        .transfer("legacy depth readback")
                        .copy_texture_to_buffer(
                            depth_output.into(),
                            readback.at(16 * 16 * 32),
                            size.width * 4,
                            size,
                        );
                }
                let point = context.submit(&mut encoder, None);
                assert!(context.wait_for(&point, 10_000), "GPU completion timed out");
                let pixels = unsafe {
                    std::slice::from_raw_parts(readback.data().cast::<f32>(), 16 * 16 * 8)
                };
                for (index, pixel) in pixels.chunks_exact(4).enumerate() {
                    let expected = if index < 16 * 16 {
                        [0.25, 0.5, 0.75, 1.0]
                    } else {
                        [0.75, 0.25, 0.5, 1.0]
                    };
                    for (value, expected) in pixel.iter().zip(expected) {
                        assert!(
                            (value - expected).abs() < 0.001,
                            "Readback mismatch: {pixel:?}"
                        );
                    }
                }
                if depth_format == gpu::TextureFormat::Depth32Float {
                    let depths = unsafe {
                        std::slice::from_raw_parts(
                            readback.data().add(16 * 16 * 32).cast::<f32>(),
                            16 * 16,
                        )
                    };
                    assert!(
                        depths.iter().all(|value| (value - 0.25).abs() < 0.001),
                        "Depth draw/resolve readback mismatch"
                    );
                }
                assert!(
                    !HAD_GPU_ERROR.load(Ordering::Relaxed),
                    "Vulkan reported an error"
                );
                for (texture, view) in [
                    color,
                    second_color,
                    depth,
                    resolved_color,
                    second_resolved_color,
                    resolved_depth,
                ] {
                    context.destroy_texture_view(view);
                    context.destroy_texture(texture);
                }
            }
            context.destroy_render_pipeline(&mut pipeline);
            println!("Passed {samples}x MSAA, {depth_format:?}, drawing + load + resolve + view recreation");
        }
    }
    context.destroy_command_encoder(&mut encoder);
    context.destroy_buffer(readback);
}

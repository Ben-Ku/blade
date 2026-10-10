//! Vulkan 1.2 fallback for devices without VK_KHR_dynamic_rendering.
//!
//! Framebuffers are imageless: cached objects never retain texture views, so
//! resizing/destroying targets does not leave cached framebuffers referencing them.
use ash::vk;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub(super) struct AttachmentImage {
    pub flags: vk::ImageCreateFlags,
    pub usage: vk::ImageUsageFlags,
    pub samples: vk::SampleCountFlags,
    pub size: [u32; 2],
    pub layers: u32,
    pub format: vk::Format,
}

impl AttachmentImage {
    pub fn swapchain(
        size: [u16; 2],
        format: crate::TextureFormat,
        usage: vk::ImageUsageFlags,
    ) -> Self {
        Self {
            flags: vk::ImageCreateFlags::empty(),
            usage,
            samples: vk::SampleCountFlags::TYPE_1,
            size: size.map(u32::from),
            layers: 1,
            format: super::map_texture_format(format),
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct Attachment {
    format: vk::Format,
    samples: vk::SampleCountFlags,
    load: vk::AttachmentLoadOp,
    store: vk::AttachmentStoreOp,
    stencil_load: vk::AttachmentLoadOp,
    stencil_store: vk::AttachmentStoreOp,
}

impl Attachment {
    fn unused(format: vk::Format, samples: vk::SampleCountFlags) -> Self {
        Self {
            format,
            samples,
            load: vk::AttachmentLoadOp::DONT_CARE,
            store: vk::AttachmentStoreOp::DONT_CARE,
            stencil_load: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store: vk::AttachmentStoreOp::DONT_CARE,
        }
    }
}

#[derive(Clone, Debug, Default, Hash, PartialEq, Eq)]
struct PassKey {
    attachments: Vec<Attachment>,
    colors: Vec<u32>,
    resolves: Vec<u32>,
    depth_stencil: Option<u32>,
    depth_stencil_resolve: Option<u32>,
    depth_mode: vk::ResolveModeFlags,
    stencil_mode: vk::ResolveModeFlags,
}

#[derive(Hash, PartialEq, Eq)]
struct FramebufferKey {
    pass: vk::RenderPass,
    images: Vec<AttachmentImage>,
    size: [u32; 2],
}

#[derive(Default)]
pub(super) struct Cache {
    passes: HashMap<PassKey, vk::RenderPass>,
    framebuffers: HashMap<FramebufferKey, vk::Framebuffer>,
}

fn reference(index: u32) -> vk::AttachmentReference2<'static> {
    vk::AttachmentReference2::default()
        .attachment(index)
        .layout(vk::ImageLayout::GENERAL)
}

impl Cache {
    fn pass(&mut self, device: &ash::Device, key: PassKey) -> vk::RenderPass {
        *self.passes.entry(key).or_insert_with_key(|key| {
            let attachments: Vec<_> = key
                .attachments
                .iter()
                .map(|at| {
                    vk::AttachmentDescription2::default()
                        .format(at.format)
                        .samples(at.samples)
                        .load_op(at.load)
                        .store_op(at.store)
                        .stencil_load_op(at.stencil_load)
                        .stencil_store_op(at.stencil_store)
                        .initial_layout(vk::ImageLayout::GENERAL)
                        .final_layout(vk::ImageLayout::GENERAL)
                })
                .collect();
            let colors: Vec<_> = key.colors.iter().copied().map(reference).collect();
            let resolves: Vec<_> = key.resolves.iter().copied().map(reference).collect();
            let depth = reference(key.depth_stencil.unwrap_or(vk::ATTACHMENT_UNUSED));
            let depth_resolve =
                reference(key.depth_stencil_resolve.unwrap_or(vk::ATTACHMENT_UNUSED));
            let mut resolve_info = vk::SubpassDescriptionDepthStencilResolve::default()
                .depth_resolve_mode(key.depth_mode)
                .stencil_resolve_mode(key.stencil_mode)
                .depth_stencil_resolve_attachment(&depth_resolve);
            let mut subpass = vk::SubpassDescription2::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&colors);
            if resolves
                .iter()
                .any(|at| at.attachment != vk::ATTACHMENT_UNUSED)
            {
                subpass = subpass.resolve_attachments(&resolves);
            }
            if key.depth_stencil.is_some() {
                subpass = subpass.depth_stencil_attachment(&depth);
            }
            if key.depth_stencil_resolve.is_some() {
                subpass = subpass.push_next(&mut resolve_info);
            }
            let subpasses = [subpass];
            let info = vk::RenderPassCreateInfo2::default()
                .attachments(&attachments)
                .subpasses(&subpasses);
            // Blade's begin_pass barrier handles dependencies between passes.
            unsafe {
                device
                    .create_render_pass2(&info, None)
                    .expect("Creating legacy render pass")
            }
        })
    }

    pub fn pipeline_pass(
        &mut self,
        device: &ash::Device,
        colors: &[vk::Format],
        depth_stencil: Option<vk::Format>,
        samples: vk::SampleCountFlags,
    ) -> vk::RenderPass {
        let mut key = PassKey::default();
        for &format in colors {
            key.colors.push(key.attachments.len() as u32);
            key.attachments.push(Attachment::unused(format, samples));
            key.resolves.push(vk::ATTACHMENT_UNUSED);
        }
        if let Some(format) = depth_stencil {
            key.depth_stencil = Some(key.attachments.len() as u32);
            key.attachments.push(Attachment::unused(format, samples));
        }
        // With one subpass, resolve references/modes and load/store ops do not
        // affect render-pass compatibility. This pass works with MSAA resolves too.
        self.pass(device, key)
    }

    pub unsafe fn begin(
        &mut self,
        device: &super::Device,
        command: vk::CommandBuffer,
        targets: &crate::RenderTargetSet<'_>,
        area: vk::Rect2D,
    ) {
        let mut key = PassKey::default();
        let mut images = Vec::new();
        let mut views = Vec::new();
        let mut clears = Vec::new();
        for rt in targets.colors {
            let info = super::command::map_render_target(rt.view, rt.init_op, rt.finish_op);
            key.colors.push(key.attachments.len() as u32);
            let mut at = Attachment::unused(rt.view.attachment.format, rt.view.attachment.samples);
            at.load = info.load_op;
            at.store = info.store_op;
            key.attachments.push(at);
            images.push(rt.view.attachment);
            views.push(rt.view.raw);
            clears.push(info.clear_value);
            let resolve = if let crate::FinishOp::ResolveTo { view, mode, .. } = rt.finish_op {
                let expected = color_resolve_mode(rt.view.format);
                assert_eq!(mode, expected, "Legacy color resolve uses average for float formats and sample zero for integers");
                let index = key.attachments.len() as u32;
                let mut at = Attachment::unused(view.attachment.format, view.attachment.samples);
                at.store = vk::AttachmentStoreOp::STORE;
                key.attachments.push(at);
                images.push(view.attachment);
                views.push(view.raw);
                clears.push(vk::ClearValue::default());
                index
            } else {
                vk::ATTACHMENT_UNUSED
            };
            key.resolves.push(resolve);
        }
        if let Some(rt) = targets.depth_stencil {
            let has_depth = rt.view.aspects.contains(crate::TexelAspects::DEPTH);
            let has_stencil = rt.view.aspects.contains(crate::TexelAspects::STENCIL);
            let depth =
                super::command::map_render_target(rt.view, rt.depth_init_op, rt.depth_finish_op);
            let stencil = super::command::map_render_target(
                rt.view,
                rt.stencil_init_op,
                rt.stencil_finish_op,
            );
            key.depth_stencil = Some(key.attachments.len() as u32);
            let mut at = Attachment::unused(rt.view.attachment.format, rt.view.attachment.samples);
            if has_depth {
                at.load = depth.load_op;
                at.store = depth.store_op;
            }
            if has_stencil {
                at.stencil_load = stencil.load_op;
                at.stencil_store = stencil.store_op;
            }
            key.attachments.push(at);
            images.push(rt.view.attachment);
            views.push(rt.view.raw);
            clears.push(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: depth.clear_value.depth_stencil.depth,
                    stencil: stencil.clear_value.depth_stencil.stencil,
                },
            });
            let depth_view = if has_depth {
                key.depth_mode = depth.resolve_mode;
                resolve_view(rt.depth_finish_op)
            } else {
                None
            };
            let stencil_view = if has_stencil {
                key.stencil_mode = stencil.resolve_mode;
                resolve_view(rt.stencil_finish_op)
            } else {
                None
            };
            if let Some(view) = depth_view.or(stencil_view) {
                if let (Some(d), Some(s)) = (depth_view, stencil_view) {
                    assert_eq!(
                        d.raw, s.raw,
                        "Legacy depth/stencil resolves must use the same image view"
                    );
                }
                assert!(
                    device.depth_resolve_modes.contains(key.depth_mode),
                    "Unsupported depth resolve mode"
                );
                assert!(
                    device.stencil_resolve_modes.contains(key.stencil_mode),
                    "Unsupported stencil resolve mode"
                );
                if has_depth && has_stencil && key.depth_mode != key.stencil_mode {
                    let one_none = key.depth_mode.is_empty() || key.stencil_mode.is_empty();
                    assert!(
                        if one_none {
                            device.independent_resolve_none
                        } else {
                            device.independent_resolve
                        },
                        "Device does not support independent depth/stencil resolves"
                    );
                }
                key.depth_stencil_resolve = Some(key.attachments.len() as u32);
                let mut at = Attachment::unused(view.attachment.format, view.attachment.samples);
                at.store = vk::AttachmentStoreOp::STORE;
                at.stencil_store = vk::AttachmentStoreOp::STORE;
                key.attachments.push(at);
                images.push(view.attachment);
                views.push(view.raw);
                clears.push(vk::ClearValue::default());
            }
        }
        let pass = self.pass(&device.core, key);
        let framebuffer_key = FramebufferKey {
            pass,
            images,
            size: [area.extent.width, area.extent.height],
        };
        let framebuffer = *self
            .framebuffers
            .entry(framebuffer_key)
            .or_insert_with_key(|key| {
                let formats: Vec<_> = key.images.iter().map(|image| [image.format]).collect();
                let image_infos: Vec<_> = key
                    .images
                    .iter()
                    .zip(&formats)
                    .map(|(image, formats)| {
                        vk::FramebufferAttachmentImageInfo::default()
                            .flags(image.flags)
                            .usage(image.usage)
                            .width(image.size[0])
                            .height(image.size[1])
                            .layer_count(image.layers)
                            .view_formats(formats)
                    })
                    .collect();
                let mut attachment_info = vk::FramebufferAttachmentsCreateInfo::default()
                    .attachment_image_infos(&image_infos);
                let mut info = vk::FramebufferCreateInfo::default()
                    .flags(vk::FramebufferCreateFlags::IMAGELESS)
                    .render_pass(pass)
                    .width(key.size[0])
                    .height(key.size[1])
                    .layers(1)
                    .push_next(&mut attachment_info);
                info.attachment_count = key.images.len() as u32;
                device
                    .core
                    .create_framebuffer(&info, None)
                    .expect("Creating imageless framebuffer")
            });
        let mut attachment_begin = vk::RenderPassAttachmentBeginInfo::default().attachments(&views);
        let info = vk::RenderPassBeginInfo::default()
            .render_pass(pass)
            .framebuffer(framebuffer)
            .render_area(area)
            .clear_values(&clears)
            .push_next(&mut attachment_begin);
        device
            .core
            .cmd_begin_render_pass(command, &info, vk::SubpassContents::INLINE);
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        for (_, framebuffer) in self.framebuffers.drain() {
            device.destroy_framebuffer(framebuffer, None);
        }
        for (_, pass) in self.passes.drain() {
            device.destroy_render_pass(pass, None);
        }
    }
}

fn resolve_view(op: crate::FinishOp) -> Option<super::TextureView> {
    match op {
        crate::FinishOp::ResolveTo { view, .. } => Some(view),
        _ => None,
    }
}

fn color_resolve_mode(format: crate::TextureFormat) -> crate::ResolveMode {
    match format {
        crate::TextureFormat::R8Uint
        | crate::TextureFormat::R32Uint
        | crate::TextureFormat::Rg32Uint
        | crate::TextureFormat::Rgba32Uint => crate::ResolveMode::Sample0,
        // Normalized formats also resolve by averaging; texel_backing() groups
        // them with integers for clear conversion, so it cannot be used here.
        _ => crate::ResolveMode::Average,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalized_and_float_color_resolves_average() {
        for format in [
            crate::TextureFormat::Rgba8Unorm,
            crate::TextureFormat::Rgba8Snorm,
            crate::TextureFormat::Bgra8UnormSrgb,
            crate::TextureFormat::Rgba16Float,
        ] {
            assert_eq!(color_resolve_mode(format), crate::ResolveMode::Average);
        }
    }

    #[test]
    fn integer_color_resolves_sample_zero() {
        for format in [
            crate::TextureFormat::R8Uint,
            crate::TextureFormat::Rgba32Uint,
        ] {
            assert_eq!(color_resolve_mode(format), crate::ResolveMode::Sample0);
        }
    }
}

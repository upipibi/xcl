//! GPU: a compute pipeline with static samplers, made by `compute_container` from the driver blob
//! another process kept on disk (`driver_cache`), samples through its own static samplers.
//!
//! Process 1 (`make`) builds the pipeline (point, linear and trilinear static samplers from S#
//! constants in the code, in the ABI root signature of `abi_rs_for`) into an empty
//! `XCL_DXIL_CACHE`, which keeps its blob. Process 2 (`load`) first makes `OTHERS` root signatures
//! with one other static sampler each, then gets the same pipeline: it is made from that blob, and
//! samples a 2x2 texture between texels with each sampler.

use super::*;
use crate::gpu::rdna2::encode;
use crate::gpu::recomp::pso::StaticSampler;
use crate::gpu::recomp::runner::{self, KernelState, TexDim, TestTexture};
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT;

const ROLE: &str = "XCL_TEST_SAMPLER_BLOB_ROLE";
const OTHERS: &str = "XCL_TEST_SAMPLER_BLOB_OTHERS";
const CLAMP: u32 = 2 | 2 << 3 | 2 << 6;
const POINT: [u32; 4] = [CLAMP, 0x00c0_0000, 0, 0];
/// MIN_MAG_LINEAR_MIP_POINT, clamp: the tone map LUT sampler of #505
const LINEAR: [u32; 4] = [CLAMP, 0x00c0_0000, 0x1450_0000, 0];
/// the same with a linear mip filter
const TRILINEAR: [u32; 4] = [CLAMP, 0x00c0_0000, 0x1850_0000, 0];
/// texel (x, y) of the 2x2 texture = (x, y, x + 2y, 1), sampled at (0.375, 0.375)
const AT_POINT: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
const FILTERED: [f32; 4] = [0.25, 0.25, 0.75, 1.0];

/// The `k`th other static sampler of the loading process: point, with a LOD bias of its own.
fn other(k: usize) -> StaticSampler {
    ([CLAMP, 0x00c0_0000, (k as u32 + 1) * 16, 0], false)
}

fn program() -> Vec<u8> {
    let mut code = Vec::new();
    let mut put = |n: &str, e: &str, f: &[(&str, u64)], l: Option<u32>| code.extend(encode(n, e, f, l).unwrap_or_else(|x| panic!("{n}: {x}")));
    // S# point s[24:27] (SSAMP 6), linear s[28:31] (7), trilinear s[32:35] (8); T# s[16:23] (SRSRC 4)
    for (base, s) in [(24u64, POINT), (28, LINEAR), (32, TRILINEAR)] {
        for (k, w) in s.into_iter().enumerate() {
            if w == 0 {
                put("S_MOV_B32", "ENC_SOP1", &[("SDST", base + k as u64), ("SSRC0", 128)], None);
            } else {
                put("S_MOV_B32", "ENC_SOP1", &[("SDST", base + k as u64), ("SSRC0", 255)], Some(w));
            }
        }
    }
    for (vdata, ssamp) in [(10u64, 6u64), (20, 7), (30, 8)] {
        put("IMAGE_SAMPLE_LZ", "ENC_MIMG", &[("VDATA", vdata), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", ssamp), ("DMASK", 0xF), ("DIM", 1)], None);
    }
    put("S_ENDPGM", "ENC_SOPP", &[], None);
    code
}

/// The kernel's DXIL and static samplers, through the translator `XCL_DIRECT_DXIL` selects.
fn translate(code: &[u8]) -> (Vec<u8>, Vec<StaticSampler>) {
    if runner::direct_dxil() {
        let k = runner::direct_kernel_full(code, true).expect("direct translation");
        return (k.dxil, k.static_samplers);
    }
    let t = crate::gpu::recomp::translate(code, &runner::options(crate::gpu::recomp::Mode::Kernel)).expect("translation");
    assert!(t.complete(), "unsupported instructions: {:?}", t.unsupported);
    let dxil = crate::gpu::recomp::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).expect("DXC");
    (dxil, t.static_samplers)
}

fn texture() -> TestTexture {
    let mut data = Vec::new();
    for y in 0..2 {
        for x in 0..2 {
            for v in [x as f32, y as f32, (x + 2 * y) as f32, 1.0] {
                data.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32G32B32A32_FLOAT, width: 2, height: 2, depth: 1, texel_bytes: 16, data, uav: false }
}

/// One child process: build or load the pipeline, sample, print one `SBLOB` line.
fn child(role: &str) {
    let others: usize = std::env::var(OTHERS).ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    if role == "make" {
        // the test's build is small and fast: keep its blob anyway
        driver_cache::force_fixture_retention();
    }
    let code = program();
    let (dxil, statics) = translate(&code);
    assert_eq!(statics, vec![(POINT, false), (LINEAR, false), (TRILINEAR, false)], "three static samplers from the code");
    let file = crate::gpu::recomp::dxc::cache_dir().join("driver").join(driver_cache::name_for_root_node(&dxil, 0, false, &statics, 0));
    let mut st = KernelState::default();
    st.sgpr[16..24].copy_from_slice(&[0, 77 << 20, 0, 9 << 28, 0, 0, 0, 0]);
    for r in st.vgpr.iter_mut() {
        r[0] = 0.375f32.to_bits();
        r[1] = 0.375f32.to_bits();
    }
    let mut reused = None;
    let r = runner::run_with_pipeline(&st, &[], &[texture()], &[], |dev, _| unsafe {
        let raw = dev.as_raw();
        for k in 0..others {
            abi_rs_for(raw, &[other(k)]).ok_or("root signature with another static sampler")?;
        }
        let rs = abi_rs_for(raw, &statics).ok_or("ABI root signature with the static samplers")?;
        let create: PsoFn = std::mem::transmute(windows::core::Interface::vtable(dev).CreateComputePipelineState);
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC::default();
        let iid = pc_iid(Kind::PipelineState);
        let mut out = std::ptr::null_mut();
        let (hr, _, from_blob) = compute_container(raw, create, &desc, rs, &dxil, &statics,
            crate::gpu::shader::fnv64(&code), &iid, &mut out, ComputeBuild::Whole);
        if hr < 0 || out.is_null() {
            return Err(format!("compute_container: {hr:#x}"));
        }
        reused = Some(from_blob);
        let rs = ID3D12RootSignature::from_raw_borrowed(&rs).ok_or("root signature")?.clone();
        Ok((ID3D12PipelineState::from_raw(out), rs))
    })
    .expect("GPU run");
    let lane0 = |at: usize| -> [f32; 4] { std::array::from_fn(|c| f32::from_bits(r.vgpr[0][at + c])) };
    for l in 0..r.vgpr.len() {
        let lane = |at: usize| -> [f32; 4] { std::array::from_fn(|c| f32::from_bits(r.vgpr[l][at + c])) };
        assert_eq!([lane(10), lane(20), lane(30)], [lane0(10), lane0(20), lane0(30)], "lane {l} differs from lane 0");
    }
    println!(
        "SBLOB role={role} others={others} from_blob={} disk_hits={} file={} point={:?} linear={:?} trilinear={:?}",
        reused.unwrap(),
        driver_cache::counts().0,
        file.exists(),
        lane0(10),
        lane0(20),
        lane0(30)
    );
}

/// What one child printed.
struct Seen {
    from_blob: bool,
    disk_hits: u64,
    file: bool,
    samples: [[f32; 4]; 3],
}

fn spawn(test: &str, role: &str, cache: &std::path::Path, others: usize, disk: bool) -> Seen {
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &format!("hle::d3d12x::real::static_sampler_blob_tests::{test}"), "--nocapture", "--test-threads=1"])
        .env(ROLE, role)
        .env(OTHERS, others.to_string())
        .env("XCL_DXIL_CACHE", cache)
        .env("XCL_PSO_DISK_CACHE", if disk { "1" } else { "0" })
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{role} process failed ({:?}):\nstdout:\n{stdout}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let line = stdout.lines().find_map(|l| l.find("SBLOB ").map(|at| &l[at..])).unwrap_or_else(|| panic!("no SBLOB line:\n{stdout}"));
    println!("{line}");
    let field = |name: &str| -> &str {
        let at = line.find(&format!(" {name}=")).unwrap_or_else(|| panic!("{name} in {line}")) + name.len() + 2;
        let rest = &line[at..];
        if rest.starts_with('[') { &rest[..rest.find(']').unwrap() + 1] } else { rest.split(' ').next().unwrap() }
    };
    let floats = |name: &str| -> [f32; 4] {
        let v: Vec<f32> = field(name).trim_matches(['[', ']']).split(',').map(|x| x.trim().parse().unwrap()).collect();
        v.try_into().unwrap()
    };
    Seen {
        from_blob: field("from_blob") == "true",
        disk_hits: field("disk_hits").parse().unwrap(),
        file: field("file") == "true",
        samples: [floats("point"), floats("linear"), floats("trilinear")],
    }
}

fn close(a: [f32; 4], b: [f32; 4]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1.0 / 512.0)
}

/// Parent: skip without the hardware, then the two processes over one private cache folder;
/// `load_disk` false: process 2 has no disk cache and builds the pipeline itself.
fn across_processes(test: &str, others: usize, load_disk: bool) {
    if let Ok(role) = std::env::var(ROLE) {
        return child(&role);
    }
    match (runner::adapter(), runner::wave64_programs_supported()) {
        (Ok(_), Ok(true)) => (),
        (Ok(_), Ok(false)) => return runner::skip_test("no wave64"),
        _ => return runner::skip_test("no GPU"),
    }
    if !runner::direct_dxil() && crate::gpu::recomp::dxc::location().is_err() {
        return runner::skip_test("no DXC");
    }
    let cache = std::env::temp_dir().join(format!("xcl-sampler-blob-{}-{others}", std::process::id()));
    let _ = std::fs::remove_dir_all(&cache);
    std::fs::create_dir_all(&cache).unwrap();
    let made = spawn(test, "make", &cache, 0, true);
    let loaded = spawn(test, "load", &cache, others, load_disk);
    let _ = std::fs::remove_dir_all(&cache);
    let expected = [AT_POINT, FILTERED, FILTERED];
    assert!(!made.from_blob && made.disk_hits == 0, "process 1 builds the pipeline itself");
    assert!(made.file, "process 1 keeps its driver blob on disk");
    assert!(made.samples.iter().zip(expected).all(|(&a, b)| close(a, b)), "process 1 samples {:?}, expected {expected:?}", made.samples);
    if load_disk {
        assert!(loaded.from_blob && loaded.disk_hits == 1, "process 2 makes the pipeline from process 1's blob");
    } else {
        assert!(!loaded.from_blob && loaded.disk_hits == 0, "process 2 builds the pipeline itself");
    }
    assert!(
        loaded.samples.iter().zip(expected).all(|(&a, b)| close(a, b)),
        "process 2 ({others} other static samplers first, from_blob={}) samples {:?}, expected {expected:?}",
        loaded.from_blob,
        loaded.samples
    );
}

/// Control: the loading process has no static samplers before the pipeline's own.
#[test]
fn gpu_compute_static_samplers_from_another_process_blob() {
    across_processes("gpu_compute_static_samplers_from_another_process_blob", 0, true);
}

/// The loading process made 10 other static samplers first.
#[test]
fn gpu_compute_static_samplers_from_another_process_blob_after_other_samplers() {
    across_processes("gpu_compute_static_samplers_from_another_process_blob_after_other_samplers", 10, true);
}

/// Control: the same 10 other static samplers first, but no disk cache in the second process:
/// it builds the pipeline itself.
#[test]
fn gpu_compute_static_samplers_built_after_other_samplers() {
    across_processes("gpu_compute_static_samplers_built_after_other_samplers", 10, false);
}
